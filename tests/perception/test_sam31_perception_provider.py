from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.perception.sam31_perception_provider as provider_module
from src.models.observation_context import CoverageScope, ObservationContext
from src.models.perception_config import PerceptionConfig
from src.models.region import RegionDefinition, RegionType
from src.models.video_metadata import VideoMetadata
from src.perception.sam31_perception_provider import (
    SAM31PerceptionError,
    Sam31PerceptionProvider,
)


class CompletedOutcome:
    pass


class IncompleteOutcome:
    pass


class FakeSegmentationResult:
    def __init__(self, records, outcome=None):
        self.records = tuple(records)
        self.outcome = outcome or CompletedOutcome()


class FakeInference:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def infer(self, video_path, context, config, regions):
        self.calls.append(
            {
                "video_path": video_path,
                "context": context,
                "config": config,
                "regions": regions,
            }
        )

        if self.error is not None:
            raise self.error

        return self.result


class FakeVideoProcessor:
    metadata = VideoMetadata(
        width=4,
        height=4,
        fps=30.0,
        frame_count=3,
    )

    def __init__(self, video_path):
        self.video_path = Path(video_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "episode.mp4"
    path.write_bytes(b"fake video")
    return path


@pytest.fixture
def context():
    return ObservationContext(
        fps=30.0,
        frame_count=3,
        frame_start=0,
        frame_end=2,
        time_offset_seconds=1.5,
        coverage_scope=CoverageScope.PARTIAL_SEGMENT,
    )


@pytest.fixture
def config():
    return PerceptionConfig(
        provider="meta",
        model="sam-3.1",
        entity_id_map={
            "1": "red_cube",
            "2": "gripper",
        },
    )


@pytest.fixture
def records():
    return (
        make_mask_record("1", 0),
        make_mask_record("2", 0),
        make_mask_record("1", 1),
        make_mask_record("2", 1),
    )


@pytest.fixture
def install_provider_dependencies(monkeypatch):
    monkeypatch.setattr(
        provider_module,
        "VideoProcessor",
        FakeVideoProcessor,
    )

    monkeypatch.setattr(
        provider_module,
        "VideoSegmentationResult",
        FakeSegmentationResult,
    )

    def fake_reconstruct(record, frame_width, frame_height):
        raster = np.zeros((frame_height, frame_width), dtype=np.uint8)

        if record.object_id == "1":
            raster[0:2, 0:2] = 1
        elif record.object_id == "2":
            raster[2:4, 2:4] = 1
        elif record.object_id == "3":
            raster[0:2, 2:4] = 1

        return raster

    monkeypatch.setattr(
        provider_module,
        "reconstruct_frame_mask",
        fake_reconstruct,
    )


def make_mask_record(object_id, frame_index, confidence=0.95):
    identity = SimpleNamespace(
        media="video",
        frame_index=frame_index,
        object_id=object_id,
    )

    frame = SimpleNamespace(frame_index=frame_index)

    return SimpleNamespace(
        kind="mask",
        object_id=object_id,
        identity=identity,
        frame=frame,
        confidence=confidence,
        bounds=SimpleNamespace(
            left=0,
            top=0,
            right=2,
            bottom=2,
        ),
    )


def make_provider(records, inference_error=None):
    inference = FakeInference(
        result=FakeSegmentationResult(records),
        error=inference_error,
    )

    return Sam31PerceptionProvider(inference), inference


def test_perceive_returns_normalized_perception_result(
    video_file,
    context,
    config,
    records,
    install_provider_dependencies,
):
    provider, inference = make_provider(records)

    result = provider.perceive(
        video_path=video_file,
        context=context,
        config=config,
    )

    assert result.provider == "meta"
    assert result.model == "sam-3.1"
    assert result.context == context
    assert result.video_metadata == FakeVideoProcessor.metadata

    assert len(result.observations) == 4
    assert len(result.spatial_relationships) == 2
    assert result.region_observations == ()

    assert {
        observation.entity_id
        for observation in result.observations
    } == {"red_cube", "gripper"}

    assert len(inference.calls) == 1
    assert inference.calls[0]["video_path"] == video_file.resolve()


def test_perceive_applies_context_time_offset(
    video_file,
    context,
    config,
    records,
    install_provider_dependencies,
):
    provider, _ = make_provider(records)

    result = provider.perceive(
        video_path=video_file,
        context=context,
        config=config,
    )

    timestamps = {
        observation.frame_index: observation.timestamp_seconds
        for observation in result.observations
        if observation.entity_id == "red_cube"
    }

    assert timestamps[0] == pytest.approx(1.5)
    assert timestamps[1] == pytest.approx(1.5 + (1 / 30))


def test_perceive_preserves_sam_confidence(
    video_file,
    context,
    config,
    records,
    install_provider_dependencies,
):
    provider, _ = make_provider(records)

    result = provider.perceive(
        video_path=video_file,
        context=context,
        config=config,
    )

    assert all(
        observation.confidence == pytest.approx(0.95)
        for observation in result.observations
    )


def test_perceive_supports_three_entities_and_all_pairs(
    video_file,
    context,
    install_provider_dependencies,
):
    config = PerceptionConfig(
        provider="meta",
        model="sam-3.1",
        entity_id_map={
            "1": "red_cube",
            "2": "gripper",
            "3": "target_box",
        },
    )

    records = (
        make_mask_record("1", 0),
        make_mask_record("2", 0),
        make_mask_record("3", 0),
    )

    provider, _ = make_provider(records)

    result = provider.perceive(
        video_path=video_file,
        context=context,
        config=config,
    )

    assert len(result.observations) == 3
    assert len(result.spatial_relationships) == 3

    observed_pairs = {
        frozenset(
            (relationship.entity_a_id, relationship.entity_b_id)
        )
        for relationship in result.spatial_relationships
    }

    assert observed_pairs == {
        frozenset(("red_cube", "gripper")),
        frozenset(("red_cube", "target_box")),
        frozenset(("gripper", "target_box")),
    }


def test_perceive_can_disable_spatial_relationships(
    video_file,
    context,
    records,
    install_provider_dependencies,
):
    config = PerceptionConfig(
        provider="meta",
        model="sam-3.1",
        entity_id_map={
            "1": "red_cube",
            "2": "gripper",
        },
        include_spatial_relationships=False,
    )

    provider, _ = make_provider(records)

    result = provider.perceive(
        video_path=video_file,
        context=context,
        config=config,
    )

    assert result.observations
    assert result.spatial_relationships == ()


def test_perceive_builds_region_observations(
    video_file,
    context,
    config,
    records,
    install_provider_dependencies,
):
    region = RegionDefinition(
        region_id="target_zone",
        name="Target Zone",
        region_type=RegionType.TARGET_ZONE,
        polygon=(
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 3.0),
            (0.0, 3.0),
        ),
        frame_width=4,
        frame_height=4,
        reference_frame=0,
    )

    provider, _ = make_provider(records)

    result = provider.perceive(
        video_path=video_file,
        context=context,
        config=config,
        regions=(region,),
    )

    assert len(result.region_observations) == 4
    assert {
        observation.region_id
        for observation in result.region_observations
    } == {"target_zone"}


def test_perceive_can_disable_region_observations(
    video_file,
    context,
    records,
    install_provider_dependencies,
):
    config = PerceptionConfig(
        provider="meta",
        model="sam-3.1",
        entity_id_map={
            "1": "red_cube",
            "2": "gripper",
        },
        include_region_observations=False,
    )

    region = RegionDefinition(
        region_id="target_zone",
        name="Target Zone",
        region_type=RegionType.TARGET_ZONE,
        polygon=(
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 3.0),
            (0.0, 3.0),
        ),
        frame_width=4,
        frame_height=4,
        reference_frame=0,
    )

    provider, _ = make_provider(records)

    result = provider.perceive(
        video_path=video_file,
        context=context,
        config=config,
        regions=(region,),
    )

    assert result.region_observations == ()


def test_perceive_rejects_incomplete_sam_result(
    video_file,
    context,
    config,
    records,
    install_provider_dependencies,
):
    inference = FakeInference(
        result=FakeSegmentationResult(
            records,
            outcome=IncompleteOutcome(),
        )
    )

    provider = Sam31PerceptionProvider(inference)

    with pytest.raises(
        SAM31PerceptionError,
        match="incomplete segmentation result",
    ):
        provider.perceive(
            video_path=video_file,
            context=context,
            config=config,
        )


def test_perceive_rejects_unconfigured_object_id(
    video_file,
    context,
    config,
    install_provider_dependencies,
):
    provider, _ = make_provider(
        (make_mask_record("99", 0),)
    )

    with pytest.raises(
        SAM31PerceptionError,
        match="unconfigured object ID",
    ):
        provider.perceive(
            video_path=video_file,
            context=context,
            config=config,
        )


def test_perceive_rejects_frame_outside_context(
    video_file,
    context,
    config,
    install_provider_dependencies,
):
    provider, _ = make_provider(
        (make_mask_record("1", 3),)
    )

    with pytest.raises(
        SAM31PerceptionError,
        match="outside the requested range",
    ):
        provider.perceive(
            video_path=video_file,
            context=context,
            config=config,
        )


def test_perceive_rejects_conflicting_frame_references(
    video_file,
    context,
    config,
    install_provider_dependencies,
):
    record = make_mask_record("1", 0)
    record.frame = SimpleNamespace(frame_index=1)

    provider, _ = make_provider((record,))

    with pytest.raises(
        SAM31PerceptionError,
        match="conflicts with its identity",
    ):
        provider.perceive(
            video_path=video_file,
            context=context,
            config=config,
        )


def test_perceive_rejects_duplicate_masks_for_same_entity_and_frame(
    video_file,
    context,
    config,
    install_provider_dependencies,
):
    provider, _ = make_provider(
        (
            make_mask_record("1", 0),
            make_mask_record("1", 0),
        )
    )

    with pytest.raises(
        SAM31PerceptionError,
        match="Duplicate SAM mask",
    ):
        provider.perceive(
            video_path=video_file,
            context=context,
            config=config,
        )


def test_perceive_rejects_invalid_confidence(
    video_file,
    context,
    config,
    install_provider_dependencies,
):
    provider, _ = make_provider(
        (make_mask_record("1", 0, confidence=1.5),)
    )

    with pytest.raises(
        SAM31PerceptionError,
        match="confidence",
    ):
        provider.perceive(
            video_path=video_file,
            context=context,
            config=config,
        )


def test_perceive_rejects_empty_reconstructed_mask(
    video_file,
    context,
    config,
    install_provider_dependencies,
    monkeypatch,
):
    monkeypatch.setattr(
        provider_module,
        "reconstruct_frame_mask",
        lambda record, frame_width, frame_height: np.zeros(
            (frame_height, frame_width),
            dtype=np.uint8,
        ),
    )

    provider, _ = make_provider(
        (make_mask_record("1", 0),)
    )

    with pytest.raises(
        SAM31PerceptionError,
        match="empty mask",
    ):
        provider.perceive(
            video_path=video_file,
            context=context,
            config=config,
        )


def test_perceive_rejects_region_with_wrong_dimensions(
    video_file,
    context,
    config,
    records,
    install_provider_dependencies,
):
    region = RegionDefinition(
        region_id="wrong_size",
        name="Wrong Size",
        region_type=RegionType.TARGET_ZONE,
        polygon=(
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
        ),
        frame_width=8,
        frame_height=8,
        reference_frame=0,
    )

    provider, _ = make_provider(records)

    with pytest.raises(
        ValueError,
        match="dimensions do not match",
    ):
        provider.perceive(
            video_path=video_file,
            context=context,
            config=config,
            regions=(region,),
        )


def test_perceive_rejects_context_with_wrong_frame_count(
    video_file,
    config,
    records,
    install_provider_dependencies,
):
    wrong_context = ObservationContext(
        fps=30.0,
        frame_count=4,
        frame_start=0,
        frame_end=2,
    )

    provider, _ = make_provider(records)

    with pytest.raises(
        ValueError,
        match="frame_count does not match",
    ):
        provider.perceive(
            video_path=video_file,
            context=wrong_context,
            config=config,
        )


def test_perceive_rejects_context_with_wrong_fps(
    video_file,
    config,
    records,
    install_provider_dependencies,
):
    wrong_context = ObservationContext(
        fps=25.0,
        frame_count=3,
        frame_start=0,
        frame_end=2,
    )

    provider, _ = make_provider(records)

    with pytest.raises(
        ValueError,
        match="fps does not match",
    ):
        provider.perceive(
            video_path=video_file,
            context=wrong_context,
            config=config,
        )


def test_perceive_wraps_inference_errors(
    video_file,
    context,
    config,
    install_provider_dependencies,
):
    inference = FakeInference(error=RuntimeError("inference failed"))
    provider = Sam31PerceptionProvider(inference)

    with pytest.raises(
        SAM31PerceptionError,
        match="inference implementation failed",
    ):
        provider.perceive(
            video_path=video_file,
            context=context,
            config=config,
        )


def test_perceive_rejects_missing_video_file(
    tmp_path,
    context,
    config,
    install_provider_dependencies,
):
    provider, _ = make_provider(())

    with pytest.raises(FileNotFoundError):
        provider.perceive(
            video_path=tmp_path / "missing.mp4",
            context=context,
            config=config,
        )


def test_perceive_rejects_invalid_video_path_type(
    context,
    config,
    install_provider_dependencies,
):
    provider, _ = make_provider(())

    with pytest.raises(TypeError, match="video_path"):
        provider.perceive(
            video_path=123,
            context=context,
            config=config,
        )


def test_constructor_rejects_invalid_inference_dependency():
    with pytest.raises(TypeError, match="SAM31Inference"):
        Sam31PerceptionProvider(object())


def test_perceive_rejects_result_with_no_masks(
    video_file,
    context,
    config,
    install_provider_dependencies,
):
    provider, _ = make_provider(
        (
            SimpleNamespace(kind="text"),
            SimpleNamespace(kind="box"),
        )
    )

    with pytest.raises(
        SAM31PerceptionError,
        match="no usable masks",
    ):
        provider.perceive(
            video_path=video_file,
            context=context,
            config=config,
        )


def test_perceive_ignores_non_mask_records(
    video_file,
    context,
    config,
    install_provider_dependencies,
):
    records = (
        SimpleNamespace(kind="text"),
        make_mask_record("1", 0),
    )

    provider, _ = make_provider(records)

    result = provider.perceive(
        video_path=video_file,
        context=context,
        config=config,
    )

    assert len(result.observations) == 1
    assert result.observations[0].entity_id == "red_cube"


def test_perceive_rejects_region_reference_outside_video(
    video_file,
    context,
    config,
    records,
    install_provider_dependencies,
):
    region = RegionDefinition(
        region_id="invalid_reference",
        name="Invalid Reference",
        region_type=RegionType.TARGET_ZONE,
        polygon=(
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
        ),
        frame_width=4,
        frame_height=4,
        reference_frame=3,
    )

    provider, _ = make_provider(records)

    with pytest.raises(
        ValueError,
        match="reference frame exceeds",
    ):
        provider.perceive(
            video_path=video_file,
            context=context,
            config=config,
            regions=(region,),
        )