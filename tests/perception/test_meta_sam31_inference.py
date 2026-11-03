from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.perception.meta_sam31_client import MetaSAM31Client
from src.perception.meta_sam31_inference import (
    MetaSAM31Inference,
    SAM31BoxPrompt,
)


@pytest.fixture
def meta_client():
    client = Mock()
    client.files = Mock()
    client.files.create = Mock()
    client.responses = Mock()
    client.responses.stream = Mock()

    return MetaSAM31Client(client=client)


@pytest.fixture
def initial_boxes():
    return {
        "1": SAM31BoxPrompt(
            x1=775,
            y1=744,
            x2=817,
            y2=788,
        ),
        "2": SAM31BoxPrompt(
            x1=700,
            y1=660,
            x2=910,
            y2=745,
        ),
    }


@pytest.fixture
def inference(meta_client, initial_boxes):
    return MetaSAM31Inference(
        client=meta_client,
        initial_boxes=initial_boxes,
    )


class TestSAM31BoxPrompt:

    def test_accepts_valid_coordinates(self):
        box = SAM31BoxPrompt(10, 20, 100, 200)

        assert box.x1 == 10
        assert box.y1 == 20
        assert box.x2 == 100
        assert box.y2 == 200

    @pytest.mark.parametrize(
        "coordinates",
        [
            (-1, 0, 10, 10),
            (0, -1, 10, 10),
            (0, 0, -1, 10),
            (0, 0, 10, -1),
        ],
    )
    def test_rejects_negative_coordinates(self, coordinates):
        with pytest.raises(ValueError):
            SAM31BoxPrompt(*coordinates)

    @pytest.mark.parametrize(
        "coordinates",
        [
            (10, 10, 5, 20),
            (10, 20, 20, 15),
        ],
    )
    def test_rejects_inverted_coordinates(self, coordinates):
        with pytest.raises(ValueError):
            SAM31BoxPrompt(*coordinates)

    @pytest.mark.parametrize(
        "coordinates",
        [
            (True, 0, 10, 10),
            (0, 1.5, 10, 10),
            (0, 0, "10", 10),
        ],
    )
    def test_rejects_non_integer_coordinates(self, coordinates):
        with pytest.raises(TypeError):
            SAM31BoxPrompt(*coordinates)

    def test_accepts_box_inside_video_dimensions(self):
        box = SAM31BoxPrompt(10, 20, 100, 200)

        box.validate_dimensions(width=1280, height=960)

    def test_rejects_box_exceeding_video_dimensions(self):
        box = SAM31BoxPrompt(10, 20, 1280, 200)

        with pytest.raises(ValueError):
            box.validate_dimensions(width=1280, height=960)


class TestMetaSAM31Inference:

    def test_builds_expected_dual_object_prompt(self, inference):
        prompt = inference._build_box_prompt(
            frame_index=0,
            width=1280,
            height=960,
        )

        expected = (
            "<0f>"
            "1<|box;x1=775;y1=744;x2=817;y2=788;w=1280;h=960|>,"
            "2<|box;x1=700;y1=660;x2=910;y2=745;w=1280;h=960|>"
        )

        assert prompt == expected

    def test_builds_prompt_using_requested_frame(self, inference):
        prompt = inference._build_box_prompt(
            frame_index=12,
            width=1280,
            height=960,
        )

        assert prompt.startswith("<12f>")

    def test_preserves_entity_prompt_order(self, inference):
        prompt = inference._build_box_prompt(
            frame_index=0,
            width=1280,
            height=960,
        )

        assert prompt.index("1<|box;") < prompt.index("2<|box;")

    def test_rejects_negative_frame_index(self, inference):
        with pytest.raises(ValueError):
            inference._build_box_prompt(
                frame_index=-1,
                width=1280,
                height=960,
            )

    def test_rejects_invalid_video_dimensions(self, inference):
        with pytest.raises(ValueError):
            inference._build_box_prompt(
                frame_index=0,
                width=0,
                height=960,
            )

    def test_rejects_boolean_frame_index(self, inference):
        with pytest.raises(ValueError):
            inference._build_box_prompt(
                frame_index=True,
                width=1280,
                height=960,
            )

    def test_rejects_box_outside_prompt_dimensions(self, inference):
        with pytest.raises(ValueError):
            inference._build_box_prompt(
                frame_index=0,
                width=800,
                height=960,
            )

    def test_rejects_empty_initial_boxes(self, meta_client):
        with pytest.raises(ValueError):
            MetaSAM31Inference(
                client=meta_client,
                initial_boxes={},
            )

    @pytest.mark.parametrize(
        "entity_id",
        [
            "cube",
            "object_1",
            "",
            "-1",
        ],
    )
    def test_rejects_non_numeric_entity_ids(
        self,
        meta_client,
        entity_id,
    ):
        with pytest.raises(ValueError):
            MetaSAM31Inference(
                client=meta_client,
                initial_boxes={
                    entity_id: SAM31BoxPrompt(10, 10, 20, 20),
                },
            )

    def test_rejects_invalid_box_type(self, meta_client):
        with pytest.raises(TypeError):
            MetaSAM31Inference(
                client=meta_client,
                initial_boxes={
                    "1": (10, 10, 20, 20),
                },
            )

    def test_rejects_invalid_client(self, initial_boxes):
        with pytest.raises(TypeError):
            MetaSAM31Inference(
                client=object(),
                initial_boxes=initial_boxes,
            )



from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.perception.meta_sam31_inference as inference_module
from src.models.observation_context import ObservationContext
from src.models.perception_config import PerceptionConfig


class FakeMetaSAM31Client:
    def __init__(self):
        self.segment_video = Mock()


class FakeVideoProcessor:
    def __init__(self, video_path, metadata):
        self.video_path = Path(video_path)
        self.metadata = metadata
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exited = True


@pytest.fixture
def inference_setup(monkeypatch, tmp_path):
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"fake video")

    metadata = SimpleNamespace(
        width=1280,
        height=960,
        fps=30.0,
        frame_count=51,
    )

    fake_client = FakeMetaSAM31Client()
    fake_video_processor = FakeVideoProcessor(
        video_path,
        metadata,
    )

    monkeypatch.setattr(
        inference_module,
        "MetaSAM31Client",
        FakeMetaSAM31Client,
    )

    monkeypatch.setattr(
        inference_module,
        "VideoProcessor",
        lambda path: fake_video_processor,
    )

    inference = inference_module.MetaSAM31Inference(
        client=fake_client,
        initial_boxes={
            "1": inference_module.SAM31BoxPrompt(
                x1=775,
                y1=744,
                x2=817,
                y2=788,
            ),
            "2": inference_module.SAM31BoxPrompt(
                x1=700,
                y1=660,
                x2=910,
                y2=745,
            ),
        },
    )

    context = ObservationContext(
        fps=30.0,
        frame_count=51,
        frame_start=0,
        frame_end=50,
    )

    config = PerceptionConfig(
        provider="meta",
        model="sam-3.1",
        entity_id_map={
            "1": "red_cube",
            "2": "gripper",
        },
    )

    return SimpleNamespace(
        video_path=video_path,
        metadata=metadata,
        client=fake_client,
        video_processor=fake_video_processor,
        inference=inference,
        context=context,
        config=config,
    )


def test_infer_builds_prompt_and_calls_client(
    inference_setup,
):
    setup = inference_setup
    expected_result = object()

    setup.client.segment_video.return_value = expected_result

    result = setup.inference.infer(
        video_path=setup.video_path,
        context=setup.context,
        config=setup.config,
        regions=(),
    )

    expected_prompt = (
        "<0f>"
        "1<|box;x1=775;y1=744;x2=817;y2=788;w=1280;h=960|>,"
        "2<|box;x1=700;y1=660;x2=910;y2=745;w=1280;h=960|>"
    )

    assert result is expected_result

    setup.client.segment_video.assert_called_once_with(
        video_path=setup.video_path.resolve(),
        prompt=expected_prompt,
    )

    assert setup.video_processor.entered
    assert setup.video_processor.exited


def test_infer_uses_context_start_frame_in_prompt(
    inference_setup,
):
    setup = inference_setup

    context = ObservationContext(
        fps=30.0,
        frame_count=51,
        frame_start=5,
        frame_end=50,
    )

    setup.inference.infer(
        video_path=setup.video_path,
        context=context,
        config=setup.config,
        regions=(),
    )

    actual_prompt = setup.client.segment_video.call_args.kwargs[
        "prompt"
    ]

    assert actual_prompt.startswith("<5f>")


def test_infer_rejects_non_path_video_argument(
    inference_setup,
):
    setup = inference_setup

    with pytest.raises(TypeError, match="video_path"):
        setup.inference.infer(
            video_path=123,
            context=setup.context,
            config=setup.config,
            regions=(),
        )

    setup.client.segment_video.assert_not_called()


def test_infer_rejects_missing_video(
    inference_setup,
    tmp_path,
):
    setup = inference_setup
    missing_path = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError):
        setup.inference.infer(
            video_path=missing_path,
            context=setup.context,
            config=setup.config,
            regions=(),
        )

    setup.client.segment_video.assert_not_called()


def test_infer_rejects_invalid_context(
    inference_setup,
):
    setup = inference_setup

    with pytest.raises(TypeError, match="ObservationContext"):
        setup.inference.infer(
            video_path=setup.video_path,
            context=None,
            config=setup.config,
            regions=(),
        )

    setup.client.segment_video.assert_not_called()


def test_infer_rejects_invalid_config(
    inference_setup,
):
    setup = inference_setup

    with pytest.raises(TypeError, match="PerceptionConfig"):
        setup.inference.infer(
            video_path=setup.video_path,
            context=setup.context,
            config=None,
            regions=(),
        )

    setup.client.segment_video.assert_not_called()


def test_infer_rejects_unsupported_model(
    inference_setup,
):
    setup = inference_setup

    config = PerceptionConfig(
        provider="meta",
        model="other-model",
        entity_id_map={
            "1": "red_cube",
            "2": "gripper",
        },
    )

    with pytest.raises(ValueError, match="sam-3.1"):
        setup.inference.infer(
            video_path=setup.video_path,
            context=setup.context,
            config=config,
            regions=(),
        )

    setup.client.segment_video.assert_not_called()


def test_infer_rejects_mismatched_entity_ids(
    inference_setup,
):
    setup = inference_setup

    config = PerceptionConfig(
        provider="meta",
        model="sam-3.1",
        entity_id_map={
            "1": "red_cube",
            "3": "gripper",
        },
    )

    with pytest.raises(ValueError, match="Initial box IDs"):
        setup.inference.infer(
            video_path=setup.video_path,
            context=setup.context,
            config=config,
            regions=(),
        )

    setup.client.segment_video.assert_not_called()


def test_infer_rejects_mismatched_video_frame_count(
    inference_setup,
):
    setup = inference_setup

    context = ObservationContext(
        fps=30.0,
        frame_count=50,
        frame_start=0,
        frame_end=49,
    )

    with pytest.raises(ValueError, match="frame count"):
        setup.inference.infer(
            video_path=setup.video_path,
            context=context,
            config=setup.config,
            regions=(),
        )

    setup.client.segment_video.assert_not_called()


def test_infer_requires_frame_range_to_reach_video_end(
    inference_setup,
):
    setup = inference_setup

    context = ObservationContext(
        fps=30.0,
        frame_count=51,
        frame_start=0,
        frame_end=49,
    )

    with pytest.raises(ValueError, match="final video frame"):
        setup.inference.infer(
            video_path=setup.video_path,
            context=context,
            config=setup.config,
            regions=(),
        )

    setup.client.segment_video.assert_not_called()


def test_infer_rejects_box_outside_video_dimensions(
    monkeypatch,
    inference_setup,
):
    setup = inference_setup

    inference = inference_module.MetaSAM31Inference(
        client=setup.client,
        initial_boxes={
            "1": inference_module.SAM31BoxPrompt(
                x1=775,
                y1=744,
                x2=1280,
                y2=788,
            ),
            "2": inference_module.SAM31BoxPrompt(
                x1=700,
                y1=660,
                x2=910,
                y2=745,
            ),
        },
    )

    with pytest.raises(ValueError, match="exceed video dimensions"):
        inference.infer(
            video_path=setup.video_path,
            context=setup.context,
            config=setup.config,
            regions=(),
        )

    setup.client.segment_video.assert_not_called()
