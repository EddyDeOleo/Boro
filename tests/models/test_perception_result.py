
import pytest

from src.models.evidence import (
    EntityObservation,
    Evidence,
    SpatialRelationship,
)
from src.models.observation_context import (
    CoverageScope,
    ObservationContext,
)
from src.models.perception_result import PerceptionResult
from src.models.region import (
    RegionObservation,
    RegionRelation,
)
from src.models.video_metadata import VideoMetadata


@pytest.fixture
def context():
    return ObservationContext(
        fps=30.0,
        frame_count=10,
        frame_start=0,
        frame_end=9,
        coverage_scope=CoverageScope.PARTIAL_SEGMENT,
    )


@pytest.fixture
def video_metadata():
    return VideoMetadata(
        width=1280,
        height=960,
        fps=30.0,
        frame_count=10,
    )


@pytest.fixture
def observations():
    return (
        EntityObservation(
            observation_id="obs_cube_0",
            entity_id="red_cube",
            frame_index=0,
            timestamp_seconds=0.0,
            centroid=(100.0, 200.0),
        ),
        EntityObservation(
            observation_id="obs_gripper_0",
            entity_id="gripper",
            frame_index=0,
            timestamp_seconds=0.0,
            centroid=(120.0, 200.0),
        ),
        EntityObservation(
            observation_id="obs_cube_1",
            entity_id="red_cube",
            frame_index=1,
            timestamp_seconds=1 / 30,
            centroid=(105.0, 205.0),
        ),
    )


@pytest.fixture
def spatial_relationships():
    return (
        SpatialRelationship(
            frame_index=0,
            entity_a_id="red_cube",
            entity_b_id="gripper",
            distance_pixels=20.0,
        ),
    )


@pytest.fixture
def region_observations():
    return (
        RegionObservation(
            entity_id="red_cube",
            region_id="target_zone",
            frame_index=0,
            relation=RegionRelation.OUTSIDE,
            centroid_inside=False,
            overlap_ratio=0.0,
        ),
    )


@pytest.fixture
def evidence():
    return (
        Evidence(
            evidence_id="evidence_001",
            evidence_type="MASK_TRACKING",
            description="Tracked object mask.",
            frame_start=0,
            frame_end=1,
            observation_ids=("obs_cube_0", "obs_cube_1"),
            entity_ids=("red_cube",),
        ),
    )


@pytest.fixture
def valid_result(
    context,
    video_metadata,
    observations,
    spatial_relationships,
    region_observations,
    evidence,
):
    return PerceptionResult(
        context=context,
        video_metadata=video_metadata,
        observations=observations,
        spatial_relationships=spatial_relationships,
        region_observations=region_observations,
        evidence=evidence,
        provider="meta",
        model="sam-3.1",
    )


def test_creates_valid_perception_result(valid_result):
    assert valid_result.provider == "meta"
    assert valid_result.model == "sam-3.1"
    assert len(valid_result.observations) == 3
    assert len(valid_result.spatial_relationships) == 1
    assert len(valid_result.region_observations) == 1
    assert len(valid_result.evidence) == 1


def test_defaults_optional_collections_to_empty(
    context,
    video_metadata,
    observations,
):
    result = PerceptionResult(
        context=context,
        video_metadata=video_metadata,
        observations=observations,
        provider="meta",
        model="sam-3.1",
    )

    assert result.spatial_relationships == ()
    assert result.region_observations == ()
    assert result.evidence == ()


def test_normalizes_collections_to_tuples(
    context,
    video_metadata,
    observations,
):
    result = PerceptionResult(
        context=context,
        video_metadata=video_metadata,
        observations=list(observations),
        spatial_relationships=[],
        region_observations=[],
        evidence=[],
        provider="meta",
        model="sam-3.1",
    )

    assert isinstance(result.observations, tuple)
    assert isinstance(result.spatial_relationships, tuple)
    assert isinstance(result.region_observations, tuple)
    assert isinstance(result.evidence, tuple)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("context", None),
        ("video_metadata", None),
    ],
)
def test_rejects_invalid_required_models(
    context,
    video_metadata,
    observations,
    field_name,
    invalid_value,
):
    values = {
        "context": context,
        "video_metadata": video_metadata,
        "observations": observations,
        "provider": "meta",
        "model": "sam-3.1",
    }
    values[field_name] = invalid_value

    with pytest.raises(ValueError):
        PerceptionResult(**values)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("observations", "invalid"),
        ("observations", None),
        ("spatial_relationships", "invalid"),
        ("region_observations", "invalid"),
        ("evidence", "invalid"),
    ],
)
def test_rejects_invalid_collection_values(
    context,
    video_metadata,
    observations,
    field_name,
    invalid_value,
):
    values = {
        "context": context,
        "video_metadata": video_metadata,
        "observations": observations,
        "provider": "meta",
        "model": "sam-3.1",
    }
    values[field_name] = invalid_value

    with pytest.raises(ValueError):
        PerceptionResult(**values)


@pytest.mark.parametrize(
    ("field_name", "invalid_item"),
    [
        ("observations", "not-an-observation"),
        ("spatial_relationships", "not-a-relationship"),
        ("region_observations", "not-a-region-observation"),
        ("evidence", "not-evidence"),
    ],
)
def test_rejects_invalid_collection_items(
    context,
    video_metadata,
    observations,
    field_name,
    invalid_item,
):
    values = {
        "context": context,
        "video_metadata": video_metadata,
        "observations": observations,
        "provider": "meta",
        "model": "sam-3.1",
    }
    values[field_name] = (invalid_item,)

    with pytest.raises(ValueError):
        PerceptionResult(**values)


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("", "sam-3.1"),
        ("   ", "sam-3.1"),
        ("meta", ""),
        ("meta", "   "),
    ],
)
def test_rejects_empty_provider_or_model(
    context,
    video_metadata,
    observations,
    provider,
    model,
):
    with pytest.raises(ValueError):
        PerceptionResult(
            context=context,
            video_metadata=video_metadata,
            observations=observations,
            provider=provider,
            model=model,
        )


def test_rejects_duplicate_observation_ids(
    context,
    video_metadata,
):
    duplicate_observations = (
        EntityObservation(
            observation_id="duplicate",
            entity_id="red_cube",
            frame_index=0,
            timestamp_seconds=0.0,
        ),
        EntityObservation(
            observation_id="duplicate",
            entity_id="gripper",
            frame_index=1,
            timestamp_seconds=1 / 30,
        ),
    )

    with pytest.raises(ValueError, match="Observation IDs must be unique"):
        PerceptionResult(
            context=context,
            video_metadata=video_metadata,
            observations=duplicate_observations,
            provider="meta",
            model="sam-3.1",
        )


def test_rejects_relationship_without_first_entity_observation(
    context,
    video_metadata,
    observations,
):
    invalid_relationship = SpatialRelationship(
        frame_index=0,
        entity_a_id="unknown_entity",
        entity_b_id="gripper",
        distance_pixels=20.0,
    )

    with pytest.raises(ValueError, match="Spatial relationship"):
        PerceptionResult(
            context=context,
            video_metadata=video_metadata,
            observations=observations,
            spatial_relationships=(invalid_relationship,),
            provider="meta",
            model="sam-3.1",
        )


def test_rejects_relationship_without_second_entity_observation(
    context,
    video_metadata,
    observations,
):
    invalid_relationship = SpatialRelationship(
        frame_index=0,
        entity_a_id="red_cube",
        entity_b_id="unknown_entity",
        distance_pixels=20.0,
    )

    with pytest.raises(ValueError, match="Spatial relationship"):
        PerceptionResult(
            context=context,
            video_metadata=video_metadata,
            observations=observations,
            spatial_relationships=(invalid_relationship,),
            provider="meta",
            model="sam-3.1",
        )


def test_rejects_region_observation_without_entity_observation(
    context,
    video_metadata,
    observations,
):
    invalid_region_observation = RegionObservation(
        entity_id="unknown_entity",
        region_id="target_zone",
        frame_index=0,
        relation=RegionRelation.OUTSIDE,
    )

    with pytest.raises(ValueError, match="Region observation"):
        PerceptionResult(
            context=context,
            video_metadata=video_metadata,
            observations=observations,
            region_observations=(invalid_region_observation,),
            provider="meta",
            model="sam-3.1",
        )


def test_entity_observation_rejects_negative_frame_index():
    with pytest.raises(
        ValueError,
        match="frame_index cannot be negative",
    ):
        EntityObservation(
            observation_id="invalid_observation",
            entity_id="red_cube",
            frame_index=-1,
            timestamp_seconds=0.0,
        )


@pytest.mark.parametrize(
    "frame_index",
    [10],
)
def test_rejects_observation_outside_video_context(
    context,
    video_metadata,
    frame_index,
):
    invalid_observation = EntityObservation(
        observation_id="invalid_observation",
        entity_id="red_cube",
        frame_index=frame_index,
        timestamp_seconds=0.0,
    )

    with pytest.raises(ValueError, match="Observation frame index"):
        PerceptionResult(
            context=context,
            video_metadata=video_metadata,
            observations=(invalid_observation,),
            provider="meta",
            model="sam-3.1",
        )


def test_rejects_relationship_outside_video_context(
    context,
    video_metadata,
    observations,
):
    invalid_relationship = SpatialRelationship(
        frame_index=10,
        entity_a_id="red_cube",
        entity_b_id="gripper",
        distance_pixels=20.0,
    )

    with pytest.raises(ValueError, match="Spatial relationship frame index"):
        PerceptionResult(
            context=context,
            video_metadata=video_metadata,
            observations=observations,
            spatial_relationships=(invalid_relationship,),
            provider="meta",
            model="sam-3.1",
        )


def test_rejects_region_observation_outside_video_context(
    context,
    video_metadata,
    observations,
):
    invalid_region_observation = RegionObservation(
        entity_id="red_cube",
        region_id="target_zone",
        frame_index=10,
        relation=RegionRelation.OUTSIDE,
    )

    with pytest.raises(ValueError, match="Region observation frame index"):
        PerceptionResult(
            context=context,
            video_metadata=video_metadata,
            observations=observations,
            region_observations=(invalid_region_observation,),
            provider="meta",
            model="sam-3.1",
        )


def test_rejects_evidence_outside_video_context(
    context,
    video_metadata,
    observations,
):
    invalid_evidence = Evidence(
        evidence_id="invalid_evidence",
        evidence_type="MASK_TRACKING",
        description="Invalid frame range.",
        frame_start=0,
        frame_end=10,
    )

    with pytest.raises(ValueError, match="Evidence frame range"):
        PerceptionResult(
            context=context,
            video_metadata=video_metadata,
            observations=observations,
            evidence=(invalid_evidence,),
            provider="meta",
            model="sam-3.1",
        )


def test_allows_missing_observations_between_frames(
    context,
    video_metadata,
):
    sparse_observations = (
        EntityObservation(
            observation_id="obs_0",
            entity_id="red_cube",
            frame_index=0,
            timestamp_seconds=0.0,
        ),
        EntityObservation(
            observation_id="obs_5",
            entity_id="red_cube",
            frame_index=5,
            timestamp_seconds=5 / 30,
        ),
    )

    result = PerceptionResult(
        context=context,
        video_metadata=video_metadata,
        observations=sparse_observations,
        provider="meta",
        model="sam-3.1",
    )

    assert len(result.observations) == 2


def test_result_is_immutable(valid_result):
    with pytest.raises(AttributeError):
        valid_result.provider = "other"
