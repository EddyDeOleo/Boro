
import pytest

from src.models.evidence import (
    EntityObservation,
    Evidence,
    SpatialRelationship,
)


def test_entity_observation_creation():
    observation = EntityObservation(
        observation_id="obs_001",
        entity_id="cube",
        frame_index=10,
        timestamp_seconds=0.333,
        bbox=(100.0, 200.0, 150.0, 250.0),
        centroid=(125.0, 225.0),
        area_pixels=2500.0,
        confidence=0.98,
        source="sam-3.1",
    )

    assert observation.entity_id == "cube"
    assert observation.frame_index == 10
    assert observation.confidence == 0.98


def test_entity_observation_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        EntityObservation(
            observation_id="obs_001",
            entity_id="cube",
            frame_index=0,
            timestamp_seconds=0.0,
            confidence=1.5,
        )


def test_entity_observation_rejects_invalid_bbox():
    with pytest.raises(ValueError):
        EntityObservation(
            observation_id="obs_001",
            entity_id="cube",
            frame_index=0,
            timestamp_seconds=0.0,
            bbox=(100.0, 200.0, 50.0, 250.0),
        )


def test_spatial_relationship_creation():
    relationship = SpatialRelationship(
        frame_index=10,
        entity_a_id="cube",
        entity_b_id="gripper",
        distance_pixels=12.5,
        delta_x_pixels=8.0,
        delta_y_pixels=-9.0,
        intersection_area_pixels=0.0,
        iou=0.0,
    )

    assert relationship.distance_pixels == 12.5
    assert relationship.entity_a_id == "cube"


def test_spatial_relationship_rejects_same_entity():
    with pytest.raises(ValueError):
        SpatialRelationship(
            frame_index=10,
            entity_a_id="cube",
            entity_b_id="cube",
        )


def test_spatial_relationship_rejects_invalid_iou():
    with pytest.raises(ValueError):
        SpatialRelationship(
            frame_index=10,
            entity_a_id="cube",
            entity_b_id="gripper",
            iou=1.5,
        )


def test_evidence_creation():
    evidence = Evidence(
        evidence_id="evidence_001",
        evidence_type="COORDINATED_MOVEMENT",
        description="Two entities show coordinated movement.",
        frame_start=10,
        frame_end=20,
        observation_ids=("obs_001", "obs_002"),
        entity_ids=("cube", "gripper"),
        metrics={"relative_displacement_pixels": 4.2},
        limitations=("Visual evidence does not prove physical grasp.",),
    )

    assert evidence.frame_start == 10
    assert evidence.frame_end == 20
    assert len(evidence.observation_ids) == 2


def test_evidence_rejects_invalid_frame_range():
    with pytest.raises(ValueError):
        Evidence(
            evidence_id="evidence_001",
            evidence_type="MOVEMENT",
            description="Movement evidence.",
            frame_start=20,
            frame_end=10,
        )


def test_evidence_rejects_duplicate_observation_ids():
    with pytest.raises(ValueError):
        Evidence(
            evidence_id="evidence_001",
            evidence_type="MOVEMENT",
            description="Movement evidence.",
            frame_start=10,
            frame_end=20,
            observation_ids=("obs_001", "obs_001"),
        )


def test_entity_observation_rejects_nan_timestamp():
    with pytest.raises(ValueError):
        EntityObservation(
            observation_id="obs_001",
            entity_id="cube",
            frame_index=0,
            timestamp_seconds=float("nan"),
        )


def test_entity_observation_rejects_infinite_centroid():
    with pytest.raises(ValueError):
        EntityObservation(
            observation_id="obs_001",
            entity_id="cube",
            frame_index=0,
            timestamp_seconds=0.0,
            centroid=(float("inf"), 10.0),
        )


def test_entity_attributes_are_immutable():
    attributes = {"area": 1200.0}

    observation = EntityObservation(
        observation_id="obs_001",
        entity_id="cube",
        frame_index=0,
        timestamp_seconds=0.0,
        attributes=attributes,
    )

    attributes["area"] = 9999.0

    assert observation.attributes["area"] == 1200.0

    with pytest.raises(TypeError):
        observation.attributes["area"] = 500.0


def test_evidence_metrics_are_immutable():
    metrics = {"distance": 10.0}

    evidence = Evidence(
        evidence_id="evidence_001",
        evidence_type="MOVEMENT",
        description="Movement evidence.",
        frame_start=0,
        frame_end=10,
        metrics=metrics,
    )

    with pytest.raises(TypeError):
        evidence.metrics["distance"] = 20.0


def test_spatial_relationship_rejects_infinite_distance():
    with pytest.raises(ValueError):
        SpatialRelationship(
            frame_index=0,
            entity_a_id="cube",
            entity_b_id="gripper",
            distance_pixels=float("inf"),
        )


def test_evidence_rejects_nan_metric():
    with pytest.raises(ValueError):
        Evidence(
            evidence_id="evidence_001",
            evidence_type="MOVEMENT",
            description="Movement evidence.",
            frame_start=0,
            frame_end=10,
            metrics={"distance": float("nan")},
        )
