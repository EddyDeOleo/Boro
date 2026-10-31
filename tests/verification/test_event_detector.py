
"""Tests for generic temporal event detection."""

from types import MappingProxyType

import pytest

from src.models.event import EventStatus
from src.models.event_detection_config import EventDetectionConfig
from src.models.evidence import EntityObservation, SpatialRelationship
from src.models.observation_context import (
    CoverageScope,
    ObservationContext,
)
from src.models.region import RegionObservation, RegionRelation
from src.models.task_roles import TaskRoles
from src.verification.event_detector import EventDetector


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def detector():
    return EventDetector()


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
def task_roles():
    return TaskRoles(
        entity_roles={
            "object": "cube",
            "effector": "gripper",
        },
        region_roles={
            "target_region": "target_zone",
        },
    )


# ============================================================
# Test data factories
# ============================================================


def make_entity_observation(
    entity_id="cube",
    frame_index=0,
    observation_id=None,
    centroid=None,
):
    if observation_id is None:
        observation_id = f"{entity_id}:{frame_index}"

    if centroid is None:
        centroid = (100.0 + frame_index, 100.0)

    return EntityObservation(
        observation_id=observation_id,
        entity_id=entity_id,
        frame_index=frame_index,
        timestamp_seconds=frame_index / 30.0,
        bbox=(90.0, 90.0, 110.0, 110.0),
        centroid=centroid,
        area_pixels=400.0,
        confidence=0.95,
        attributes={},
        source="test",
    )


def make_spatial_relationship(
    entity_a_id="cube",
    entity_b_id="gripper",
    frame_index=0,
    distance_pixels=5.0,
):
    return SpatialRelationship(
        frame_index=frame_index,
        entity_a_id=entity_a_id,
        entity_b_id=entity_b_id,
        distance_pixels=distance_pixels,
        delta_x_pixels=2.0,
        delta_y_pixels=3.0,
        intersection_area_pixels=0.0,
        iou=0.0,
        overlap_ratio_a=0.0,
        overlap_ratio_b=0.0,
        attributes={"centroid_distance_pixels": 10.0},
    )


def make_region_observation(
    entity_id="cube",
    region_id="target_zone",
    frame_index=0,
    relation=None,
    overlap_ratio=0.0,
):
    """Build a region observation with configurable mask overlap."""

    if relation is None:
        if overlap_ratio <= 0.0:
            relation = RegionRelation.OUTSIDE
        elif overlap_ratio >= 1.0:
            relation = RegionRelation.INSIDE
        else:
            relation = RegionRelation.PARTIALLY_INTERSECTING

    return RegionObservation(
        entity_id=entity_id,
        region_id=region_id,
        frame_index=frame_index,
        relation=relation,
        centroid_inside=overlap_ratio > 0.0,
        intersection_area_pixels=overlap_ratio * 400.0,
        overlap_ratio=overlap_ratio,
    )


def make_coordinated_observations(
    frames=range(10),
    object_motion=(2.0, 0.0),
    effector_motion=(2.0, 0.0),
    mask_distance=5.0,
):
    """Build aligned observations for an object and an effector."""

    entity_observations = []
    relationships = []

    for frame in frames:
        object_observation = make_entity_observation(
            entity_id="cube",
            frame_index=frame,
            centroid=(
                100.0 + frame * object_motion[0],
                100.0 + frame * object_motion[1],
            ),
        )

        effector_observation = make_entity_observation(
            entity_id="gripper",
            frame_index=frame,
            centroid=(
                100.0 + frame * effector_motion[0],
                100.0 + frame * effector_motion[1],
            ),
        )

        entity_observations.extend(
            [
                object_observation,
                effector_observation,
            ]
        )

        relationships.append(
            make_spatial_relationship(
                frame_index=frame,
                distance_pixels=mask_distance,
            )
        )

    return entity_observations, relationships


def make_separation_observations(
    frames=range(10),
    initial_object=(100.0, 100.0),
    initial_effector=(100.0, 100.0),
    object_motion=(2.0, 0.0),
    effector_motion=(0.0, 0.0),
):
    """Build paired observations with configurable relative motion.

    The centroid distance is calculated from the two entity tracks.
    SpatialRelationship is deliberately not required for this helper.
    """

    observations = []

    for frame in frames:
        object_observation = make_entity_observation(
            entity_id="cube",
            frame_index=frame,
            centroid=(
                initial_object[0] + frame * object_motion[0],
                initial_object[1] + frame * object_motion[1],
            ),
        )

        effector_observation = make_entity_observation(
            entity_id="gripper",
            frame_index=frame,
            centroid=(
                initial_effector[0] + frame * effector_motion[0],
                initial_effector[1] + frame * effector_motion[1],
            ),
        )

        observations.extend(
            [
                object_observation,
                effector_observation,
            ]
        )

    return observations


def get_events_by_type(events, event_type):
    return [
        event
        for event in events
        if event.event_type == event_type
    ]


# ============================================================
# Initialization
# ============================================================


class TestEventDetectorInitialization:

    def test_create_detector_with_default_config(self):
        detector = EventDetector()

        assert detector.config is not None

    def test_create_detector_with_custom_config(self):
        config = EventDetectionConfig(min_observations=5)
        detector = EventDetector(config=config)

        assert detector.config is config

    def test_invalid_config_type_raises_type_error(self):
        with pytest.raises(
            TypeError,
            match="config must be an EventDetectionConfig",
        ):
            EventDetector(config="invalid")


# ============================================================
# Input validation
# ============================================================


class TestEventDetectorInputValidation:

    def test_rejects_entity_observations_as_non_sequence(
        self,
        detector,
        context,
        task_roles,
    ):
        with pytest.raises(
            TypeError,
            match="entity_observations must be a sequence",
        ):
            detector.detect(
                None,
                [],
                [],
                task_roles,
                context,
            )

    def test_rejects_string_as_observation_sequence(
        self,
        detector,
        context,
        task_roles,
    ):
        with pytest.raises(
            TypeError,
            match="entity_observations must be a sequence",
        ):
            detector.detect(
                "invalid",
                [],
                [],
                task_roles,
                context,
            )

    def test_rejects_invalid_entity_observation_type(
        self,
        detector,
        context,
        task_roles,
    ):
        with pytest.raises(
            TypeError,
            match=r"entity_observations\[0\] must be an instance of EntityObservation",
        ):
            detector.detect(
                ["invalid"],
                [],
                [],
                task_roles,
                context,
            )

    def test_rejects_invalid_spatial_relationship_type(
        self,
        detector,
        context,
        task_roles,
    ):
        with pytest.raises(
            TypeError,
            match=r"spatial_relationships\[0\] must be an instance of SpatialRelationship",
        ):
            detector.detect(
                [],
                ["invalid"],
                [],
                task_roles,
                context,
            )

    def test_rejects_invalid_region_observation_type(
        self,
        detector,
        context,
        task_roles,
    ):
        with pytest.raises(
            TypeError,
            match=r"region_observations\[0\] must be an instance of RegionObservation",
        ):
            detector.detect(
                [],
                [],
                ["invalid"],
                task_roles,
                context,
            )

    def test_rejects_invalid_task_roles(
        self,
        detector,
        context,
    ):
        with pytest.raises(
            TypeError,
            match="task_roles must be a TaskRoles instance",
        ):
            detector.detect([], [], [], "invalid", context)

    def test_rejects_invalid_context(
        self,
        detector,
        task_roles,
    ):
        with pytest.raises(
            TypeError,
            match="context must be an ObservationContext instance",
        ):
            detector.detect([], [], [], task_roles, "invalid")

    def test_empty_inputs_return_empty_tuple(
        self,
        detector,
        context,
        task_roles,
    ):
        result = detector.detect(
            [],
            [],
            [],
            task_roles,
            context,
        )

        assert result == ()
        assert isinstance(result, tuple)


# ============================================================
# Entity observation validation
# ============================================================


class TestEntityObservationValidation:

    def test_rejects_observation_before_context_start(
        self,
        detector,
        task_roles,
    ):
        observation = make_entity_observation(frame_index=0)

        partial_context = ObservationContext(
            fps=30.0,
            frame_count=10,
            frame_start=2,
            frame_end=9,
        )

        with pytest.raises(
            ValueError,
            match="outside the observed interval",
        ):
            detector.detect(
                [observation],
                [],
                [],
                task_roles,
                partial_context,
            )

    def test_rejects_observation_after_context_end(
        self,
        detector,
        context,
        task_roles,
    ):
        observation = make_entity_observation(frame_index=10)

        with pytest.raises(
            ValueError,
            match="outside the observed interval",
        ):
            detector.detect(
                [observation],
                [],
                [],
                task_roles,
                context,
            )

    def test_rejects_duplicate_observation_id(
        self,
        detector,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation(
                entity_id="cube",
                frame_index=0,
                observation_id="duplicate",
            ),
            make_entity_observation(
                entity_id="gripper",
                frame_index=1,
                observation_id="duplicate",
            ),
        ]

        with pytest.raises(
            ValueError,
            match="Duplicate observation_id",
        ):
            detector.detect(
                observations,
                [],
                [],
                task_roles,
                context,
            )

    def test_rejects_duplicate_entity_in_same_frame(
        self,
        detector,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation(
                frame_index=2,
                observation_id="cube:2:first",
            ),
            make_entity_observation(
                frame_index=2,
                observation_id="cube:2:second",
            ),
        ]

        with pytest.raises(
            ValueError,
            match="Duplicate entity observation",
        ):
            detector.detect(
                observations,
                [],
                [],
                task_roles,
                context,
            )

    def test_allows_same_entity_in_different_frames(
        self,
        detector,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation(frame_index=1),
            make_entity_observation(frame_index=3),
            make_entity_observation(frame_index=5),
        ]

        prepared = detector._prepare_observations(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert len(prepared.entity_observations["cube"]) == 3

    def test_allows_missing_frames_between_observations(
        self,
        detector,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation(frame_index=1),
            make_entity_observation(frame_index=4),
        ]

        prepared = detector._prepare_observations(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert [
            item.frame_index
            for item in prepared.entity_observations["cube"]
        ] == [1, 4]


# ============================================================
# Spatial relationship validation
# ============================================================


class TestSpatialRelationshipValidation:

    def test_rejects_unknown_first_entity(
        self,
        detector,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation(entity_id="cube"),
        ]

        relationships = [
            make_spatial_relationship(entity_a_id="unknown"),
        ]

        with pytest.raises(
            ValueError,
            match="references unknown entity: 'unknown'",
        ):
            detector.detect(
                observations,
                relationships,
                [],
                task_roles,
                context,
            )

    def test_rejects_unknown_second_entity(
        self,
        detector,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation(entity_id="cube"),
        ]

        relationships = [
            make_spatial_relationship(entity_b_id="unknown"),
        ]

        with pytest.raises(
            ValueError,
            match="references unknown entity: 'unknown'",
        ):
            detector.detect(
                observations,
                relationships,
                [],
                task_roles,
                context,
            )

    def test_model_rejects_relationship_with_same_entity(self):
        with pytest.raises(
            ValueError,
            match="An entity cannot have a relationship with itself",
        ):
            make_spatial_relationship(
                entity_a_id="cube",
                entity_b_id="cube",
            )

    def test_rejects_duplicate_relationship_in_same_frame(
        self,
        detector,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation("cube", 0),
            make_entity_observation("gripper", 0),
        ]

        relationships = [
            make_spatial_relationship(
                entity_a_id="cube",
                entity_b_id="gripper",
                frame_index=2,
            ),
            make_spatial_relationship(
                entity_a_id="gripper",
                entity_b_id="cube",
                frame_index=2,
            ),
        ]

        with pytest.raises(
            ValueError,
            match="Duplicate spatial relationship",
        ):
            detector.detect(
                observations,
                relationships,
                [],
                task_roles,
                context,
            )

    def test_rejects_relationship_outside_context(
        self,
        detector,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation("cube", 0),
            make_entity_observation("gripper", 0),
        ]

        relationships = [
            make_spatial_relationship(frame_index=10),
        ]

        with pytest.raises(
            ValueError,
            match="SpatialRelationship frame_index",
        ):
            detector.detect(
                observations,
                relationships,
                [],
                task_roles,
                context,
            )


# ============================================================
# Region observation validation
# ============================================================


class TestRegionObservationValidation:

    def test_rejects_region_for_unknown_entity(
        self,
        detector,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation(entity_id="cube"),
        ]

        regions = [
            make_region_observation(entity_id="unknown"),
        ]

        with pytest.raises(
            ValueError,
            match="RegionObservation references unknown entity",
        ):
            detector.detect(
                observations,
                [],
                regions,
                task_roles,
                context,
            )

    def test_rejects_duplicate_region_observation(
        self,
        detector,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation(entity_id="cube"),
        ]

        regions = [
            make_region_observation(frame_index=2),
            make_region_observation(frame_index=2),
        ]

        with pytest.raises(
            ValueError,
            match="Duplicate region observation",
        ):
            detector.detect(
                observations,
                [],
                regions,
                task_roles,
                context,
            )

    def test_rejects_region_observation_outside_context(
        self,
        detector,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation(entity_id="cube"),
        ]

        regions = [
            make_region_observation(frame_index=10),
        ]

        with pytest.raises(
            ValueError,
            match="RegionObservation frame_index",
        ):
            detector.detect(
                observations,
                [],
                regions,
                task_roles,
                context,
            )


# ============================================================
# Observation preparation
# ============================================================


class TestObservationPreparation:

    def test_groups_entity_observations_by_entity_id(
        self,
        detector,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation("cube", 2),
            make_entity_observation("gripper", 1),
            make_entity_observation("cube", 0),
        ]

        prepared = detector._prepare_observations(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert set(prepared.entity_observations) == {
            "cube",
            "gripper",
        }

        assert [
            item.frame_index
            for item in prepared.entity_observations["cube"]
        ] == [0, 2]

    def test_sorts_entity_observations_chronologically(
        self,
        detector,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation(frame_index=5),
            make_entity_observation(frame_index=1),
            make_entity_observation(frame_index=3),
        ]

        prepared = detector._prepare_observations(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert [
            item.frame_index
            for item in prepared.entity_observations["cube"]
        ] == [1, 3, 5]

    def test_groups_spatial_relationships_by_entity_pair(
        self,
        detector,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation("cube", 0),
            make_entity_observation("gripper", 0),
        ]

        relationships = [
            make_spatial_relationship(frame_index=2),
            make_spatial_relationship(frame_index=0),
        ]

        prepared = detector._prepare_observations(
            observations,
            relationships,
            [],
            task_roles,
            context,
        )

        assert ("cube", "gripper") in prepared.spatial_relationships

        assert [
            item.frame_index
            for item in prepared.spatial_relationships[
                ("cube", "gripper")
            ]
        ] == [0, 2]

    def test_groups_region_observations_by_entity_and_region(
        self,
        detector,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation("cube", 0),
        ]

        regions = [
            make_region_observation(frame_index=2),
            make_region_observation(frame_index=0),
        ]

        prepared = detector._prepare_observations(
            observations,
            [],
            regions,
            task_roles,
            context,
        )

        key = ("cube", "target_zone")

        assert [
            item.frame_index
            for item in prepared.region_observations[key]
        ] == [0, 2]

    def test_prepared_groups_are_immutable(
        self,
        detector,
        context,
        task_roles,
    ):
        prepared = detector._prepare_observations(
            [make_entity_observation(frame_index=0)],
            [],
            [],
            task_roles,
            context,
        )

        assert isinstance(
            prepared.entity_observations,
            MappingProxyType,
        )

        with pytest.raises(TypeError):
            prepared.entity_observations["new_entity"] = ()


# ============================================================
# OBJECT_MOVING
# ============================================================


class TestObjectMovingDetection:

    def test_detects_object_moving_when_displacement_exceeds_threshold(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector()

        observations = [
            make_entity_observation(
                frame_index=frame,
                centroid=(100.0 + frame * 2.0, 100.0),
            )
            for frame in range(10)
        ]

        events = detector.detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert len(events) == 1

        event = events[0]

        assert event.event_type == "OBJECT_MOVING"
        assert event.status == EventStatus.DETECTED
        assert event.entity_ids == ("cube",)
        assert event.frame_start == 0
        assert event.frame_end == 9
        assert event.confidence is not None
        assert len(event.evidence) == 1

        evidence = event.evidence[0]

        assert len(evidence.observation_ids) == 10
        assert evidence.metrics["displacement_pixels"] == pytest.approx(
            18.0
        )

    def test_does_not_detect_movement_below_threshold(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector(
            EventDetectionConfig(
                min_object_displacement_pixels=20.0,
            )
        )

        observations = [
            make_entity_observation(
                frame_index=frame,
                centroid=(100.0 + frame * 0.5, 100.0),
            )
            for frame in range(10)
        ]

        events = detector.detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(events, "OBJECT_MOVING") == []

    def test_does_not_detect_with_insufficient_observations(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector(
            EventDetectionConfig(min_observations=5)
        )

        observations = [
            make_entity_observation(
                frame_index=frame,
                centroid=(100.0 + frame * 10.0, 100.0),
            )
            for frame in range(4)
        ]

        events = detector.detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(events, "OBJECT_MOVING") == []

    def test_does_not_detect_below_minimum_duration(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector(
            EventDetectionConfig(
                min_event_duration_seconds=1.0,
                min_object_displacement_pixels=1.0,
            )
        )

        observations = [
            make_entity_observation(
                frame_index=frame,
                centroid=(100.0 + frame * 10.0, 100.0),
            )
            for frame in range(10)
        ]

        events = detector.detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(events, "OBJECT_MOVING") == []

    def test_splits_sequences_when_frame_gap_exceeds_configuration(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector(
            EventDetectionConfig(
                max_frame_gap=1,
                min_observations=3,
                min_event_duration_seconds=0.05,
                min_object_displacement_pixels=2.0,
            )
        )

        observations = [
            make_entity_observation(
                frame_index=frame,
                centroid=(100.0 + index * 3.0, 100.0),
            )
            for index, frame in enumerate([0, 1, 2, 6, 7, 8])
        ]

        events = get_events_by_type(
            detector.detect(
                observations,
                [],
                [],
                task_roles,
                context,
            ),
            "OBJECT_MOVING",
        )

        assert len(events) == 2
        assert (events[0].frame_start, events[0].frame_end) == (0, 2)
        assert (events[1].frame_start, events[1].frame_end) == (6, 8)

    def test_allows_gap_within_configured_limit(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector(
            EventDetectionConfig(
                max_frame_gap=1,
                min_observations=3,
                min_event_duration_seconds=0.05,
                min_object_displacement_pixels=2.0,
            )
        )

        observations = [
            make_entity_observation(
                frame_index=frame,
                centroid=(100.0 + index * 3.0, 100.0),
            )
            for index, frame in enumerate([0, 1, 3])
        ]

        events = get_events_by_type(
            detector.detect(
                observations,
                [],
                [],
                task_roles,
                context,
            ),
            "OBJECT_MOVING",
        )

        assert len(events) == 1
        assert events[0].frame_start == 0
        assert events[0].frame_end == 3

    def test_does_not_detect_non_object_entities(self, context):
        observations = [
            make_entity_observation(
                entity_id="gripper",
                frame_index=frame,
                centroid=(100.0 + frame * 10.0, 100.0),
            )
            for frame in range(10)
        ]

        roles = TaskRoles(
            entity_roles={"effector": "gripper"},
            region_roles={},
        )

        events = EventDetector().detect(
            observations,
            [],
            [],
            roles,
            context,
        )

        assert get_events_by_type(events, "OBJECT_MOVING") == []

    def test_evidence_contains_detector_limitations(
        self,
        context,
        task_roles,
    ):
        observations = [
            make_entity_observation(
                frame_index=frame,
                centroid=(100.0 + frame * 2.0, 100.0),
            )
            for frame in range(10)
        ]

        events = EventDetector().detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        limitations = events[0].evidence[0].limitations

        assert any(
            "physical-world distance" in item
            for item in limitations
        )
        assert any(
            "cause" in item
            for item in limitations
        )


# ============================================================
# OBJECT_MOVING_WITH_EFFECTOR
# ============================================================


class TestObjectMovingWithEffectorDetection:

    def test_detects_coordinated_movement(
        self,
        context,
        task_roles,
    ):
        observations, relationships = make_coordinated_observations()

        events = EventDetector().detect(
            observations,
            relationships,
            [],
            task_roles,
            context,
        )

        coordinated_events = get_events_by_type(
            events,
            "OBJECT_MOVING_WITH_EFFECTOR",
        )

        assert len(coordinated_events) == 1

        event = coordinated_events[0]

        assert event.status == EventStatus.DETECTED
        assert event.entity_ids == ("cube", "gripper")
        assert event.frame_start == 0
        assert event.frame_end == 9
        assert event.confidence is not None

        evidence = event.evidence[0]

        assert evidence.evidence_type == "COORDINATED_ENTITY_MOTION"
        assert evidence.entity_ids == ("cube", "gripper")
        assert len(evidence.observation_ids) == 20
        assert evidence.metrics["mean_motion_similarity"] == pytest.approx(1.0)
        assert evidence.metrics["object_displacement_pixels"] == pytest.approx(18.0)
        assert evidence.metrics["effector_displacement_pixels"] == pytest.approx(18.0)

    def test_does_not_detect_opposite_directions(
        self,
        context,
        task_roles,
    ):
        observations, relationships = make_coordinated_observations(
            object_motion=(2.0, 0.0),
            effector_motion=(-2.0, 0.0),
        )

        events = EventDetector().detect(
            observations,
            relationships,
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "OBJECT_MOVING_WITH_EFFECTOR",
        ) == []

    def test_does_not_detect_below_similarity_threshold(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector(
            EventDetectionConfig(min_motion_similarity=0.95)
        )

        observations, relationships = make_coordinated_observations(
            object_motion=(2.0, 0.0),
            effector_motion=(0.0, 2.0),
        )

        events = detector.detect(
            observations,
            relationships,
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "OBJECT_MOVING_WITH_EFFECTOR",
        ) == []

    def test_does_not_detect_outside_proximity_threshold(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector(
            EventDetectionConfig(max_relative_distance_pixels=10.0)
        )

        observations, relationships = make_coordinated_observations(
            mask_distance=20.0,
        )

        events = detector.detect(
            observations,
            relationships,
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "OBJECT_MOVING_WITH_EFFECTOR",
        ) == []

    @pytest.mark.parametrize(
        ("object_motion", "effector_motion"),
        [
            ((0.0, 0.0), (2.0, 0.0)),
            ((2.0, 0.0), (0.0, 0.0)),
        ],
    )
    def test_does_not_detect_when_either_entity_is_stationary(
        self,
        context,
        task_roles,
        object_motion,
        effector_motion,
    ):
        observations, relationships = make_coordinated_observations(
            object_motion=object_motion,
            effector_motion=effector_motion,
        )

        events = EventDetector().detect(
            observations,
            relationships,
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "OBJECT_MOVING_WITH_EFFECTOR",
        ) == []

    def test_does_not_detect_with_insufficient_observations(
        self,
        context,
        task_roles,
    ):
        observations, relationships = make_coordinated_observations(
            frames=range(4),
        )

        detector = EventDetector(
            EventDetectionConfig(min_observations=5)
        )

        events = detector.detect(
            observations,
            relationships,
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "OBJECT_MOVING_WITH_EFFECTOR",
        ) == []

    def test_does_not_detect_without_spatial_relationships(
        self,
        context,
        task_roles,
    ):
        observations, _ = make_coordinated_observations()

        events = EventDetector().detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "OBJECT_MOVING_WITH_EFFECTOR",
        ) == []

    def test_splits_sequences_when_frame_gap_exceeds_limit(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector(
            EventDetectionConfig(
                max_frame_gap=1,
                min_observations=3,
                min_event_duration_seconds=0.05,
                min_object_displacement_pixels=2.0,
            )
        )

        observations, relationships = make_coordinated_observations(
            frames=[0, 1, 2, 6, 7, 8],
        )

        events = get_events_by_type(
            detector.detect(
                observations,
                relationships,
                [],
                task_roles,
                context,
            ),
            "OBJECT_MOVING_WITH_EFFECTOR",
        )

        assert len(events) == 2
        assert (events[0].frame_start, events[0].frame_end) == (0, 2)
        assert (events[1].frame_start, events[1].frame_end) == (6, 8)

    def test_allows_configured_frame_gap(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector(
            EventDetectionConfig(
                max_frame_gap=1,
                min_observations=3,
                min_event_duration_seconds=0.05,
                min_object_displacement_pixels=2.0,
            )
        )

        observations, relationships = make_coordinated_observations(
            frames=[0, 1, 3],
        )

        events = get_events_by_type(
            detector.detect(
                observations,
                relationships,
                [],
                task_roles,
                context,
            ),
            "OBJECT_MOVING_WITH_EFFECTOR",
        )

        assert len(events) == 1
        assert events[0].frame_start == 0
        assert events[0].frame_end == 3

    def test_does_not_detect_without_object_role(self, context):
        observations, relationships = make_coordinated_observations()

        roles = TaskRoles(
            entity_roles={"effector": "gripper"},
            region_roles={},
        )

        events = EventDetector().detect(
            observations,
            relationships,
            [],
            roles,
            context,
        )

        assert get_events_by_type(
            events,
            "OBJECT_MOVING_WITH_EFFECTOR",
        ) == []

    def test_does_not_detect_without_effector_role(self, context):
        observations, relationships = make_coordinated_observations()

        roles = TaskRoles(
            entity_roles={"object": "cube"},
            region_roles={},
        )

        events = EventDetector().detect(
            observations,
            relationships,
            [],
            roles,
            context,
        )

        assert get_events_by_type(
            events,
            "OBJECT_MOVING_WITH_EFFECTOR",
        ) == []

    def test_evidence_does_not_claim_physical_grasp(
        self,
        context,
        task_roles,
    ):
        observations, relationships = make_coordinated_observations()

        events = EventDetector().detect(
            observations,
            relationships,
            [],
            task_roles,
            context,
        )

        coordinated_events = get_events_by_type(
            events,
            "OBJECT_MOVING_WITH_EFFECTOR",
        )

        assert len(coordinated_events) == 1

        limitations = coordinated_events[0].evidence[0].limitations

        assert any(
            "does not prove physical contact" in item
            for item in limitations
        )
        assert any(
            "calibrated probability" in item
            for item in limitations
        )


# ============================================================
# RELATIVE_SEPARATION
# ============================================================


class TestRelativeSeparationDetection:

    def test_detects_increasing_relative_separation(
        self,
        context,
        task_roles,
    ):
        observations = make_separation_observations(
            object_motion=(2.0, 0.0),
            effector_motion=(0.0, 0.0),
        )

        events = EventDetector().detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        separation_events = get_events_by_type(
            events,
            "RELATIVE_SEPARATION",
        )

        assert len(separation_events) == 1

        event = separation_events[0]

        assert event.status == EventStatus.DETECTED
        assert event.entity_ids == ("cube", "gripper")
        assert event.frame_start == 0
        assert event.frame_end == 9
        assert event.confidence is not None
        assert len(event.evidence) == 1

        evidence = event.evidence[0]

        assert evidence.evidence_type == "RELATIVE_CENTROID_SEPARATION"
        assert evidence.entity_ids == ("cube", "gripper")
        assert len(evidence.observation_ids) == 20

        assert evidence.metrics[
            "initial_centroid_distance_pixels"
        ] == pytest.approx(0.0)

        assert evidence.metrics[
            "final_centroid_distance_pixels"
        ] == pytest.approx(18.0)

        assert evidence.metrics[
            "separation_increase_pixels"
        ] == pytest.approx(18.0)

        assert evidence.metrics[
            "positive_separation_steps"
        ] == 9

        assert evidence.metrics[
            "positive_step_ratio"
        ] == pytest.approx(1.0)

    def test_does_not_detect_below_separation_threshold(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector(
            EventDetectionConfig(
                min_relative_separation_increase_pixels=20.0,
            )
        )

        observations = make_separation_observations(
            object_motion=(1.0, 0.0),
            effector_motion=(0.0, 0.0),
        )

        events = detector.detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "RELATIVE_SEPARATION",
        ) == []

    def test_does_not_detect_constant_centroid_distance(
        self,
        context,
        task_roles,
    ):
        observations = make_separation_observations(
            object_motion=(2.0, 0.0),
            effector_motion=(2.0, 0.0),
        )

        events = EventDetector().detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "RELATIVE_SEPARATION",
        ) == []

    def test_does_not_detect_decreasing_separation(
        self,
        context,
        task_roles,
    ):
        observations = make_separation_observations(
            initial_object=(200.0, 100.0),
            initial_effector=(100.0, 100.0),
            object_motion=(-2.0, 0.0),
            effector_motion=(0.0, 0.0),
        )

        events = EventDetector().detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "RELATIVE_SEPARATION",
        ) == []

    def test_does_not_detect_with_insufficient_observations(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector(
            EventDetectionConfig(min_observations=5)
        )

        observations = make_separation_observations(
            frames=range(4),
            object_motion=(10.0, 0.0),
        )

        events = EventDetector().detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "RELATIVE_SEPARATION",
        ) == []

    def test_does_not_detect_below_minimum_duration(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector(
            EventDetectionConfig(
                min_event_duration_seconds=1.0,
                min_relative_separation_increase_pixels=1.0,
            )
        )

        observations = make_separation_observations(
            object_motion=(10.0, 0.0),
        )

        events = detector.detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "RELATIVE_SEPARATION",
        ) == []

    def test_does_not_detect_single_separation_jump_followed_by_plateau(
        self,
        context,
        task_roles,
    ):
        """A single increase followed by flat measurements is insufficient."""

        observations = []

        for frame in range(10):
            object_x = 100.0 if frame == 0 else 118.0

            observations.extend(
                [
                    make_entity_observation(
                        entity_id="cube",
                        frame_index=frame,
                        centroid=(object_x, 100.0),
                    ),
                    make_entity_observation(
                        entity_id="gripper",
                        frame_index=frame,
                        centroid=(100.0, 100.0),
                    ),
                ]
            )

        events = EventDetector().detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "RELATIVE_SEPARATION",
        ) == []

    def test_does_not_detect_when_half_of_steps_are_increasing(
        self,
        context,
        task_roles,
    ):
        """The trend threshold is strictly greater than 50 percent."""

        # Four increasing steps and four flat steps.
        # Exactly 50 percent of the steps are increasing.
        distances = [
            0.0,
            2.0,
            2.0,
            4.0,
            4.0,
            6.0,
            6.0,
            8.0,
            8.0,
        ]

        observations = []

        for frame, distance in enumerate(distances):
            observations.extend(
                [
                    make_entity_observation(
                        entity_id="cube",
                        frame_index=frame,
                        centroid=(100.0 + distance, 100.0),
                    ),
                    make_entity_observation(
                        entity_id="gripper",
                        frame_index=frame,
                        centroid=(100.0, 100.0),
                    ),
                ]
            )

        events = EventDetector().detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "RELATIVE_SEPARATION",
        ) == []

    def test_splits_sequences_when_frame_gap_exceeds_configuration(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector(
            EventDetectionConfig(
                max_frame_gap=1,
                min_observations=3,
                min_event_duration_seconds=0.05,
                min_relative_separation_increase_pixels=2.0,
            )
        )

        observations = make_separation_observations(
            frames=[0, 1, 2, 6, 7, 8],
            object_motion=(3.0, 0.0),
        )

        events = get_events_by_type(
            detector.detect(
                observations,
                [],
                [],
                task_roles,
                context,
            ),
            "RELATIVE_SEPARATION",
        )

        assert len(events) == 2
        assert (events[0].frame_start, events[0].frame_end) == (0, 2)
        assert (events[1].frame_start, events[1].frame_end) == (6, 8)

    def test_allows_gap_within_configured_limit(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector(
            EventDetectionConfig(
                max_frame_gap=1,
                min_observations=3,
                min_event_duration_seconds=0.05,
                min_relative_separation_increase_pixels=2.0,
            )
        )

        observations = make_separation_observations(
            frames=[0, 1, 3],
            object_motion=(3.0, 0.0),
        )

        events = get_events_by_type(
            detector.detect(
                observations,
                [],
                [],
                task_roles,
                context,
            ),
            "RELATIVE_SEPARATION",
        )

        assert len(events) == 1
        assert events[0].frame_start == 0
        assert events[0].frame_end == 3

    def test_does_not_detect_without_object_role(self, context):
        observations = make_separation_observations()

        roles = TaskRoles(
            entity_roles={"effector": "gripper"},
            region_roles={},
        )

        events = EventDetector().detect(
            observations,
            [],
            [],
            roles,
            context,
        )

        assert get_events_by_type(
            events,
            "RELATIVE_SEPARATION",
        ) == []

    def test_does_not_detect_without_effector_role(self, context):
        observations = make_separation_observations()

        roles = TaskRoles(
            entity_roles={"object": "cube"},
            region_roles={},
        )

        events = EventDetector().detect(
            observations,
            [],
            [],
            roles,
            context,
        )

        assert get_events_by_type(
            events,
            "RELATIVE_SEPARATION",
        ) == []

    def test_does_not_detect_when_entity_observations_are_not_paired(
        self,
        context,
        task_roles,
    ):
        """Entities observed in disjoint frame intervals cannot be paired."""

        observations = []

        observations.extend(
            [
                make_entity_observation(
                    entity_id="cube",
                    frame_index=frame,
                    centroid=(100.0 + frame * 2.0, 100.0),
                )
                for frame in range(5)
            ]
        )

        observations.extend(
            [
                make_entity_observation(
                    entity_id="gripper",
                    frame_index=frame,
                    centroid=(100.0, 100.0),
                )
                for frame in range(5, 10)
            ]
        )

        events = EventDetector().detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "RELATIVE_SEPARATION",
        ) == []

    def test_evidence_explains_that_separation_does_not_prove_release(
        self,
        context,
        task_roles,
    ):
        observations = make_separation_observations(
            object_motion=(2.0, 0.0),
            effector_motion=(0.0, 0.0),
        )

        events = EventDetector().detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        separation_events = get_events_by_type(
            events,
            "RELATIVE_SEPARATION",
        )

        assert len(separation_events) == 1

        limitations = separation_events[0].evidence[0].limitations

        assert any(
            "does not prove" in item and "released" in item
            for item in limitations
        )
        assert any(
            "physical-world distance" in item
            for item in limitations
        )
        assert any(
            "calibrated probability" in item
            for item in limitations
        )

    def test_separation_detection_does_not_require_spatial_relationships(
        self,
        context,
        task_roles,
    ):
        observations = make_separation_observations(
            object_motion=(2.0, 0.0),
            effector_motion=(0.0, 0.0),
        )

        events = EventDetector().detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert len(
            get_events_by_type(events, "RELATIVE_SEPARATION")
        ) == 1

    def test_returns_events_in_chronological_order(
        self,
        context,
        task_roles,
    ):
        observations = make_separation_observations(
            object_motion=(2.0, 0.0),
            effector_motion=(0.0, 0.0),
        )

        events = EventDetector().detect(
            observations,
            [],
            [],
            task_roles,
            context,
        )

        assert list(events) == sorted(
            events,
            key=lambda event: (
                event.frame_start,
                event.frame_end,
                event.event_type,
                event.event_id,
            ),
        )


# ============================================================
# STABLE_PLACEMENT
# ============================================================


class TestStablePlacementDetection:

    @pytest.fixture
    def stable_detector(self):
        return EventDetector(
            EventDetectionConfig(
                min_observations=3,
                max_frame_gap=1,
                min_region_overlap_ratio=0.80,
                min_stable_duration_seconds=0.05,
                max_stationary_speed_pixels_per_second=3.0,
            )
        )

    def make_stable_observations(
        self,
        frames=range(3),
        overlap_ratio=0.90,
        centroids=None,
    ):
        """Build object and target-region observations."""

        entity_observations = []
        region_observations = []

        if centroids is None:
            centroids = {
                frame: (100.0, 100.0)
                for frame in frames
            }

        for frame in frames:
            entity_observations.append(
                make_entity_observation(
                    entity_id="cube",
                    frame_index=frame,
                    centroid=centroids[frame],
                )
            )

            region_observations.append(
                make_region_observation(
                    entity_id="cube",
                    region_id="target_zone",
                    frame_index=frame,
                    overlap_ratio=overlap_ratio,
                )
            )

        return entity_observations, region_observations

    def test_detects_stable_object_inside_target_region(
        self,
        context,
        task_roles,
        stable_detector,
    ):
        entity_observations, region_observations = (
            self.make_stable_observations()
        )

        events = stable_detector.detect(
            entity_observations,
            [],
            region_observations,
            task_roles,
            context,
        )

        stable_events = get_events_by_type(
            events,
            "STABLE_PLACEMENT",
        )

        assert len(stable_events) == 1

        event = stable_events[0]

        assert event.status == EventStatus.DETECTED
        assert event.entity_ids == ("cube",)
        assert event.frame_start == 0
        assert event.frame_end == 2
        assert event.timestamp_start == pytest.approx(0.0)
        assert event.timestamp_end == pytest.approx(2 / 30.0)
        assert event.confidence is not None
        assert len(event.evidence) == 1
        assert event.evidence[0].evidence_type == "STABLE_IN_REGION"
        assert event.evidence[0].entity_ids == ("cube",)

    def test_does_not_detect_when_stable_duration_is_insufficient(
        self,
        context,
        task_roles,
    ):
        detector = EventDetector(
            EventDetectionConfig(
                min_observations=3,
                min_region_overlap_ratio=0.80,
                min_stable_duration_seconds=0.10,
                max_stationary_speed_pixels_per_second=3.0,
            )
        )

        entity_observations = [
            make_entity_observation(
                frame_index=frame,
                centroid=(100.0, 100.0),
            )
            for frame in range(3)
        ]

        region_observations = [
            make_region_observation(
                frame_index=frame,
                overlap_ratio=0.90,
            )
            for frame in range(3)
        ]

        events = detector.detect(
            entity_observations,
            [],
            region_observations,
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "STABLE_PLACEMENT",
        ) == []

    def test_does_not_detect_when_object_moves_too_fast(
        self,
        context,
        task_roles,
        stable_detector,
    ):
        entity_observations = [
            make_entity_observation(
                frame_index=frame,
                centroid=(100.0 + frame * 1.0, 100.0),
            )
            for frame in range(3)
        ]

        region_observations = [
            make_region_observation(
                frame_index=frame,
                overlap_ratio=0.90,
            )
            for frame in range(3)
        ]

        events = stable_detector.detect(
            entity_observations,
            [],
            region_observations,
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "STABLE_PLACEMENT",
        ) == []

    def test_does_not_detect_below_region_overlap_threshold(
        self,
        context,
        task_roles,
        stable_detector,
    ):
        entity_observations, region_observations = (
            self.make_stable_observations(
                overlap_ratio=0.70,
            )
        )

        events = stable_detector.detect(
            entity_observations,
            [],
            region_observations,
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "STABLE_PLACEMENT",
        ) == []

    def test_does_not_bridge_frame_gap_exceeding_configuration(
        self,
        context,
        task_roles,
        stable_detector,
    ):
        frames = [0, 1, 4]

        entity_observations, region_observations = (
            self.make_stable_observations(
                frames=frames,
            )
        )

        events = stable_detector.detect(
            entity_observations,
            [],
            region_observations,
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "STABLE_PLACEMENT",
        ) == []

    def test_does_not_bridge_region_overlap_interruption(
        self,
        context,
        task_roles,
        stable_detector,
    ):
        entity_observations = [
            make_entity_observation(
                frame_index=frame,
                centroid=(100.0, 100.0),
            )
            for frame in range(5)
        ]

        overlap_by_frame = {
            0: 0.90,
            1: 0.90,
            2: 0.50,
            3: 0.90,
            4: 0.90,
        }

        region_observations = [
            make_region_observation(
                frame_index=frame,
                overlap_ratio=overlap_by_frame[frame],
            )
            for frame in range(5)
        ]

        events = stable_detector.detect(
            entity_observations,
            [],
            region_observations,
            task_roles,
            context,
        )

        assert get_events_by_type(
            events,
            "STABLE_PLACEMENT",
        ) == []

    def test_detects_stability_after_initial_movement(
        self,
        context,
        task_roles,
        stable_detector,
    ):
        centroids = {
            0: (100.0, 100.0),
            1: (105.0, 100.0),
            2: (105.01, 100.0),
            3: (105.02, 100.0),
            4: (105.03, 100.0),
        }

        entity_observations, region_observations = (
            self.make_stable_observations(
                frames=range(5),
                centroids=centroids,
            )
        )

        events = stable_detector.detect(
            entity_observations,
            [],
            region_observations,
            task_roles,
            context,
        )

        stable_events = get_events_by_type(
            events,
            "STABLE_PLACEMENT",
        )

        assert len(stable_events) == 1

        event = stable_events[0]

        # The initial high-speed movement must not be included
        # in the stable interval.
        assert event.frame_start == 1
        assert event.frame_end == 4

    def test_does_not_detect_without_object_role(
        self,
        context,
    ):
        entity_observations, region_observations = (
            self.make_stable_observations()
        )

        roles = TaskRoles(
            entity_roles={"effector": "gripper"},
            region_roles={"target_region": "target_zone"},
        )

        events = EventDetector(
            EventDetectionConfig(
                min_observations=3,
                min_region_overlap_ratio=0.80,
                min_stable_duration_seconds=0.05,
            )
        ).detect(
            entity_observations,
            [],
            region_observations,
            roles,
            context,
        )

        assert get_events_by_type(
            events,
            "STABLE_PLACEMENT",
        ) == []

    def test_does_not_detect_without_target_region_role(
        self,
        context,
    ):
        entity_observations, region_observations = (
            self.make_stable_observations()
        )

        roles = TaskRoles(
            entity_roles={"object": "cube"},
            region_roles={},
        )

        events = EventDetector(
            EventDetectionConfig(
                min_observations=3,
                min_region_overlap_ratio=0.80,
                min_stable_duration_seconds=0.05,
            )
        ).detect(
            entity_observations,
            [],
            region_observations,
            roles,
            context,
        )

        assert get_events_by_type(
            events,
            "STABLE_PLACEMENT",
        ) == []

    def test_evidence_does_not_claim_physical_placement(
        self,
        context,
        task_roles,
        stable_detector,
    ):
        entity_observations, region_observations = (
            self.make_stable_observations()
        )

        events = stable_detector.detect(
            entity_observations,
            [],
            region_observations,
            task_roles,
            context,
        )

        stable_events = get_events_by_type(
            events,
            "STABLE_PLACEMENT",
        )

        assert len(stable_events) == 1

        limitations = stable_events[0].evidence[0].limitations

        assert len(limitations) > 0

