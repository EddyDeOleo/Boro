
"""Tests for generic temporal event models."""

from dataclasses import FrozenInstanceError

import pytest

from src.models.event import EventStatus, TemporalEvent
from src.models.evidence import Evidence


@pytest.fixture
def valid_evidence():
    """Provide valid evidence for detected events."""

    return Evidence(
        evidence_id="evidence_001",
        evidence_type="MOTION_ANALYSIS",
        description="Sustained coordinated movement was observed.",
        frame_start=10,
        frame_end=20,
        observation_ids=(
            "sam:red_cube:10",
            "sam:gripper:10",
        ),
        entity_ids=(
            "red_cube",
            "gripper",
        ),
        metrics={
            "mean_relative_distance_pixels": 54.2,
            "sustained_frames": 11,
        },
        limitations=(
            "Pixel displacement does not establish physical contact.",
        ),
    )


@pytest.fixture
def detected_event(valid_evidence):
    """Provide a valid detected temporal event."""

    return TemporalEvent(
        event_id="event_001",
        event_type="OBJECT_MOVING_WITH_EFFECTOR",
        status=EventStatus.DETECTED,
        frame_start=10,
        frame_end=20,
        timestamp_start=0.33,
        timestamp_end=0.67,
        entity_ids=("red_cube", "gripper"),
        confidence=0.87,
        evidence=(valid_evidence,),
        attributes={
            "mean_relative_distance_pixels": 54.2,
            "sustained_frames": 11,
        },
    )


class TestEventStatus:
    """Tests for event status values."""

    def test_event_status_values(self):
        assert EventStatus.DETECTED.value == "DETECTED"
        assert EventStatus.NOT_DETECTED.value == "NOT_DETECTED"
        assert EventStatus.NOT_OBSERVABLE.value == "NOT_OBSERVABLE"

    def test_event_status_is_string_enum(self):
        assert isinstance(EventStatus.DETECTED, str)
        assert EventStatus.DETECTED == "DETECTED"


class TestTemporalEvent:
    """Tests for TemporalEvent."""

    def test_create_detected_event(self, detected_event, valid_evidence):
        assert detected_event.event_id == "event_001"
        assert detected_event.event_type == "OBJECT_MOVING_WITH_EFFECTOR"
        assert detected_event.status == EventStatus.DETECTED
        assert detected_event.frame_start == 10
        assert detected_event.frame_end == 20
        assert detected_event.timestamp_start == 0.33
        assert detected_event.timestamp_end == 0.67
        assert detected_event.entity_ids == ("red_cube", "gripper")
        assert detected_event.confidence == 0.87
        assert detected_event.evidence == (valid_evidence,)

    def test_create_not_detected_event(self):
        event = TemporalEvent(
            event_id="event_002",
            event_type="OBJECT_ENTERED_REGION",
            status=EventStatus.NOT_DETECTED,
            frame_start=0,
            frame_end=50,
            timestamp_start=0.0,
            timestamp_end=1.67,
            entity_ids=("red_cube",),
        )

        assert event.status == EventStatus.NOT_DETECTED
        assert event.confidence is None
        assert event.evidence == ()

    def test_create_not_observable_event(self):
        event = TemporalEvent(
            event_id="event_003",
            event_type="OBJECT_RESTING_IN_REGION",
            status=EventStatus.NOT_OBSERVABLE,
            frame_start=40,
            frame_end=50,
            timestamp_start=1.33,
            timestamp_end=1.67,
            entity_ids=("red_cube",),
            attributes={
                "reason": "Video segment ended before the event could be evaluated.",
            },
        )

        assert event.status == EventStatus.NOT_OBSERVABLE
        assert event.confidence is None
        assert event.attributes["reason"].startswith("Video segment ended")

    def test_create_event_without_entities(self):
        event = TemporalEvent(
            event_id="event_004",
            event_type="TASK_INTERRUPTED",
            status=EventStatus.NOT_OBSERVABLE,
            frame_start=0,
            frame_end=10,
            timestamp_start=0.0,
            timestamp_end=0.33,
            entity_ids=(),
        )

        assert event.entity_ids == ()

    def test_create_event_with_empty_attributes(self):
        event = TemporalEvent(
            event_id="event_005",
            event_type="UNKNOWN_EVENT",
            status=EventStatus.NOT_DETECTED,
            frame_start=0,
            frame_end=5,
            timestamp_start=0.0,
            timestamp_end=0.17,
            entity_ids=(),
        )

        assert event.attributes == {}

    def test_detected_event_requires_confidence(self, valid_evidence):
        with pytest.raises(
            ValueError,
            match="must include a confidence score",
        ):
            TemporalEvent(
                event_id="event_006",
                event_type="POSSIBLE_GRASP",
                status=EventStatus.DETECTED,
                frame_start=0,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=0.33,
                entity_ids=("red_cube", "gripper"),
                evidence=(valid_evidence,),
            )

    def test_detected_event_requires_evidence(self):
        with pytest.raises(
            ValueError,
            match="must include supporting evidence",
        ):
            TemporalEvent(
                event_id="event_007",
                event_type="POSSIBLE_GRASP",
                status=EventStatus.DETECTED,
                frame_start=0,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=0.33,
                entity_ids=("red_cube", "gripper"),
                confidence=0.85,
            )

    @pytest.mark.parametrize(
        "status",
        [
            EventStatus.NOT_DETECTED,
            EventStatus.NOT_OBSERVABLE,
        ],
    )
    def test_non_detected_events_reject_confidence(self, status):
        with pytest.raises(
            ValueError,
            match="Confidence must be None",
        ):
            TemporalEvent(
                event_id="event_008",
                event_type="OBJECT_ENTERED_REGION",
                status=status,
                frame_start=0,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=0.33,
                entity_ids=("red_cube",),
                confidence=0.75,
            )

    @pytest.mark.parametrize(
        "confidence",
        [
            -0.1,
            1.1,
            float("inf"),
            float("-inf"),
            float("nan"),
            True,
            "0.8",
        ],
    )
    def test_invalid_confidence_raises_value_error(
        self,
        confidence,
        valid_evidence,
    ):
        with pytest.raises(ValueError):
            TemporalEvent(
                event_id="event_009",
                event_type="OBJECT_MOVING_WITH_EFFECTOR",
                status=EventStatus.DETECTED,
                frame_start=0,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=0.33,
                entity_ids=("red_cube", "gripper"),
                confidence=confidence,
                evidence=(valid_evidence,),
            )

    @pytest.mark.parametrize(
        "confidence",
        [
            0.0,
            0.5,
            1.0,
        ],
    )
    def test_valid_confidence_values(self, confidence, valid_evidence):
        event = TemporalEvent(
            event_id="event_010",
            event_type="OBJECT_MOVING_WITH_EFFECTOR",
            status=EventStatus.DETECTED,
            frame_start=0,
            frame_end=10,
            timestamp_start=0.0,
            timestamp_end=0.33,
            entity_ids=("red_cube", "gripper"),
            confidence=confidence,
            evidence=(valid_evidence,),
        )

        assert event.confidence == confidence

    def test_frame_start_cannot_exceed_frame_end(self):
        with pytest.raises(
            ValueError,
            match="frame_start cannot be greater",
        ):
            TemporalEvent(
                event_id="event_011",
                event_type="OBJECT_ENTERED_REGION",
                status=EventStatus.NOT_DETECTED,
                frame_start=20,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=1.0,
                entity_ids=("red_cube",),
            )

    def test_timestamp_start_cannot_exceed_timestamp_end(self):
        with pytest.raises(
            ValueError,
            match="timestamp_start cannot be greater",
        ):
            TemporalEvent(
                event_id="event_012",
                event_type="OBJECT_ENTERED_REGION",
                status=EventStatus.NOT_DETECTED,
                frame_start=0,
                frame_end=10,
                timestamp_start=2.0,
                timestamp_end=1.0,
                entity_ids=("red_cube",),
            )

    @pytest.mark.parametrize(
        "frame_start",
        [
            -1,
            1.5,
            True,
            "0",
        ],
    )
    def test_invalid_frame_start_raises_value_error(self, frame_start):
        with pytest.raises(ValueError):
            TemporalEvent(
                event_id="event_013",
                event_type="OBJECT_ENTERED_REGION",
                status=EventStatus.NOT_DETECTED,
                frame_start=frame_start,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=1.0,
                entity_ids=("red_cube",),
            )

    @pytest.mark.parametrize(
        "timestamp",
        [
            -1.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            True,
            "1.0",
        ],
    )
    def test_invalid_timestamp_raises_value_error(self, timestamp):
        with pytest.raises(ValueError):
            TemporalEvent(
                event_id="event_014",
                event_type="OBJECT_ENTERED_REGION",
                status=EventStatus.NOT_DETECTED,
                frame_start=0,
                frame_end=10,
                timestamp_start=timestamp,
                timestamp_end=1.0,
                entity_ids=("red_cube",),
            )

    def test_empty_event_id_raises_value_error(self):
        with pytest.raises(ValueError, match="event_id"):
            TemporalEvent(
                event_id="",
                event_type="OBJECT_ENTERED_REGION",
                status=EventStatus.NOT_DETECTED,
                frame_start=0,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=1.0,
                entity_ids=("red_cube",),
            )

    def test_empty_event_type_raises_value_error(self):
        with pytest.raises(ValueError, match="event_type"):
            TemporalEvent(
                event_id="event_015",
                event_type=" ",
                status=EventStatus.NOT_DETECTED,
                frame_start=0,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=1.0,
                entity_ids=("red_cube",),
            )

    def test_invalid_event_status_raises_value_error(self):
        with pytest.raises(ValueError, match="EventStatus"):
            TemporalEvent(
                event_id="event_016",
                event_type="OBJECT_ENTERED_REGION",
                status="DETECTED",
                frame_start=0,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=1.0,
                entity_ids=("red_cube",),
            )

    def test_duplicate_entity_ids_raise_value_error(self):
        with pytest.raises(
            ValueError,
            match="cannot contain duplicates",
        ):
            TemporalEvent(
                event_id="event_017",
                event_type="OBJECT_MOVING_WITH_EFFECTOR",
                status=EventStatus.NOT_DETECTED,
                frame_start=0,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=1.0,
                entity_ids=("red_cube", "red_cube"),
            )

    def test_empty_entity_id_raises_value_error(self):
        with pytest.raises(ValueError, match="entity_id"):
            TemporalEvent(
                event_id="event_018",
                event_type="OBJECT_ENTERED_REGION",
                status=EventStatus.NOT_DETECTED,
                frame_start=0,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=1.0,
                entity_ids=("",),
            )

    def test_entity_ids_must_be_tuple(self):
        with pytest.raises(ValueError, match="must be a tuple"):
            TemporalEvent(
                event_id="event_019",
                event_type="OBJECT_ENTERED_REGION",
                status=EventStatus.NOT_DETECTED,
                frame_start=0,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=1.0,
                entity_ids=["red_cube"],
            )

    def test_evidence_must_be_tuple(self):
        with pytest.raises(ValueError, match="must be a tuple"):
            TemporalEvent(
                event_id="event_020",
                event_type="OBJECT_ENTERED_REGION",
                status=EventStatus.NOT_DETECTED,
                frame_start=0,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=1.0,
                entity_ids=("red_cube",),
                evidence=[],
            )

    def test_invalid_evidence_item_raises_value_error(self):
        with pytest.raises(ValueError, match="Evidence instance"):
            TemporalEvent(
                event_id="event_021",
                event_type="OBJECT_ENTERED_REGION",
                status=EventStatus.NOT_DETECTED,
                frame_start=0,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=1.0,
                entity_ids=("red_cube",),
                evidence=("invalid",),
            )

    def test_attributes_are_immutable(self, detected_event):
        with pytest.raises(TypeError):
            detected_event.attributes["new_metric"] = 10.0

    def test_external_attributes_changes_do_not_affect_event(self):
        attributes = {
            "sustained_frames": 10,
        }

        event = TemporalEvent(
            event_id="event_022",
            event_type="OBJECT_ENTERED_REGION",
            status=EventStatus.NOT_DETECTED,
            frame_start=0,
            frame_end=10,
            timestamp_start=0.0,
            timestamp_end=1.0,
            entity_ids=("red_cube",),
            attributes=attributes,
        )

        attributes["sustained_frames"] = 20

        assert event.attributes["sustained_frames"] == 10

    def test_event_is_immutable(self, detected_event):
        with pytest.raises(FrozenInstanceError):
            detected_event.event_id = "event_modified"

    @pytest.mark.parametrize(
        "attributes",
        [
            None,
            [],
            "invalid",
            123,
        ],
    )
    def test_invalid_attributes_type_raises_value_error(self, attributes):
        with pytest.raises(ValueError, match="attributes must be a mapping"):
            TemporalEvent(
                event_id="event_023",
                event_type="OBJECT_ENTERED_REGION",
                status=EventStatus.NOT_DETECTED,
                frame_start=0,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=1.0,
                entity_ids=("red_cube",),
                attributes=attributes,
            )

    def test_empty_attribute_key_raises_value_error(self):
        with pytest.raises(ValueError, match="attribute key"):
            TemporalEvent(
                event_id="event_024",
                event_type="OBJECT_ENTERED_REGION",
                status=EventStatus.NOT_DETECTED,
                frame_start=0,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=1.0,
                entity_ids=("red_cube",),
                attributes={"": 10},
            )

    def test_non_finite_attribute_raises_value_error(self):
        with pytest.raises(ValueError, match="finite float"):
            TemporalEvent(
                event_id="event_025",
                event_type="OBJECT_ENTERED_REGION",
                status=EventStatus.NOT_DETECTED,
                frame_start=0,
                frame_end=10,
                timestamp_start=0.0,
                timestamp_end=1.0,
                entity_ids=("red_cube",),
                attributes={"invalid_metric": float("nan")},
            )

    def test_detected_event_can_have_multiple_evidence_items(
        self,
        valid_evidence,
    ):
        second_evidence = Evidence(
            evidence_id="evidence_002",
            evidence_type="SPATIAL_ANALYSIS",
            description="Relative distance remained within the configured range.",
            frame_start=12,
            frame_end=18,
            entity_ids=("red_cube", "gripper"),
        )

        event = TemporalEvent(
            event_id="event_026",
            event_type="POSSIBLE_GRASP",
            status=EventStatus.DETECTED,
            frame_start=10,
            frame_end=20,
            timestamp_start=0.33,
            timestamp_end=0.67,
            entity_ids=("red_cube", "gripper"),
            confidence=0.78,
            evidence=(valid_evidence, second_evidence),
        )

        assert len(event.evidence) == 2
        assert event.evidence[0].evidence_id == "evidence_001"
        assert event.evidence[1].evidence_id == "evidence_002"
