
import pytest

from src.models.event import EventStatus
from src.presentation.models.timeline_event_view import TimelineEventView


@pytest.fixture
def valid_event():
    return TimelineEventView(
        event_id="event-001",
        event_type="STABLE_PLACEMENT",
        status=EventStatus.DETECTED,
        frame_start=10,
        frame_end=40,
        timestamp_start=0.333333,
        timestamp_end=1.333333,
        entity_ids=("entity-001",),
        confidence=0.95,
        evidence_ids=("evidence-001",),
        attributes={
            "mean_speed": 0.12,
            "stable_duration": 1.0,
            "region_id": "target_zone_01",
        },
    )


def test_creates_valid_timeline_event_view(valid_event):
    assert valid_event.event_id == "event-001"
    assert valid_event.event_type == "STABLE_PLACEMENT"
    assert valid_event.status == EventStatus.DETECTED
    assert valid_event.frame_start == 10
    assert valid_event.frame_end == 40
    assert valid_event.timestamp_start == 0.333333
    assert valid_event.timestamp_end == 1.333333
    assert valid_event.entity_ids == ("entity-001",)
    assert valid_event.confidence == 0.95
    assert valid_event.evidence_ids == ("evidence-001",)
    assert valid_event.attributes["region_id"] == "target_zone_01"


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_id", ""),
        ("event_type", ""),
    ],
)
def test_rejects_empty_required_strings(valid_event, field, value):
    values = {
        "event_id": valid_event.event_id,
        "event_type": valid_event.event_type,
        "status": valid_event.status,
        "frame_start": valid_event.frame_start,
        "frame_end": valid_event.frame_end,
        "timestamp_start": valid_event.timestamp_start,
        "timestamp_end": valid_event.timestamp_end,
        "entity_ids": valid_event.entity_ids,
        "confidence": valid_event.confidence,
        "evidence_ids": valid_event.evidence_ids,
        "attributes": dict(valid_event.attributes),
    }
    values[field] = value

    with pytest.raises(ValueError):
        TimelineEventView(**values)


def test_rejects_invalid_status(valid_event):
    values = {
        "event_id": valid_event.event_id,
        "event_type": valid_event.event_type,
        "status": "UNKNOWN",
        "frame_start": valid_event.frame_start,
        "frame_end": valid_event.frame_end,
        "timestamp_start": valid_event.timestamp_start,
        "timestamp_end": valid_event.timestamp_end,
        "entity_ids": valid_event.entity_ids,
        "confidence": valid_event.confidence,
        "evidence_ids": valid_event.evidence_ids,
        "attributes": dict(valid_event.attributes),
    }

    with pytest.raises((ValueError, TypeError)):
        TimelineEventView(**values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame_start", -1),
        ("frame_end", -1),
        ("timestamp_start", -1.0),
        ("timestamp_end", -1.0),
    ],
)
def test_rejects_negative_positions(valid_event, field, value):
    values = {
        "event_id": valid_event.event_id,
        "event_type": valid_event.event_type,
        "status": valid_event.status,
        "frame_start": valid_event.frame_start,
        "frame_end": valid_event.frame_end,
        "timestamp_start": valid_event.timestamp_start,
        "timestamp_end": valid_event.timestamp_end,
        "entity_ids": valid_event.entity_ids,
        "confidence": valid_event.confidence,
        "evidence_ids": valid_event.evidence_ids,
        "attributes": dict(valid_event.attributes),
    }
    values[field] = value

    with pytest.raises((ValueError, TypeError)):
        TimelineEventView(**values)


def test_rejects_frame_end_before_frame_start(valid_event):
    values = {
        "event_id": valid_event.event_id,
        "event_type": valid_event.event_type,
        "status": valid_event.status,
        "frame_start": 40,
        "frame_end": 10,
        "timestamp_start": valid_event.timestamp_start,
        "timestamp_end": valid_event.timestamp_end,
        "entity_ids": valid_event.entity_ids,
        "confidence": valid_event.confidence,
        "evidence_ids": valid_event.evidence_ids,
        "attributes": dict(valid_event.attributes),
    }

    with pytest.raises(ValueError):
        TimelineEventView(**values)


def test_rejects_timestamp_end_before_timestamp_start(valid_event):
    values = {
        "event_id": valid_event.event_id,
        "event_type": valid_event.event_type,
        "status": valid_event.status,
        "frame_start": valid_event.frame_start,
        "frame_end": valid_event.frame_end,
        "timestamp_start": 2.0,
        "timestamp_end": 1.0,
        "entity_ids": valid_event.entity_ids,
        "confidence": valid_event.confidence,
        "evidence_ids": valid_event.evidence_ids,
        "attributes": dict(valid_event.attributes),
    }

    with pytest.raises(ValueError):
        TimelineEventView(**values)


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01, float("nan"), float("inf"), float("-inf")],
)
def test_rejects_invalid_confidence(valid_event, confidence):
    values = {
        "event_id": valid_event.event_id,
        "event_type": valid_event.event_type,
        "status": valid_event.status,
        "frame_start": valid_event.frame_start,
        "frame_end": valid_event.frame_end,
        "timestamp_start": valid_event.timestamp_start,
        "timestamp_end": valid_event.timestamp_end,
        "entity_ids": valid_event.entity_ids,
        "confidence": confidence,
        "evidence_ids": valid_event.evidence_ids,
        "attributes": dict(valid_event.attributes),
    }

    with pytest.raises((ValueError, TypeError)):
        TimelineEventView(**values)


def test_rejects_detected_event_without_confidence(valid_event):
    values = {
        "event_id": valid_event.event_id,
        "event_type": valid_event.event_type,
        "status": EventStatus.DETECTED,
        "frame_start": valid_event.frame_start,
        "frame_end": valid_event.frame_end,
        "timestamp_start": valid_event.timestamp_start,
        "timestamp_end": valid_event.timestamp_end,
        "entity_ids": valid_event.entity_ids,
        "confidence": None,
        "evidence_ids": valid_event.evidence_ids,
        "attributes": dict(valid_event.attributes),
    }

    with pytest.raises(ValueError):
        TimelineEventView(**values)


def test_rejects_detected_event_without_evidence(valid_event):
    values = {
        "event_id": valid_event.event_id,
        "event_type": valid_event.event_type,
        "status": EventStatus.DETECTED,
        "frame_start": valid_event.frame_start,
        "frame_end": valid_event.frame_end,
        "timestamp_start": valid_event.timestamp_start,
        "timestamp_end": valid_event.timestamp_end,
        "entity_ids": valid_event.entity_ids,
        "confidence": valid_event.confidence,
        "evidence_ids": (),
        "attributes": dict(valid_event.attributes),
    }

    with pytest.raises(ValueError):
        TimelineEventView(**values)


@pytest.mark.parametrize(
    "status",
    [EventStatus.NOT_DETECTED, EventStatus.NOT_OBSERVABLE],
)
def test_allows_non_detected_events_without_confidence_or_evidence(status):
    event = TimelineEventView(
        event_id="event-002",
        event_type="STABLE_PLACEMENT",
        status=status,
        frame_start=0,
        frame_end=30,
        timestamp_start=0.0,
        timestamp_end=1.0,
        entity_ids=("entity-001",),
        confidence=None,
        evidence_ids=(),
        attributes={},
    )

    assert event.status == status
    assert event.confidence is None
    assert event.evidence_ids == ()


def test_freezes_attributes_mapping(valid_event):
    with pytest.raises(TypeError):
        valid_event.attributes["new_attribute"] = 1.0


def test_rejects_non_finite_attribute_values():
    with pytest.raises((ValueError, TypeError)):
        TimelineEventView(
            event_id="event-003",
            event_type="OBJECT_MOVING",
            status=EventStatus.DETECTED,
            frame_start=0,
            frame_end=10,
            timestamp_start=0.0,
            timestamp_end=1.0,
            entity_ids=("entity-001",),
            confidence=0.9,
            evidence_ids=("evidence-001",),
            attributes={"speed": float("nan")},
        )


def test_rejects_duplicate_entity_ids():
    with pytest.raises(ValueError):
        TimelineEventView(
            event_id="event-004",
            event_type="OBJECT_MOVING",
            status=EventStatus.DETECTED,
            frame_start=0,
            frame_end=10,
            timestamp_start=0.0,
            timestamp_end=1.0,
            entity_ids=("entity-001", "entity-001"),
            confidence=0.9,
            evidence_ids=("evidence-001",),
            attributes={},
        )


def test_rejects_duplicate_evidence_ids():
    with pytest.raises(ValueError):
        TimelineEventView(
            event_id="event-005",
            event_type="OBJECT_MOVING",
            status=EventStatus.DETECTED,
            frame_start=0,
            frame_end=10,
            timestamp_start=0.0,
            timestamp_end=1.0,
            entity_ids=("entity-001",),
            confidence=0.9,
            evidence_ids=("evidence-001", "evidence-001"),
            attributes={},
        )


def test_timeline_event_view_is_immutable(valid_event):
    with pytest.raises((AttributeError, TypeError)):
        valid_event.confidence = 0.5
