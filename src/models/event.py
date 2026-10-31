
"""Generic temporal event models for robotic execution analysis."""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from src.models.evidence import Evidence, MetadataValue


class EventStatus(str, Enum):
    """Observability and detection status of a temporal event."""

    DETECTED = "DETECTED"
    NOT_DETECTED = "NOT_DETECTED"
    NOT_OBSERVABLE = "NOT_OBSERVABLE"


def _validate_non_empty(value: str, field_name: str) -> None:
    """Validate a non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _validate_frame_index(value: int, field_name: str) -> None:
    """Validate a non-negative integer frame index."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")

    if value < 0:
        raise ValueError(f"{field_name} cannot be negative.")


def _validate_timestamp(value: float, field_name: str) -> None:
    """Validate a finite non-negative timestamp."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")

    if not isfinite(value) or value < 0:
        raise ValueError(
            f"{field_name} must be finite and non-negative."
        )


def _validate_confidence(value: float) -> None:
    """Validate an event confidence score."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be numeric.")

    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be between 0 and 1.")


def _freeze_attributes(
    attributes: Mapping[str, MetadataValue],
) -> Mapping[str, MetadataValue]:
    """Validate and freeze event metadata."""

    if not isinstance(attributes, Mapping):
        raise ValueError("attributes must be a mapping.")

    copied_attributes = dict(attributes)

    for key, value in copied_attributes.items():
        _validate_non_empty(key, "attribute key")

        if value is not None and not isinstance(
            value,
            (float, int, str, bool),
        ):
            raise ValueError(
                f"Unsupported attribute value for {key!r}."
            )

        if isinstance(value, float) and not isfinite(value):
            raise ValueError(
                f"Attribute {key!r} must contain a finite float."
            )

    return MappingProxyType(copied_attributes)


@dataclass(frozen=True)
class TemporalEvent:
    """A temporal event or event assessment derived from observations.

    frame_start and frame_end represent the evaluated interval.
    They do not necessarily indicate that the event occurred.

    confidence represents the detector's confidence in a detected
    event, not a calibrated physical probability.
    """

    event_id: str
    event_type: str
    status: EventStatus

    frame_start: int
    frame_end: int

    timestamp_start: float
    timestamp_end: float

    entity_ids: tuple[str, ...]

    confidence: float | None = None

    evidence: tuple[Evidence, ...] = ()
    attributes: Mapping[str, MetadataValue] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _validate_non_empty(self.event_id, "event_id")
        _validate_non_empty(self.event_type, "event_type")

        if not isinstance(self.status, EventStatus):
            raise ValueError("status must be an EventStatus value.")

        _validate_frame_index(self.frame_start, "frame_start")
        _validate_frame_index(self.frame_end, "frame_end")

        if self.frame_start > self.frame_end:
            raise ValueError(
                "frame_start cannot be greater than frame_end."
            )

        _validate_timestamp(self.timestamp_start, "timestamp_start")
        _validate_timestamp(self.timestamp_end, "timestamp_end")

        if self.timestamp_start > self.timestamp_end:
            raise ValueError(
                "timestamp_start cannot be greater than timestamp_end."
            )

        if not isinstance(self.entity_ids, tuple):
            raise ValueError("entity_ids must be a tuple.")

        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise ValueError("entity_ids cannot contain duplicates.")

        for entity_id in self.entity_ids:
            _validate_non_empty(entity_id, "entity_id")

        if not isinstance(self.evidence, tuple):
            raise ValueError("evidence must be a tuple.")

        for item in self.evidence:
            if not isinstance(item, Evidence):
                raise ValueError(
                    "Every evidence item must be an Evidence instance."
                )

        if self.confidence is not None:
            _validate_confidence(self.confidence)

        if self.status == EventStatus.DETECTED:
            if self.confidence is None:
                raise ValueError(
                    "Detected events must include a confidence score."
                )

            if not self.evidence:
                raise ValueError(
                    "Detected events must include supporting evidence."
                )

        elif self.confidence is not None:
            raise ValueError(
                "Confidence must be None when the event "
                "is not detected."
            )

        object.__setattr__(
            self,
            "attributes",
            _freeze_attributes(self.attributes),
        )
