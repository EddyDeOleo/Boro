
"""Presentation model for temporal execution events."""

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from src.models.evidence import MetadataValue
from src.models.event import EventStatus


def _validate_non_empty(value: str, field_name: str) -> None:
    """Validate a non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string."
        )


def _validate_frame_index(value: int, field_name: str) -> None:
    """Validate a non-negative frame index."""

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


def _freeze_attributes(
    attributes: Mapping[str, MetadataValue],
) -> Mapping[str, MetadataValue]:
    """Validate and freeze event attributes."""

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
class TimelineEventView:
    """Presentation-ready representation of a temporal event."""

    event_id: str
    event_type: str
    status: str

    frame_start: int
    frame_end: int

    timestamp_start: float
    timestamp_end: float

    entity_ids: tuple[str, ...]
    confidence: float | None

    evidence_ids: tuple[str, ...] = ()
    attributes: Mapping[str, MetadataValue] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate timeline event presentation data."""

        _validate_non_empty(self.event_id, "event_id")
        _validate_non_empty(self.event_type, "event_type")

        if not isinstance(self.status, str) or self.status not in {
            status.value for status in EventStatus
        }:
            raise ValueError("Invalid event status.")

        _validate_frame_index(self.frame_start, "frame_start")
        _validate_frame_index(self.frame_end, "frame_end")

        if self.frame_start > self.frame_end:
            raise ValueError(
                "frame_start cannot exceed frame_end."
            )

        _validate_timestamp(
            self.timestamp_start,
            "timestamp_start",
        )
        _validate_timestamp(
            self.timestamp_end,
            "timestamp_end",
        )

        if self.timestamp_start > self.timestamp_end:
            raise ValueError(
                "timestamp_start cannot exceed timestamp_end."
            )

        self._validate_identifiers(self.entity_ids, "entity_ids")
        self._validate_identifiers(self.evidence_ids, "evidence_ids")

        if self.confidence is not None:
            if (
                isinstance(self.confidence, bool)
                or not isinstance(self.confidence, (int, float))
            ):
                raise ValueError("confidence must be numeric.")

            if (
                not isfinite(self.confidence)
                or not 0.0 <= self.confidence <= 1.0
            ):
                raise ValueError(
                    "confidence must be between 0 and 1."
                )

        if self.status == EventStatus.DETECTED.value:
            if self.confidence is None:
                raise ValueError(
                    "Detected events must include confidence."
                )

            if not self.evidence_ids:
                raise ValueError(
                    "Detected events must include evidence references."
                )

        elif self.confidence is not None:
            raise ValueError(
                "Confidence must be None for non-detected events."
            )

        object.__setattr__(
            self,
            "attributes",
            _freeze_attributes(self.attributes),
        )

    @staticmethod
    def _validate_identifiers(
        identifiers: tuple[str, ...],
        field_name: str,
    ) -> None:
        """Validate unique, non-empty identifier tuples."""

        if not isinstance(identifiers, tuple):
            raise ValueError(f"{field_name} must be a tuple.")

        for identifier in identifiers:
            _validate_non_empty(identifier, field_name)

        if len(identifiers) != len(set(identifiers)):
            raise ValueError(
                f"{field_name} must not contain duplicates."
            )
