
"""Presentation model for traceable execution evidence."""

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from src.models.evidence import MetadataValue


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


def _freeze_metrics(
    metrics: Mapping[str, MetadataValue],
) -> Mapping[str, MetadataValue]:
    """Validate and freeze evidence metrics."""

    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be a mapping.")

    copied_metrics = dict(metrics)

    for key, value in copied_metrics.items():
        _validate_non_empty(key, "metric key")

        if value is not None and not isinstance(
            value,
            (float, int, str, bool),
        ):
            raise ValueError(
                f"Unsupported metric value for {key!r}."
            )

        if isinstance(value, float) and not isfinite(value):
            raise ValueError(
                f"Metric {key!r} must contain a finite float."
            )

    return MappingProxyType(copied_metrics)


@dataclass(frozen=True)
class EvidenceView:
    """Presentation-ready representation of execution evidence."""

    evidence_id: str
    evidence_type: str
    description: str

    frame_start: int
    frame_end: int

    timestamp_start_seconds: float
    timestamp_end_seconds: float

    observation_ids: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()

    metrics: Mapping[str, MetadataValue] = field(
        default_factory=dict
    )
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate evidence presentation data."""

        _validate_non_empty(self.evidence_id, "evidence_id")
        _validate_non_empty(self.evidence_type, "evidence_type")
        _validate_non_empty(self.description, "description")

        _validate_frame_index(self.frame_start, "frame_start")
        _validate_frame_index(self.frame_end, "frame_end")

        if self.frame_start > self.frame_end:
            raise ValueError(
                "frame_start cannot exceed frame_end."
            )

        _validate_timestamp(
            self.timestamp_start_seconds,
            "timestamp_start_seconds",
        )
        _validate_timestamp(
            self.timestamp_end_seconds,
            "timestamp_end_seconds",
        )

        if (
            self.timestamp_start_seconds
            > self.timestamp_end_seconds
        ):
            raise ValueError(
                "timestamp_start_seconds cannot exceed "
                "timestamp_end_seconds."
            )

        self._validate_identifiers(
            self.observation_ids,
            "observation_ids",
        )
        self._validate_identifiers(
            self.entity_ids,
            "entity_ids",
        )

        if not isinstance(self.limitations, tuple):
            raise ValueError("limitations must be a tuple.")

        for limitation in self.limitations:
            _validate_non_empty(limitation, "limitation")

        if len(self.limitations) != len(set(self.limitations)):
            raise ValueError("limitations cannot contain duplicates.")

        object.__setattr__(
            self,
            "metrics",
            _freeze_metrics(self.metrics),
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
                f"{field_name} cannot contain duplicates."
            )
