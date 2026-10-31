
"""Generic observation and evidence models for robotic execution analysis."""

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping, TypeAlias


Point2D: TypeAlias = tuple[float, float]
BoundingBox: TypeAlias = tuple[float, float, float, float]
MetadataValue: TypeAlias = float | int | str | bool | None


def _validate_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be empty.")


def _validate_frame_index(frame_index: int) -> None:
    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise ValueError("frame_index must be an integer.")

    if frame_index < 0:
        raise ValueError("frame_index cannot be negative.")


def _validate_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")

    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite.")


def _validate_non_negative(value: float, field_name: str) -> None:
    _validate_finite(value, field_name)

    if value < 0:
        raise ValueError(f"{field_name} cannot be negative.")


def _validate_point(point: Point2D, field_name: str) -> None:
    if len(point) != 2:
        raise ValueError(f"{field_name} must contain two coordinates.")

    for coordinate in point:
        _validate_finite(coordinate, field_name)


def _freeze_metadata(
    metadata: Mapping[str, MetadataValue],
    field_name: str,
) -> Mapping[str, MetadataValue]:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")

    copied_metadata = dict(metadata)

    for key, value in copied_metadata.items():
        _validate_non_empty(key, f"{field_name} key")

        if value is not None and not isinstance(
            value, (float, int, str, bool)
        ):
            raise ValueError(f"Unsupported value in {field_name}: {key}")

        if isinstance(value, float):
            _validate_finite(value, f"{field_name}[{key}]")

    return MappingProxyType(copied_metadata)


@dataclass(frozen=True)
class EntityObservation:
    """Visual measurements of an entity in a specific video frame."""

    observation_id: str
    entity_id: str
    frame_index: int
    timestamp_seconds: float

    bbox: BoundingBox | None = None
    centroid: Point2D | None = None
    area_pixels: float | None = None
    confidence: float | None = None

    attributes: Mapping[str, MetadataValue] = field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty(self.observation_id, "observation_id")
        _validate_non_empty(self.entity_id, "entity_id")
        _validate_frame_index(self.frame_index)
        _validate_non_negative(self.timestamp_seconds, "timestamp_seconds")

        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError("bbox must contain four coordinates.")

            for coordinate in self.bbox:
                _validate_finite(coordinate, "bbox")

            x_min, y_min, x_max, y_max = self.bbox

            if x_min > x_max or y_min > y_max:
                raise ValueError("bbox coordinates are invalid.")

        if self.centroid is not None:
            _validate_point(self.centroid, "centroid")

        if self.area_pixels is not None:
            _validate_non_negative(self.area_pixels, "area_pixels")

        if self.confidence is not None:
            _validate_finite(self.confidence, "confidence")

            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1.")

        if self.source is not None:
            _validate_non_empty(self.source, "source")

        object.__setattr__(
            self,
            "attributes",
            _freeze_metadata(self.attributes, "attributes"),
        )


@dataclass(frozen=True)
class SpatialRelationship:
    """Geometric relationship between two entities in one frame."""

    frame_index: int
    entity_a_id: str
    entity_b_id: str

    distance_pixels: float | None = None
    delta_x_pixels: float | None = None
    delta_y_pixels: float | None = None

    intersection_area_pixels: float | None = None
    iou: float | None = None

    overlap_ratio_a: float | None = None
    overlap_ratio_b: float | None = None

    attributes: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_frame_index(self.frame_index)
        _validate_non_empty(self.entity_a_id, "entity_a_id")
        _validate_non_empty(self.entity_b_id, "entity_b_id")

        if self.entity_a_id == self.entity_b_id:
            raise ValueError("An entity cannot have a relationship with itself.")

        non_negative_fields = {
            "distance_pixels": self.distance_pixels,
            "intersection_area_pixels": self.intersection_area_pixels,
        }

        for field_name, value in non_negative_fields.items():
            if value is not None:
                _validate_non_negative(value, field_name)

        signed_fields = {
            "delta_x_pixels": self.delta_x_pixels,
            "delta_y_pixels": self.delta_y_pixels,
        }

        for field_name, value in signed_fields.items():
            if value is not None:
                _validate_finite(value, field_name)

        bounded_fields = {
            "iou": self.iou,
            "overlap_ratio_a": self.overlap_ratio_a,
            "overlap_ratio_b": self.overlap_ratio_b,
        }

        for field_name, value in bounded_fields.items():
            if value is not None:
                _validate_finite(value, field_name)

                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{field_name} must be between 0 and 1.")

        object.__setattr__(
            self,
            "attributes",
            _freeze_metadata(self.attributes, "attributes"),
        )


@dataclass(frozen=True)
class Evidence:
    """Traceable evidence supporting an event or verification result."""

    evidence_id: str
    evidence_type: str
    description: str

    frame_start: int
    frame_end: int

    observation_ids: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()

    metrics: Mapping[str, MetadataValue] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_non_empty(self.evidence_id, "evidence_id")
        _validate_non_empty(self.evidence_type, "evidence_type")
        _validate_non_empty(self.description, "description")

        _validate_frame_index(self.frame_start)
        _validate_frame_index(self.frame_end)

        if self.frame_start > self.frame_end:
            raise ValueError("frame_start cannot be greater than frame_end.")

        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("observation_ids cannot contain duplicates.")

        for observation_id in self.observation_ids:
            _validate_non_empty(observation_id, "observation_id")

        for entity_id in self.entity_ids:
            _validate_non_empty(entity_id, "entity_id")

        for limitation in self.limitations:
            _validate_non_empty(limitation, "limitation")

        object.__setattr__(
            self,
            "metrics",
            _freeze_metadata(self.metrics, "metrics"),
        )
