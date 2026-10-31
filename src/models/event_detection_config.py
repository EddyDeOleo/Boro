
"""Configuration for temporal event detection."""

from dataclasses import dataclass
from math import isfinite


def _validate_positive_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")

    if not isfinite(value) or value <= 0:
        raise ValueError(
            f"{field_name} must be finite and greater than zero."
        )


def _validate_non_negative_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")

    if not isfinite(value) or value < 0:
        raise ValueError(
            f"{field_name} must be finite and non-negative."
        )


@dataclass(frozen=True)
class EventDetectionConfig:
    min_event_duration_seconds: float = 0.20
    min_observations: int = 3
    max_frame_gap: int = 1

    min_object_displacement_pixels: float = 5.0
    max_relative_distance_pixels: float = 30.0
    min_relative_separation_increase_pixels: float = 5.0

    min_motion_similarity: float = 0.75
    max_stationary_speed_pixels_per_second: float = 3.0

    min_region_overlap_ratio: float = 0.80
    min_stable_duration_seconds: float = 0.50

    def __post_init__(self) -> None:
        _validate_positive_number(
            self.min_event_duration_seconds,
            "min_event_duration_seconds",
        )
        _validate_positive_number(
            self.min_object_displacement_pixels,
            "min_object_displacement_pixels",
        )
        _validate_positive_number(
            self.max_relative_distance_pixels,
            "max_relative_distance_pixels",
        )
        _validate_positive_number(
            self.min_relative_separation_increase_pixels,
            "min_relative_separation_increase_pixels",
        )
        _validate_positive_number(
            self.max_stationary_speed_pixels_per_second,
            "max_stationary_speed_pixels_per_second",
        )
        _validate_positive_number(
            self.min_stable_duration_seconds,
            "min_stable_duration_seconds",
        )

        if (
            isinstance(self.min_observations, bool)
            or not isinstance(self.min_observations, int)
            or self.min_observations < 2
        ):
            raise ValueError("min_observations must be an integer >= 2.")

        if (
            isinstance(self.max_frame_gap, bool)
            or not isinstance(self.max_frame_gap, int)
            or self.max_frame_gap < 0
        ):
            raise ValueError("max_frame_gap must be a non-negative integer.")

        if (
            isinstance(self.min_motion_similarity, bool)
            or not isinstance(self.min_motion_similarity, (int, float))
            or not isfinite(self.min_motion_similarity)
            or not 0.0 <= self.min_motion_similarity <= 1.0
        ):
            raise ValueError(
                "min_motion_similarity must be between 0 and 1."
            )

        _validate_non_negative_number(
            self.min_region_overlap_ratio,
            "min_region_overlap_ratio",
        )

        if self.min_region_overlap_ratio > 1.0:
            raise ValueError("min_region_overlap_ratio cannot exceed 1.")
