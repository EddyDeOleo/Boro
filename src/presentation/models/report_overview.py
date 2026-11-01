
"""Presentation model for robotic execution report overview."""

from dataclasses import dataclass
from math import isfinite


def _validate_non_negative(value: float, field_name: str) -> None:
    """Validate a finite non-negative numeric value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")

    if not isfinite(value) or value < 0:
        raise ValueError(
            f"{field_name} must be finite and non-negative."
        )


def _validate_positive(value: float, field_name: str) -> None:
    """Validate a finite positive numeric value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")

    if not isfinite(value) or value <= 0:
        raise ValueError(
            f"{field_name} must be finite and greater than zero."
        )


def _validate_positive_integer(value: int, field_name: str) -> None:
    """Validate a positive integer value."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")


@dataclass(frozen=True)
class ReportOverview:
    """Summary and temporal coverage information for a report."""

    verdict: str
    summary: str
    coverage_scope: str
    covers_entire_context: bool

    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float

    fps: float
    context_frame_count: int
    observed_frame_count: int

    def __post_init__(self) -> None:
        """Validate overview data."""

        for field_name, value in (
            ("verdict", self.verdict),
            ("summary", self.summary),
            ("coverage_scope", self.coverage_scope),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string."
                )

        if not isinstance(self.covers_entire_context, bool):
            raise ValueError(
                "covers_entire_context must be a boolean."
            )

        _validate_non_negative(
            self.start_time_seconds,
            "start_time_seconds",
        )
        _validate_non_negative(
            self.end_time_seconds,
            "end_time_seconds",
        )
        _validate_non_negative(
            self.duration_seconds,
            "duration_seconds",
        )
        _validate_positive(self.fps, "fps")

        if self.start_time_seconds > self.end_time_seconds:
            raise ValueError(
                "start_time_seconds cannot exceed end_time_seconds."
            )

        if self.duration_seconds > (
            self.end_time_seconds - self.start_time_seconds
        ) + 1e-9:
            raise ValueError(
                "duration_seconds cannot exceed the observation interval."
            )

        _validate_positive_integer(
            self.context_frame_count,
            "context_frame_count",
        )
        _validate_positive_integer(
            self.observed_frame_count,
            "observed_frame_count",
        )

        if self.observed_frame_count > self.context_frame_count:
            raise ValueError(
                "observed_frame_count cannot exceed context_frame_count."
            )
