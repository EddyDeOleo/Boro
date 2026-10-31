
"""Video observation context and coverage metadata."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class CoverageScope(str, Enum):
    COMPLETE_TASK = "COMPLETE_TASK"
    PARTIAL_SEGMENT = "PARTIAL_SEGMENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ObservationContext:
    fps: float
    frame_count: int
    frame_start: int
    frame_end: int
    time_offset_seconds: float = 0.0
    coverage_scope: CoverageScope = CoverageScope.UNKNOWN

    def __post_init__(self) -> None:
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)):
            raise ValueError("fps must be numeric.")
        if not isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and greater than zero.")

        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int):
            raise ValueError("frame_count must be an integer.")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be greater than zero.")

        for field_name, value in (
            ("frame_start", self.frame_start),
            ("frame_end", self.frame_end),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative.")

        if self.frame_start > self.frame_end:
            raise ValueError("frame_start cannot exceed frame_end.")

        if self.frame_end >= self.frame_count:
            raise ValueError("frame_end must be smaller than frame_count.")

        if (
            isinstance(self.time_offset_seconds, bool)
            or not isinstance(self.time_offset_seconds, (int, float))
        ):
            raise ValueError("time_offset_seconds must be numeric.")

        if (
            not isfinite(self.time_offset_seconds)
            or self.time_offset_seconds < 0
        ):
            raise ValueError(
                "time_offset_seconds must be finite and non-negative."
            )

        if not isinstance(self.coverage_scope, CoverageScope):
            raise ValueError("coverage_scope must be a CoverageScope value.")

    @property
    def observed_frame_count(self) -> int:
        return self.frame_end - self.frame_start + 1

    @property
    def duration_seconds(self) -> float:
        return (self.frame_end - self.frame_start) / self.fps

    @property
    def start_time_seconds(self) -> float:
        return self.time_offset_seconds + self.frame_start / self.fps

    @property
    def end_time_seconds(self) -> float:
        return self.time_offset_seconds + self.frame_end / self.fps

    @property
    def covers_entire_video(self) -> bool:
        return (
            self.frame_start == 0
            and self.frame_end == self.frame_count - 1
        )
