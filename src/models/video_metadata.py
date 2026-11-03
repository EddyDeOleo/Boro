
"""Video metadata model."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class VideoMetadata:
    """Immutable technical metadata extracted from a video file."""

    width: int
    height: int
    fps: float
    frame_count: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("width", self.width),
            ("height", self.height),
            ("frame_count", self.frame_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer.")

            if value <= 0:
                raise ValueError(
                    f"{field_name} must be greater than zero."
                )

        if isinstance(self.fps, bool) or not isinstance(
            self.fps, (int, float)
        ):
            raise ValueError("fps must be numeric.")

        if not isfinite(self.fps) or self.fps <= 0:
            raise ValueError(
                "fps must be finite and greater than zero."
            )

    @property
    def duration_seconds(self) -> float:
        """Return the estimated video duration in seconds."""

        return self.frame_count / self.fps
