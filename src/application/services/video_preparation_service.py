
"""Prepare video resources and observation context for Boro."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src.models.observation_context import (
    CoverageScope,
    ObservationContext,
)
from src.models.video_metadata import VideoMetadata
from src.perception.video_processor import VideoProcessor


class PreparedVideo:
    """Hold an open video processor and its observation context."""

    def __init__(
        self,
        processor: VideoProcessor,
        context: ObservationContext,
    ) -> None:
        """Initialize a prepared video resource."""

        self._processor = processor
        self.context = context
        self._closed = False

    @property
    def metadata(self) -> VideoMetadata:
        """Return the technical metadata of the prepared video."""

        return self._processor.metadata

    @property
    def source_path(self) -> Path:
        """Return the absolute path of the source video."""

        return self._processor.source_path

    @property
    def is_open(self) -> bool:
        """Return whether the prepared video resource is open."""

        return not self._closed and self._processor.is_open

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over the frames selected by the observation context."""

        if self._closed:
            raise RuntimeError("Prepared video is closed.")

        return self._processor.iter_frames(
            start_frame=self.context.frame_start,
            end_frame=self.context.frame_end,
        )

    def close(self) -> None:
        """Release the underlying video resource."""

        if not self._closed:
            self._processor.close()
            self._closed = True

    def __enter__(self) -> "PreparedVideo":
        """Support usage as a context manager."""

        if self._closed:
            raise RuntimeError("Prepared video is closed.")

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Release the video resource when leaving the context."""

        self.close()


class VideoPreparationService:
    """Prepare video resources and their observation metadata."""

    def prepare(
        self,
        video_path: str | Path,
        coverage_scope: CoverageScope = CoverageScope.UNKNOWN,
        frame_start: int = 0,
        frame_end: int | None = None,
        time_offset_seconds: float = 0.0,
    ) -> PreparedVideo:
        """
        Open a video and create its observation context.

        Frame indices are zero-based and frame_end is inclusive.
        The coverage scope must be explicitly provided when known.
        """

        processor = VideoProcessor(video_path)

        try:
            metadata = processor.metadata

            if frame_end is None:
                frame_end = metadata.frame_count - 1

            context = ObservationContext(
                fps=metadata.fps,
                frame_count=metadata.frame_count,
                frame_start=frame_start,
                frame_end=frame_end,
                time_offset_seconds=time_offset_seconds,
                coverage_scope=coverage_scope,
            )

            return PreparedVideo(
                processor=processor,
                context=context,
            )

        except Exception:
            processor.close()
            raise
