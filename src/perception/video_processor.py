
"""Video file validation, metadata extraction, and frame iteration."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from src.models.video_metadata import VideoMetadata


class VideoProcessingError(RuntimeError):
    """Raised when a video cannot be processed reliably."""


class VideoProcessor:
    """Open a video and provide sequential access to its frames."""

    def __init__(self, video_path: str | Path) -> None:
        """Open the video and extract its metadata."""

        self.video_path = Path(video_path)
        self._capture: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video file does not exist: {self.video_path}"
            )

        if not self.video_path.is_file():
            raise ValueError(
                f"Video path is not a file: {self.video_path}"
            )

        capture = cv2.VideoCapture(str(self.video_path))

        if not capture.isOpened():
            capture.release()
            raise VideoProcessingError(
                f"Unable to open video file: {self.video_path}"
            )

        self._capture = capture

        try:
            self._metadata = self._extract_metadata()
        except Exception:
            self.close()
            raise

    @property
    def metadata(self) -> VideoMetadata:
        """Return the extracted video metadata."""

        if self._metadata is None:
            raise VideoProcessingError(
                "Video metadata is not available."
            )

        return self._metadata

    @property
    def source_path(self) -> Path:
        """Return the absolute path of the source video."""

        return self.video_path.resolve()

    @property
    def is_open(self) -> bool:
        """Return whether the video capture is currently open."""

        return self._capture is not None

    def _extract_metadata(self) -> VideoMetadata:
        """Read and validate metadata reported by OpenCV."""

        capture = self._require_capture()

        raw_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        raw_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        raw_fps = capture.get(cv2.CAP_PROP_FPS)
        raw_frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        numeric_values = {
            "width": raw_width,
            "height": raw_height,
            "fps": raw_fps,
            "frame_count": raw_frame_count,
        }

        for field_name, value in numeric_values.items():
            if not np.isfinite(value):
                raise VideoProcessingError(
                    f"Video metadata field '{field_name}' "
                    "is not finite."
                )

        if raw_width <= 0:
            raise VideoProcessingError(
                "Video width must be greater than zero."
            )

        if raw_height <= 0:
            raise VideoProcessingError(
                "Video height must be greater than zero."
            )

        if raw_fps <= 0:
            raise VideoProcessingError(
                "Video FPS must be greater than zero."
            )

        if raw_frame_count <= 0:
            raise VideoProcessingError(
                "Video frame count must be greater than zero."
            )

        try:
            return VideoMetadata(
                width=int(round(raw_width)),
                height=int(round(raw_height)),
                fps=float(raw_fps),
                frame_count=int(round(raw_frame_count)),
            )
        except ValueError as exc:
            raise VideoProcessingError(
                f"Invalid video metadata: {exc}"
            ) from exc

    def iter_frames(
        self,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """
        Yield decoded frames sequentially.

        Frame indices are zero-based and end_frame is inclusive.
        """

        capture = self._require_capture()
        metadata = self.metadata

        self._validate_frame_range(
            start_frame=start_frame,
            end_frame=end_frame,
            frame_count=metadata.frame_count,
        )

        if end_frame is None:
            end_frame = metadata.frame_count - 1

        seek_success = capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame,
        )

        if not seek_success:
            raise VideoProcessingError(
                f"Unable to seek to frame {start_frame}."
            )

        for frame_index in range(start_frame, end_frame + 1):
            success, frame = capture.read()

            if not success or frame is None:
                raise VideoProcessingError(
                    f"Unable to decode frame {frame_index}."
                )

            if (
                frame.ndim < 2
                or frame.shape[0] != metadata.height
                or frame.shape[1] != metadata.width
            ):
                raise VideoProcessingError(
                    f"Decoded frame {frame_index} has unexpected "
                    f"dimensions: {frame.shape}."
                )

            yield frame_index, frame

    @staticmethod
    def _validate_frame_range(
        start_frame: int,
        end_frame: int | None,
        frame_count: int,
    ) -> None:
        """Validate a requested zero-based frame range."""

        if isinstance(start_frame, bool) or not isinstance(
            start_frame, int
        ):
            raise ValueError("start_frame must be an integer.")

        if start_frame < 0 or start_frame >= frame_count:
            raise ValueError(
                "start_frame must be within the video frame range."
            )

        if end_frame is not None:
            if isinstance(end_frame, bool) or not isinstance(
                end_frame, int
            ):
                raise ValueError("end_frame must be an integer.")

            if end_frame < start_frame:
                raise ValueError(
                    "end_frame cannot be smaller than start_frame."
                )

            if end_frame >= frame_count:
                raise ValueError(
                    "end_frame must be within the video frame range."
                )

    def _require_capture(self) -> cv2.VideoCapture:
        """Return the active capture or raise if it has been closed."""

        if self._capture is None:
            raise VideoProcessingError(
                "Video capture is closed."
            )

        return self._capture

    def close(self) -> None:
        """Release the underlying OpenCV video capture."""

        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoProcessor":
        """Support usage as a context manager."""

        self._require_capture()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Release video resources when leaving the context."""

        self.close()
