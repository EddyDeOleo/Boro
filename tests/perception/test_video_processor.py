
"""Tests for VideoProcessor."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.perception.video_processor import (
    VideoProcessingError,
    VideoProcessor,
)


class FakeVideoCapture:
    """Minimal OpenCV VideoCapture test double."""

    def __init__(
        self,
        frames: list[np.ndarray] | None = None,
        metadata: dict[int, float] | None = None,
        opened: bool = True,
        seek_success: bool = True,
    ) -> None:
        self.frames = frames or []
        self.metadata = metadata or {}
        self.opened = opened
        self.seek_success = seek_success
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def get(self, property_id: int) -> float:
        return self.metadata.get(property_id, 0.0)

    def set(self, property_id: int, value: float) -> bool:
        if property_id != cv2.CAP_PROP_POS_FRAMES:
            return False

        if not self.seek_success:
            return False

        self.position = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.position >= len(self.frames):
            return False, None

        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy()

    def release(self) -> None:
        self.released = True


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    path = tmp_path / "test_video.mp4"
    path.touch()
    return path


@pytest.fixture
def video_frames() -> list[np.ndarray]:
    return [
        np.full((480, 640, 3), index, dtype=np.uint8)
        for index in range(5)
    ]


@pytest.fixture
def video_metadata() -> dict[int, float]:
    return {
        cv2.CAP_PROP_FRAME_WIDTH: 640.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 5.0,
    }


def install_capture(
    monkeypatch: pytest.MonkeyPatch,
    capture: FakeVideoCapture,
) -> None:
    monkeypatch.setattr(
        "src.perception.video_processor.cv2.VideoCapture",
        lambda _: capture,
    )


def test_video_processor_extracts_metadata(
    video_file: Path,
    video_metadata: dict[int, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeVideoCapture(metadata=video_metadata)
    install_capture(monkeypatch, capture)

    processor = VideoProcessor(video_file)

    assert processor.metadata.width == 640
    assert processor.metadata.height == 480
    assert processor.metadata.fps == 30.0
    assert processor.metadata.frame_count == 5
    assert processor.metadata.duration_seconds == pytest.approx(
        5 / 30
    )

    processor.close()


def test_video_processor_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_file = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError):
        VideoProcessor(missing_file)


def test_video_processor_rejects_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        VideoProcessor(tmp_path)


def test_video_processor_rejects_unopenable_file(
    video_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeVideoCapture(opened=False)
    install_capture(monkeypatch, capture)

    with pytest.raises(VideoProcessingError):
        VideoProcessor(video_file)

    assert capture.released


@pytest.mark.parametrize(
    ("property_id", "value"),
    [
        (cv2.CAP_PROP_FRAME_WIDTH, 0.0),
        (cv2.CAP_PROP_FRAME_HEIGHT, 0.0),
        (cv2.CAP_PROP_FPS, 0.0),
        (cv2.CAP_PROP_FRAME_COUNT, 0.0),
        (cv2.CAP_PROP_FPS, float("nan")),
        (cv2.CAP_PROP_FRAME_COUNT, float("inf")),
    ],
)
def test_video_processor_rejects_invalid_metadata(
    video_file: Path,
    video_metadata: dict[int, float],
    property_id: int,
    value: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_metadata = dict(video_metadata)
    invalid_metadata[property_id] = value

    capture = FakeVideoCapture(metadata=invalid_metadata)
    install_capture(monkeypatch, capture)

    with pytest.raises(VideoProcessingError):
        VideoProcessor(video_file)

    assert capture.released


def test_video_processor_iterates_all_frames(
    video_file: Path,
    video_frames: list[np.ndarray],
    video_metadata: dict[int, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeVideoCapture(
        frames=video_frames,
        metadata=video_metadata,
    )
    install_capture(monkeypatch, capture)

    with VideoProcessor(video_file) as processor:
        results = list(processor.iter_frames())

        assert [index for index, _ in results] == [
            0, 1, 2, 3, 4
        ]
        assert len(results) == 5
        assert np.array_equal(results[2][1], video_frames[2])

    assert capture.released


def test_video_processor_iterates_requested_frame_range(
    video_file: Path,
    video_frames: list[np.ndarray],
    video_metadata: dict[int, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeVideoCapture(
        frames=video_frames,
        metadata=video_metadata,
    )
    install_capture(monkeypatch, capture)

    with VideoProcessor(video_file) as processor:
        results = list(
            processor.iter_frames(
                start_frame=1,
                end_frame=3,
            )
        )

    assert [index for index, _ in results] == [1, 2, 3]


@pytest.mark.parametrize(
    ("start_frame", "end_frame"),
    [
        (-1, 2),
        (5, None),
        (3, 2),
        (0, 5),
        (True, 2),
        (0, 2.5),
    ],
)
def test_video_processor_rejects_invalid_frame_ranges(
    video_file: Path,
    video_metadata: dict[int, float],
    start_frame: Any,
    end_frame: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeVideoCapture(metadata=video_metadata)
    install_capture(monkeypatch, capture)

    with VideoProcessor(video_file) as processor:
        with pytest.raises(ValueError):
            list(
                processor.iter_frames(
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )


def test_video_processor_rejects_seek_failure(
    video_file: Path,
    video_metadata: dict[int, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeVideoCapture(
        metadata=video_metadata,
        seek_success=False,
    )
    install_capture(monkeypatch, capture)

    with VideoProcessor(video_file) as processor:
        with pytest.raises(VideoProcessingError):
            list(processor.iter_frames())


def test_video_processor_rejects_decode_failure(
    video_file: Path,
    video_metadata: dict[int, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeVideoCapture(
        frames=[],
        metadata=video_metadata,
    )
    install_capture(monkeypatch, capture)

    with VideoProcessor(video_file) as processor:
        with pytest.raises(VideoProcessingError):
            list(processor.iter_frames())


def test_video_processor_rejects_unexpected_frame_dimensions(
    video_file: Path,
    video_metadata: dict[int, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    capture = FakeVideoCapture(
        frames=[wrong_frame],
        metadata=video_metadata,
    )
    install_capture(monkeypatch, capture)

    with VideoProcessor(video_file) as processor:
        with pytest.raises(VideoProcessingError):
            list(processor.iter_frames(end_frame=0))


def test_video_processor_rejects_operations_after_close(
    video_file: Path,
    video_metadata: dict[int, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeVideoCapture(metadata=video_metadata)
    install_capture(monkeypatch, capture)

    processor = VideoProcessor(video_file)
    processor.close()

    assert not processor.is_open

    with pytest.raises(VideoProcessingError):
        list(processor.iter_frames())

    with pytest.raises(VideoProcessingError):
        processor.__enter__()

    processor.close()
    assert capture.released


def test_source_path_returns_absolute_path(
    video_file: Path,
    video_metadata: dict[int, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeVideoCapture(metadata=video_metadata)
    install_capture(monkeypatch, capture)

    processor = VideoProcessor(video_file)

    try:
        assert processor.source_path.is_absolute()
        assert processor.source_path == video_file.resolve()
    finally:
        processor.close()