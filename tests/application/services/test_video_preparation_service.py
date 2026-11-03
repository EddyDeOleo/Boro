
"""Tests for VideoPreparationService."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.models.observation_context import (
    CoverageScope,
    ObservationContext,
)
from src.models.video_metadata import VideoMetadata
from src.application.services.video_preparation_service import (
    PreparedVideo,
    VideoPreparationService,
)


class FakeVideoProcessor:
    """Test double for VideoProcessor."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        self.metadata = VideoMetadata(
            width=640,
            height=480,
            fps=30.0,
            frame_count=10,
        )
        self.is_open = True
        self.closed = False
        self.iteration_arguments: dict[str, int] | None = None

    @property
    def source_path(self) -> Path:
        """Return the absolute path of the source video."""
        return self.video_path.resolve()

    def iter_frames(
        self,
        start_frame: int = 0,
        end_frame: int | None = None,
    ):
        self.iteration_arguments = {
            "start_frame": start_frame,
            "end_frame": end_frame,
        }

        if end_frame is None:
            end_frame = self.metadata.frame_count - 1

        for frame_index in range(start_frame, end_frame + 1):
            yield frame_index, np.zeros(
                (self.metadata.height, self.metadata.width, 3),
                dtype=np.uint8,
            )

    def close(self) -> None:
        self.closed = True
        self.is_open = False


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    path = tmp_path / "test_video.mp4"
    path.touch()
    return path


@pytest.fixture
def processor_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> list[FakeVideoProcessor]:
    instances: list[FakeVideoProcessor] = []

    def create_processor(
        video_path: str | Path,
    ) -> FakeVideoProcessor:
        processor = FakeVideoProcessor(video_path)
        instances.append(processor)
        return processor

    monkeypatch.setattr(
        "src.application.services.video_preparation_service.VideoProcessor",
        create_processor,
    )

    return instances


def test_prepare_returns_prepared_video(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    prepared = VideoPreparationService().prepare(video_file)

    assert isinstance(prepared, PreparedVideo)
    assert prepared.is_open
    assert prepared.metadata.width == 640
    assert prepared.metadata.height == 480
    assert prepared.metadata.fps == 30.0
    assert prepared.metadata.frame_count == 10

    prepared.close()


def test_prepare_builds_observation_context(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    prepared = VideoPreparationService().prepare(video_file)

    assert prepared.context == ObservationContext(
        fps=30.0,
        frame_count=10,
        frame_start=0,
        frame_end=9,
        time_offset_seconds=0.0,
        coverage_scope=CoverageScope.UNKNOWN,
    )

    prepared.close()


def test_prepare_uses_requested_coverage_scope(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    prepared = VideoPreparationService().prepare(
        video_file,
        coverage_scope=CoverageScope.COMPLETE_TASK,
    )

    assert (
        prepared.context.coverage_scope
        == CoverageScope.COMPLETE_TASK
    )

    prepared.close()


def test_prepare_uses_requested_frame_range(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    prepared = VideoPreparationService().prepare(
        video_file,
        frame_start=2,
        frame_end=6,
    )

    assert prepared.context.frame_start == 2
    assert prepared.context.frame_end == 6
    assert prepared.context.observed_frame_count == 5

    prepared.close()


def test_prepare_uses_requested_time_offset(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    prepared = VideoPreparationService().prepare(
        video_file,
        time_offset_seconds=5.8,
    )

    assert prepared.context.time_offset_seconds == 5.8
    assert prepared.context.start_time_seconds == 5.8
    assert prepared.context.end_time_seconds == pytest.approx(6.1)

    prepared.close()


def test_prepare_defaults_to_all_video_frames(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    prepared = VideoPreparationService().prepare(video_file)

    assert prepared.context.frame_start == 0
    assert prepared.context.frame_end == 9
    assert prepared.context.covers_entire_video

    prepared.close()


def test_prepare_preserves_partial_video_coverage(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    prepared = VideoPreparationService().prepare(
        video_file,
        coverage_scope=CoverageScope.PARTIAL_SEGMENT,
    )

    assert (
        prepared.context.coverage_scope
        == CoverageScope.PARTIAL_SEGMENT
    )
    assert prepared.context.covers_entire_video

    prepared.close()


def test_prepare_does_not_infer_complete_task_coverage(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    prepared = VideoPreparationService().prepare(video_file)

    assert prepared.context.coverage_scope == CoverageScope.UNKNOWN

    prepared.close()


def test_prepared_video_iterates_context_frame_range(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    prepared = VideoPreparationService().prepare(
        video_file,
        frame_start=2,
        frame_end=4,
    )

    frames = list(prepared.iter_frames())

    assert [index for index, _ in frames] == [2, 3, 4]
    assert processor_instances[0].iteration_arguments == {
        "start_frame": 2,
        "end_frame": 4,
    }

    prepared.close()


def test_prepared_video_frames_have_expected_dimensions(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    prepared = VideoPreparationService().prepare(video_file)

    frames = list(prepared.iter_frames())

    assert len(frames) == 10

    for _, frame in frames:
        assert frame.shape == (480, 640, 3)

    prepared.close()


@pytest.mark.parametrize(
    ("kwargs", "expected_field"),
    [
        ({"frame_start": -1}, "frame_start"),
        ({"frame_start": 10}, "frame_start"),
        ({"frame_start": 5, "frame_end": 4}, "frame_end"),
        ({"frame_end": 10}, "frame_end"),
        ({"frame_start": True}, "frame_start"),
        ({"frame_end": 2.5}, "frame_end"),
        ({"time_offset_seconds": -1.0}, "time_offset_seconds"),
        (
            {"time_offset_seconds": float("nan")},
            "time_offset_seconds",
        ),
    ],
)
def test_prepare_rejects_invalid_observation_context(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
    kwargs: dict[str, Any],
    expected_field: str,
) -> None:
    with pytest.raises(ValueError, match=expected_field):
        VideoPreparationService().prepare(
            video_file,
            **kwargs,
        )

    assert processor_instances[0].closed


def test_prepare_rejects_invalid_coverage_scope(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    with pytest.raises(ValueError, match="coverage_scope"):
        VideoPreparationService().prepare(
            video_file,
            coverage_scope="COMPLETE_TASK",
        )

    assert processor_instances[0].closed


def test_prepared_video_context_manager_releases_resource(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    with VideoPreparationService().prepare(video_file) as prepared:
        assert prepared.is_open

    assert not prepared.is_open
    assert processor_instances[0].closed


def test_prepared_video_close_is_idempotent(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    prepared = VideoPreparationService().prepare(video_file)

    prepared.close()
    prepared.close()

    assert not prepared.is_open
    assert processor_instances[0].closed


def test_prepared_video_rejects_iteration_after_close(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    prepared = VideoPreparationService().prepare(video_file)
    prepared.close()

    with pytest.raises(RuntimeError, match="closed"):
        list(prepared.iter_frames())


def test_prepared_video_rejects_context_entry_after_close(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    prepared = VideoPreparationService().prepare(video_file)
    prepared.close()

    with pytest.raises(RuntimeError, match="closed"):
        prepared.__enter__()


def test_prepared_video_exposes_source_path(
    video_file: Path,
    processor_instances: list[FakeVideoProcessor],
) -> None:
    with VideoPreparationService().prepare(
        video_file
    ) as prepared_video:
        assert prepared_video.source_path.is_absolute()
        assert prepared_video.source_path == video_file.resolve()
