
"""Tests for the VideoMetadata model."""

import pytest

from src.models.video_metadata import VideoMetadata


def test_video_metadata_accepts_valid_values() -> None:
    metadata = VideoMetadata(
        width=1280,
        height=960,
        fps=30.0,
        frame_count=407,
    )

    assert metadata.width == 1280
    assert metadata.height == 960
    assert metadata.fps == 30.0
    assert metadata.frame_count == 407


def test_video_metadata_calculates_duration() -> None:
    metadata = VideoMetadata(
        width=1280,
        height=960,
        fps=30.0,
        frame_count=407,
    )

    assert metadata.duration_seconds == pytest.approx(407 / 30)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("width", 0),
        ("width", -1),
        ("height", 0),
        ("height", -1),
        ("frame_count", 0),
        ("frame_count", -1),
        ("width", True),
        ("height", 960.5),
        ("frame_count", 10.5),
    ],
)
def test_video_metadata_rejects_invalid_integer_fields(
    field_name: str,
    value: object,
) -> None:
    values = {
        "width": 1280,
        "height": 960,
        "fps": 30.0,
        "frame_count": 407,
    }
    values[field_name] = value

    with pytest.raises(ValueError):
        VideoMetadata(**values)


@pytest.mark.parametrize(
    "fps",
    [
        0,
        -1,
        float("inf"),
        float("-inf"),
        float("nan"),
        True,
        "30",
    ],
)
def test_video_metadata_rejects_invalid_fps(
    fps: object,
) -> None:
    with pytest.raises(ValueError):
        VideoMetadata(
            width=1280,
            height=960,
            fps=fps,
            frame_count=407,
        )
