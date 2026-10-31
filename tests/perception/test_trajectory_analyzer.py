
from pathlib import Path

import pytest

from src.perception.sam_parser import parse_sam_file
from src.perception.trajectory_analyzer import (
    calculate_motion,
    detect_stable_intervals,
    summarize_trajectory,
    analyze_trajectory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAM_OUTPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "sam_episode_000025_raw.txt"
)


@pytest.fixture
def detections():
    if not SAM_OUTPUT.is_file():
        pytest.skip(
            "Real SAM evaluation output is not available."
        )

    return parse_sam_file(SAM_OUTPUT)


def test_calculate_motion(detections):
    samples = calculate_motion(detections, fps=30)

    assert len(samples) == 36

    first = samples[0]

    assert first.frame == 15
    assert first.delta_x is None
    assert first.delta_y is None

    second = samples[1]

    assert second.frame == 16
    assert second.delta_x == pytest.approx(17.5)
    assert second.delta_y == pytest.approx(8.5)

    assert second.distance > 0
    assert second.speed > 0


def test_object_moves_downward(detections):
    samples = calculate_motion(detections)

    first = samples[0]
    last = samples[5]

    vertical_displacement = last.center_y - first.center_y

    assert vertical_displacement > 70


def test_object_becomes_stable(detections):
    samples = calculate_motion(detections)

    intervals = detect_stable_intervals(
        samples,
        window_size=5,
        tolerance_px=3.0,
    )

    assert len(intervals) > 0

    assert any(
        interval.start_frame >= 20
        and interval.end_frame >= 30
        for interval in intervals
    )


def test_complete_trajectory_analysis(detections):
    result = analyze_trajectory(
        detections,
        fps=30,
        window_size=5,
        tolerance_px=3.0,
    )

    assert result.object_id == 1
    assert result.fps == 30
    assert len(result.samples) == 36
    assert len(result.stable_intervals) > 0

    assert result.summary.detection_count == 36
    assert result.summary.first_frame == 15
    assert result.summary.last_frame == 50

    assert result.summary.total_displacement_x == pytest.approx(34.5)
    assert result.summary.total_displacement_y == pytest.approx(75.5)

    assert result.summary.max_speed > 0
    assert result.summary.max_vertical_speed > 0

    assert result.summary.duration_seconds == pytest.approx(35 / 30)


def test_empty_trajectory_raises_error():
    with pytest.raises(ValueError):
        analyze_trajectory([])


def test_summarize_trajectory(detections):
    samples = calculate_motion(detections, fps=30)

    summary = summarize_trajectory(samples, fps=30)

    assert summary.detection_count == 36
    assert summary.first_frame == 15
    assert summary.last_frame == 50

    assert summary.total_displacement_x == pytest.approx(34.5)
    assert summary.total_displacement_y == pytest.approx(75.5)

    assert summary.max_speed > 0
    assert summary.max_vertical_speed > 0

    assert 15 <= summary.max_speed_frame <= 50
    assert 15 <= summary.max_vertical_speed_frame <= 50

    assert summary.duration_seconds == pytest.approx(35 / 30)


def test_summarize_empty_trajectory_raises_error():
    with pytest.raises(ValueError):
        summarize_trajectory((), fps=30)


def test_summarize_trajectory_sorts_samples(detections):
    samples = calculate_motion(detections, fps=30)

    reversed_samples = list(reversed(samples))

    summary = summarize_trajectory(reversed_samples, fps=30)

    assert summary.first_frame == 15
    assert summary.last_frame == 50
    assert summary.total_displacement_y == pytest.approx(75.5)
