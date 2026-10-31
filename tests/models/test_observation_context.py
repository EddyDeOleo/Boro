
"""Tests for video observation context models."""

from dataclasses import FrozenInstanceError

import pytest

from src.models.observation_context import (
    CoverageScope,
    ObservationContext,
)


class TestObservationContext:

    def test_create_observation_context(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=0,
            frame_end=99,
            time_offset_seconds=0.0,
            coverage_scope=CoverageScope.COMPLETE_TASK,
        )

        assert context.fps == 30.0
        assert context.frame_count == 100
        assert context.frame_start == 0
        assert context.frame_end == 99
        assert context.time_offset_seconds == 0.0
        assert context.coverage_scope == CoverageScope.COMPLETE_TASK

    def test_default_values(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=0,
            frame_end=99,
        )

        assert context.time_offset_seconds == 0.0
        assert context.coverage_scope == CoverageScope.UNKNOWN

    def test_create_partial_segment_context(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=300,
            frame_start=50,
            frame_end=100,
            coverage_scope=CoverageScope.PARTIAL_SEGMENT,
        )

        assert context.coverage_scope == CoverageScope.PARTIAL_SEGMENT
        assert context.frame_start == 50
        assert context.frame_end == 100

    def test_create_unknown_coverage_context(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=0,
            frame_end=99,
            coverage_scope=CoverageScope.UNKNOWN,
        )

        assert context.coverage_scope == CoverageScope.UNKNOWN

    def test_observed_frame_count(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=10,
            frame_end=19,
        )

        assert context.observed_frame_count == 10

    def test_duration_seconds(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=0,
            frame_end=30,
        )

        assert context.duration_seconds == pytest.approx(1.0)

    def test_duration_seconds_for_single_frame(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=10,
            frame_end=10,
        )

        assert context.duration_seconds == 0.0

    def test_start_time_seconds_without_offset(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=30,
            frame_end=60,
        )

        assert context.start_time_seconds == pytest.approx(1.0)

    def test_end_time_seconds_without_offset(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=30,
            frame_end=60,
        )

        assert context.end_time_seconds == pytest.approx(2.0)

    def test_timestamps_include_time_offset(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=30,
            frame_end=60,
            time_offset_seconds=5.0,
        )

        assert context.start_time_seconds == pytest.approx(6.0)
        assert context.end_time_seconds == pytest.approx(7.0)

    def test_covers_entire_video(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=0,
            frame_end=99,
        )

        assert context.covers_entire_video is True

    def test_does_not_cover_entire_video_when_start_is_not_zero(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=10,
            frame_end=99,
        )

        assert context.covers_entire_video is False

    def test_does_not_cover_entire_video_when_end_is_incomplete(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=0,
            frame_end=98,
        )

        assert context.covers_entire_video is False

    def test_full_video_does_not_imply_complete_task(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=0,
            frame_end=99,
            coverage_scope=CoverageScope.UNKNOWN,
        )

        assert context.covers_entire_video is True
        assert context.coverage_scope == CoverageScope.UNKNOWN

    @pytest.mark.parametrize(
        "fps",
        [0, -1, -30.0, float("inf"), float("-inf"), float("nan"), True, "30"],
    )
    def test_invalid_fps_raises_value_error(self, fps):
        with pytest.raises(ValueError, match="fps"):
            ObservationContext(
                fps=fps,
                frame_count=100,
                frame_start=0,
                frame_end=99,
            )

    @pytest.mark.parametrize(
        "frame_count",
        [0, -1, 1.5, True, "100", None],
    )
    def test_invalid_frame_count_raises_value_error(self, frame_count):
        with pytest.raises(ValueError, match="frame_count"):
            ObservationContext(
                fps=30.0,
                frame_count=frame_count,
                frame_start=0,
                frame_end=0,
            )

    @pytest.mark.parametrize(
        "frame_index",
        [-1, 1.5, True, "10", None],
    )
    def test_invalid_frame_start_raises_value_error(self, frame_index):
        with pytest.raises(ValueError, match="frame_start"):
            ObservationContext(
                fps=30.0,
                frame_count=100,
                frame_start=frame_index,
                frame_end=20,
            )

    @pytest.mark.parametrize(
        "frame_index",
        [-1, 1.5, True, "20", None],
    )
    def test_invalid_frame_end_raises_value_error(self, frame_index):
        with pytest.raises(ValueError, match="frame_end"):
            ObservationContext(
                fps=30.0,
                frame_count=100,
                frame_start=0,
                frame_end=frame_index,
            )

    def test_frame_start_cannot_exceed_frame_end(self):
        with pytest.raises(
            ValueError,
            match="frame_start cannot exceed frame_end",
        ):
            ObservationContext(
                fps=30.0,
                frame_count=100,
                frame_start=50,
                frame_end=20,
            )

    def test_frame_end_must_be_smaller_than_frame_count(self):
        with pytest.raises(
            ValueError,
            match="frame_end must be smaller than frame_count",
        ):
            ObservationContext(
                fps=30.0,
                frame_count=100,
                frame_start=0,
                frame_end=100,
            )

    @pytest.mark.parametrize(
        "offset",
        [-1.0, float("inf"), float("-inf"), float("nan"), True, "5.0", None],
    )
    def test_invalid_time_offset_raises_value_error(self, offset):
        with pytest.raises(ValueError, match="time_offset_seconds"):
            ObservationContext(
                fps=30.0,
                frame_count=100,
                frame_start=0,
                frame_end=99,
                time_offset_seconds=offset,
            )

    @pytest.mark.parametrize(
        "scope",
        ["COMPLETE_TASK", "PARTIAL_SEGMENT", "UNKNOWN", None, 1],
    )
    def test_invalid_coverage_scope_raises_value_error(self, scope):
        with pytest.raises(
            ValueError,
            match="coverage_scope must be a CoverageScope value",
        ):
            ObservationContext(
                fps=30.0,
                frame_count=100,
                frame_start=0,
                frame_end=99,
                coverage_scope=scope,
            )

    def test_context_is_immutable(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=0,
            frame_end=99,
        )

        with pytest.raises(FrozenInstanceError):
            context.fps = 60.0

    def test_coverage_scope_is_string_enum(self):
        assert CoverageScope.COMPLETE_TASK == "COMPLETE_TASK"
        assert CoverageScope.PARTIAL_SEGMENT == "PARTIAL_SEGMENT"
        assert CoverageScope.UNKNOWN == "UNKNOWN"
