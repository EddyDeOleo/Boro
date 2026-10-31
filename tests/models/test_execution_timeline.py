"""Tests for the execution timeline model."""

from dataclasses import FrozenInstanceError

import pytest

from src.models.event import EventStatus, TemporalEvent
from src.models.execution_timeline import ExecutionTimeline
from src.models.observation_context import (
    CoverageScope,
    ObservationContext,
)


@pytest.fixture
def valid_context():
    """Provide a valid observation context."""

    return ObservationContext(
        fps=30.0,
        frame_count=100,
        frame_start=0,
        frame_end=99,
        time_offset_seconds=5.0,
        coverage_scope=CoverageScope.COMPLETE_TASK,
    )


@pytest.fixture
def valid_event():
    """Provide a valid temporal event."""

    return TemporalEvent(
        event_id="event_001",
        event_type="OBJECT_MOVING",
        status=EventStatus.NOT_DETECTED,
        frame_start=10,
        frame_end=20,
        timestamp_start=5.33,
        timestamp_end=5.67,
        entity_ids=("red_cube",),
    )


class TestExecutionTimeline:

    def test_create_execution_timeline(
        self,
        valid_event,
        valid_context,
    ):
        timeline = ExecutionTimeline(
            events=(valid_event,),
            context=valid_context,
        )

        assert timeline.events == (valid_event,)
        assert timeline.context is valid_context

    def test_create_empty_execution_timeline(self, valid_context):
        timeline = ExecutionTimeline(
            events=(),
            context=valid_context,
        )

        assert timeline.events == ()

    def test_start_time_seconds(self, valid_context):
        timeline = ExecutionTimeline(
            events=(),
            context=valid_context,
        )

        assert timeline.start_time_seconds == pytest.approx(5.0)

    def test_end_time_seconds(self, valid_context):
        timeline = ExecutionTimeline(
            events=(),
            context=valid_context,
        )

        assert timeline.end_time_seconds == pytest.approx(
            5.0 + 99 / 30.0
        )

    def test_coverage_scope(self, valid_context):
        timeline = ExecutionTimeline(
            events=(),
            context=valid_context,
        )

        assert timeline.coverage_scope == CoverageScope.COMPLETE_TASK

    def test_partial_coverage_scope(self):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=10,
            frame_end=50,
            coverage_scope=CoverageScope.PARTIAL_SEGMENT,
        )

        timeline = ExecutionTimeline(
            events=(),
            context=context,
        )

        assert timeline.coverage_scope == CoverageScope.PARTIAL_SEGMENT

    def test_events_must_be_tuple(
        self,
        valid_event,
        valid_context,
    ):
        with pytest.raises(
            TypeError,
            match="events must be a tuple",
        ):
            ExecutionTimeline(
                events=[valid_event],
                context=valid_context,
            )

    def test_all_events_must_be_temporal_events(self, valid_context):
        with pytest.raises(
            TypeError,
            match="TemporalEvent instances",
        ):
            ExecutionTimeline(
                events=("invalid_event",),
                context=valid_context,
            )

    def test_context_must_be_observation_context(self):
        with pytest.raises(
            TypeError,
            match="ObservationContext instance",
        ):
            ExecutionTimeline(
                events=(),
                context="invalid_context",
            )

    def test_timeline_is_immutable(
        self,
        valid_event,
        valid_context,
    ):
        timeline = ExecutionTimeline(
            events=(valid_event,),
            context=valid_context,
        )

        with pytest.raises(FrozenInstanceError):
            timeline.events = ()

    def test_timeline_preserves_original_event_instances(
        self,
        valid_event,
        valid_context,
    ):
        timeline = ExecutionTimeline(
            events=(valid_event,),
            context=valid_context,
        )

        assert timeline.events[0] is valid_event

    def test_context_properties_reflect_original_context(
        self,
        valid_context,
    ):
        timeline = ExecutionTimeline(
            events=(),
            context=valid_context,
        )

        assert timeline.start_time_seconds == (
            valid_context.start_time_seconds
        )

        assert timeline.end_time_seconds == (
            valid_context.end_time_seconds
        )

        assert timeline.coverage_scope == valid_context.coverage_scope