"""Tests for the execution timeline builder."""

import pytest

from src.models.event import EventStatus, TemporalEvent
from src.models.execution_timeline import ExecutionTimeline
from src.models.observation_context import (
    CoverageScope,
    ObservationContext,
)
from src.verification.execution_timeline_builder import (
    ExecutionTimelineBuilder,
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
        coverage_scope=CoverageScope.PARTIAL_SEGMENT,
    )


def create_event(
    event_id: str,
    timestamp_start: float,
    timestamp_end: float,
) -> TemporalEvent:
    """Create a valid non-detected event for testing."""

    return TemporalEvent(
        event_id=event_id,
        event_type="TEST_EVENT",
        status=EventStatus.NOT_DETECTED,
        frame_start=0,
        frame_end=10,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        entity_ids=("red_cube",),
    )


class TestExecutionTimelineBuilder:

    def test_build_execution_timeline(self, valid_context):
        event = create_event("event_001", 5.0, 5.5)

        builder = ExecutionTimelineBuilder()

        timeline = builder.build(
            events=(event,),
            context=valid_context,
        )

        assert isinstance(timeline, ExecutionTimeline)
        assert timeline.events == (event,)
        assert timeline.context is valid_context

    def test_build_empty_timeline(self, valid_context):
        builder = ExecutionTimelineBuilder()

        timeline = builder.build(
            events=(),
            context=valid_context,
        )

        assert timeline.events == ()

    def test_events_are_sorted_by_start_timestamp(
        self,
        valid_context,
    ):
        first = create_event("event_001", 5.0, 5.5)
        second = create_event("event_002", 6.0, 6.5)
        third = create_event("event_003", 5.5, 5.8)

        builder = ExecutionTimelineBuilder()

        timeline = builder.build(
            events=(second, first, third),
            context=valid_context,
        )

        assert tuple(
            event.event_id for event in timeline.events
        ) == (
            "event_001",
            "event_003",
            "event_002",
        )

    def test_events_with_same_start_are_sorted_by_end_timestamp(
        self,
        valid_context,
    ):
        first = create_event("event_001", 5.0, 6.0)
        second = create_event("event_002", 5.0, 5.5)
        third = create_event("event_003", 5.0, 5.8)

        builder = ExecutionTimelineBuilder()

        timeline = builder.build(
            events=(first, second, third),
            context=valid_context,
        )

        assert tuple(
            event.event_id for event in timeline.events
        ) == (
            "event_002",
            "event_003",
            "event_001",
        )

    def test_events_with_same_timestamps_are_sorted_by_event_id(
        self,
        valid_context,
    ):
        first = create_event("event_003", 5.0, 5.5)
        second = create_event("event_001", 5.0, 5.5)
        third = create_event("event_002", 5.0, 5.5)

        builder = ExecutionTimelineBuilder()

        timeline = builder.build(
            events=(first, second, third),
            context=valid_context,
        )

        assert tuple(
            event.event_id for event in timeline.events
        ) == (
            "event_001",
            "event_002",
            "event_003",
        )

    def test_overlapping_events_are_preserved(self, valid_context):
        first = create_event("event_001", 5.0, 6.0)
        second = create_event("event_002", 5.5, 6.5)

        builder = ExecutionTimelineBuilder()

        timeline = builder.build(
            events=(second, first),
            context=valid_context,
        )

        assert len(timeline.events) == 2
        assert timeline.events == (first, second)

        assert (
            timeline.events[0].timestamp_end
            > timeline.events[1].timestamp_start
        )

    def test_original_event_instances_are_preserved(
        self,
        valid_context,
    ):
        first = create_event("event_001", 5.0, 5.5)
        second = create_event("event_002", 6.0, 6.5)

        builder = ExecutionTimelineBuilder()

        timeline = builder.build(
            events=(second, first),
            context=valid_context,
        )

        assert timeline.events[0] is first
        assert timeline.events[1] is second

    def test_generator_input_is_supported(self, valid_context):
        first = create_event("event_001", 5.0, 5.5)
        second = create_event("event_002", 6.0, 6.5)

        def event_generator():
            yield second
            yield first

        builder = ExecutionTimelineBuilder()

        timeline = builder.build(
            events=event_generator(),
            context=valid_context,
        )

        assert timeline.events == (first, second)

    def test_input_collection_is_not_modified(self, valid_context):
        first = create_event("event_001", 5.0, 5.5)
        second = create_event("event_002", 6.0, 6.5)

        original_events = [second, first]

        builder = ExecutionTimelineBuilder()

        builder.build(
            events=original_events,
            context=valid_context,
        )

        assert original_events == [second, first]

    @pytest.mark.parametrize(
        "events",
        [
            None,
            "invalid",
            b"invalid",
            123,
        ],
    )
    def test_invalid_events_input_raises_type_error(
        self,
        events,
        valid_context,
    ):
        builder = ExecutionTimelineBuilder()

        with pytest.raises(TypeError, match="iterable"):
            builder.build(
                events=events,
                context=valid_context,
            )

    def test_invalid_event_item_raises_type_error(self, valid_context):
        builder = ExecutionTimelineBuilder()

        with pytest.raises(
            TypeError,
            match="TemporalEvent instances",
        ):
            builder.build(
                events=("invalid_event",),
                context=valid_context,
            )

    def test_invalid_context_raises_type_error(self):
        builder = ExecutionTimelineBuilder()

        with pytest.raises(
            TypeError,
            match="ObservationContext instance",
        ):
            builder.build(
                events=(),
                context="invalid_context",
            )

    def test_partial_coverage_is_preserved(self, valid_context):
        builder = ExecutionTimelineBuilder()

        timeline = builder.build(
            events=(),
            context=valid_context,
        )

        assert timeline.coverage_scope == CoverageScope.PARTIAL_SEGMENT