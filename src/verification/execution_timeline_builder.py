"""Build chronological execution timelines from detected events."""

from collections.abc import Iterable

from src.models.event import TemporalEvent
from src.models.execution_timeline import ExecutionTimeline
from src.models.observation_context import ObservationContext


class ExecutionTimelineBuilder:
    """Build immutable execution timelines."""

    def build(
        self,
        events: Iterable[TemporalEvent],
        context: ObservationContext,
    ) -> ExecutionTimeline:
        """Build a timeline preserving and chronologically ordering events."""

        if isinstance(events, (str, bytes)):
            raise TypeError(
                "events must be an iterable of TemporalEvent instances."
            )

        if not isinstance(events, Iterable):
            raise TypeError(
                "events must be an iterable of TemporalEvent instances."
            )

        if not isinstance(context, ObservationContext):
            raise TypeError(
                "context must be an ObservationContext instance."
            )

        event_list = tuple(events)

        if not all(
            isinstance(event, TemporalEvent)
            for event in event_list
        ):
            raise TypeError(
                "All events must be TemporalEvent instances."
            )

        ordered_events = tuple(
            sorted(
                event_list,
                key=lambda event: (
                    event.timestamp_start,
                    event.timestamp_end,
                    event.event_id,
                ),
            )
        )

        return ExecutionTimeline(
            events=ordered_events,
            context=context,
        )