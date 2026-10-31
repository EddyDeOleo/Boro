"""Execution timeline model for robotic task analysis."""

from dataclasses import dataclass

from src.models.event import TemporalEvent
from src.models.observation_context import (
    CoverageScope,
    ObservationContext,
)


@dataclass(frozen=True)
class ExecutionTimeline:
    """Chronologically ordered events within an observation context."""

    events: tuple[TemporalEvent, ...]
    context: ObservationContext

    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple):
            raise TypeError("events must be a tuple.")

        if not all(
            isinstance(event, TemporalEvent)
            for event in self.events
        ):
            raise TypeError(
                "All events must be TemporalEvent instances."
            )

        if not isinstance(self.context, ObservationContext):
            raise TypeError(
                "context must be an ObservationContext instance."
            )

    @property
    def start_time_seconds(self) -> float:
        """Return the observation context start time."""

        return self.context.start_time_seconds

    @property
    def end_time_seconds(self) -> float:
        """Return the observation context end time."""

        return self.context.end_time_seconds

    @property
    def coverage_scope(self) -> CoverageScope:
        """Return the observation context coverage scope."""

        return self.context.coverage_scope