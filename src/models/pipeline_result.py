
"""Models representing the output of a verification pipeline."""

from dataclasses import dataclass

from src.models.evidence import EntityObservation, SpatialRelationship
from src.models.event import TemporalEvent
from src.models.execution_timeline import ExecutionTimeline
from src.models.observation_context import ObservationContext
from src.models.region import RegionObservation
from src.models.verification_result import VerificationResult


@dataclass(frozen=True)
class PipelineResult:
    """Complete output produced by a verification pipeline execution."""

    entity_observations: tuple[EntityObservation, ...]
    spatial_relationships: tuple[SpatialRelationship, ...]
    region_observations: tuple[RegionObservation, ...]
    events: tuple[TemporalEvent, ...]
    verification: VerificationResult
    context: ObservationContext
    execution_timeline: ExecutionTimeline

    def __post_init__(self) -> None:
        """Validate the pipeline result."""

        collections = (
            ("entity_observations", self.entity_observations, EntityObservation),
            ("spatial_relationships", self.spatial_relationships, SpatialRelationship),
            ("region_observations", self.region_observations, RegionObservation),
            ("events", self.events, TemporalEvent),
        )

        for field_name, values, expected_type in collections:
            if not isinstance(values, tuple):
                raise ValueError(f"{field_name} must be a tuple.")

            if not all(
                isinstance(item, expected_type)
                for item in values
            ):
                raise ValueError(
                    f"Every item in {field_name} must be a "
                    f"{expected_type.__name__}."
                )

        if not isinstance(self.verification, VerificationResult):
            raise ValueError(
                "verification must be a VerificationResult."
            )

        if not isinstance(self.context, ObservationContext):
            raise ValueError(
                "context must be an ObservationContext."
            )

        if not isinstance(self.execution_timeline, ExecutionTimeline):
            raise ValueError(
                "execution_timeline must be an ExecutionTimeline."
            )

        if self.execution_timeline.events != self.events:
            raise ValueError(
                "execution_timeline events must match pipeline events."
            )

        if self.execution_timeline.context != self.context:
            raise ValueError(
                "execution_timeline context must match pipeline context."
            )