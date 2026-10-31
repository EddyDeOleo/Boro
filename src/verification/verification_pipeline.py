"""Orchestrate perception adaptation, event detection, and verification."""

from collections.abc import Mapping, Sequence

import numpy as np

from src.adapters.sam_observation_adapter import adapt_sam_observations
from src.models.observation_context import ObservationContext
from src.models.pipeline_result import PipelineResult
from src.models.region import RegionDefinition
from src.models.task_roles import TaskRoles
from src.models.verification_criteria import VerificationCriteria
from src.perception.mask_analyzer import (
    MaskMetrics,
    MaskRelationshipMetrics,
    RelativeMetrics,
)
from src.verification.event_detector import EventDetector
from src.verification.execution_timeline_builder import (
    ExecutionTimelineBuilder,
)
from src.verification.verdict_engine import VerdictEngine


class VerificationPipeline:
    """Coordinate perception adaptation, event detection, and verification."""

    def __init__(
        self,
        event_detector: EventDetector | None = None,
        verdict_engine: VerdictEngine | None = None,
    ) -> None:
        """Initialize the pipeline components."""

        if event_detector is not None and not isinstance(
            event_detector,
            EventDetector,
        ):
            raise TypeError(
                "event_detector must be an EventDetector instance."
            )

        if verdict_engine is not None and not isinstance(
            verdict_engine,
            VerdictEngine,
        ):
            raise TypeError(
                "verdict_engine must be a VerdictEngine instance."
            )

        self.event_detector = event_detector or EventDetector()
        self.verdict_engine = verdict_engine or VerdictEngine()
        self.timeline_builder = ExecutionTimelineBuilder()

    def run(
        self,
        object_metrics: Sequence[MaskMetrics],
        relative_metrics: Sequence[RelativeMetrics],
        relationship_metrics: Sequence[MaskRelationshipMetrics],
        masks_by_frame: Mapping[int, Mapping[str, np.ndarray]],
        object_id_map: Mapping[str, str],
        confidences: Mapping[tuple[int, str], float | None],
        relative_entity_ids: tuple[str, str],
        task_roles: TaskRoles,
        criteria: Sequence[VerificationCriteria],
        context: ObservationContext,
        region: RegionDefinition | None = None,
        region_sam_object_id: str | None = None,
        source: str = "sam-3.1",
    ) -> PipelineResult:
        """Execute the complete verification workflow.

        FPS and temporal offset are taken from ObservationContext
        to maintain a single source of truth for video metadata.
        """

        self._validate_inputs(
            task_roles=task_roles,
            criteria=criteria,
            context=context,
        )

        observations = adapt_sam_observations(
            object_metrics=object_metrics,
            relative_metrics=relative_metrics,
            relationship_metrics=relationship_metrics,
            masks_by_frame=masks_by_frame,
            object_id_map=object_id_map,
            confidences=confidences,
            fps=context.fps,
            relative_entity_ids=relative_entity_ids,
            region=region,
            region_sam_object_id=region_sam_object_id,
            time_offset_seconds=context.time_offset_seconds,
            source=source,
        )

        events = self.event_detector.detect(
            entity_observations=observations.entity_observations,
            spatial_relationships=observations.spatial_relationships,
            region_observations=observations.region_observations,
            task_roles=task_roles,
            context=context,
        )

        verification = self.verdict_engine.evaluate(
            criteria=criteria,
            events=events,
            task_roles=task_roles,
            context=context,
        )

        execution_timeline = self.timeline_builder.build(
            events=events,
            context=context,
        )

        return PipelineResult(
            entity_observations=observations.entity_observations,
            spatial_relationships=observations.spatial_relationships,
            region_observations=observations.region_observations,
            events=events,
            verification=verification,
            context=context,
            execution_timeline=execution_timeline,
        )

    @staticmethod
    def _validate_inputs(
        task_roles: TaskRoles,
        criteria: Sequence[VerificationCriteria],
        context: ObservationContext,
    ) -> None:
        """Validate pipeline-level inputs."""

        if not isinstance(task_roles, TaskRoles):
            raise TypeError(
                "task_roles must be a TaskRoles instance."
            )

        if not isinstance(context, ObservationContext):
            raise TypeError(
                "context must be an ObservationContext instance."
            )

        if isinstance(criteria, (str, bytes)) or not isinstance(
            criteria,
            Sequence,
        ):
            raise TypeError(
                "criteria must be a sequence of VerificationCriteria."
            )

        if not criteria:
            raise ValueError("criteria must not be empty.")

        if not all(
            isinstance(criterion, VerificationCriteria)
            for criterion in criteria
        ):
            raise TypeError(
                "Every item in criteria must be a VerificationCriteria."
            )