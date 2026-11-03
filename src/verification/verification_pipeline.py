
"""Orchestrate perception, event detection, and verification."""

from collections.abc import Sequence

from src.models.perception_result import PerceptionResult
from src.models.pipeline_result import PipelineResult
from src.models.task_roles import TaskRoles
from src.models.verification_criteria import VerificationCriteria
from src.verification.event_detector import EventDetector
from src.verification.execution_timeline_builder import (
    ExecutionTimelineBuilder,
)
from src.verification.verdict_engine import VerdictEngine


class VerificationPipeline:
    """Coordinate perception, event detection, and verification."""

    def __init__(
        self,
        event_detector: EventDetector | None = None,
        verdict_engine: VerdictEngine | None = None,
        timeline_builder: ExecutionTimelineBuilder | None = None,
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

        if timeline_builder is not None and not isinstance(
            timeline_builder,
            ExecutionTimelineBuilder,
        ):
            raise TypeError(
                "timeline_builder must be an ExecutionTimelineBuilder instance."
            )

        self.event_detector = (
            event_detector
            if event_detector is not None
            else EventDetector()
        )

        self.verdict_engine = (
            verdict_engine
            if verdict_engine is not None
            else VerdictEngine()
        )

        self.timeline_builder = (
            timeline_builder
            if timeline_builder is not None
            else ExecutionTimelineBuilder()
        )

    def run(
        self,
        perception_result: PerceptionResult,
        task_roles: TaskRoles,
        criteria: Sequence[VerificationCriteria],
    ) -> PipelineResult:
        """Execute the complete verification workflow.

        All perception data is obtained from the provider-independent
        PerceptionResult contract.
        """

        self._validate_inputs(
            perception_result=perception_result,
            task_roles=task_roles,
            criteria=criteria,
        )

        context = perception_result.context

        detected_events = self.event_detector.detect(
            entity_observations=perception_result.observations,
            spatial_relationships=perception_result.spatial_relationships,
            region_observations=perception_result.region_observations,
            task_roles=task_roles,
            context=context,
        )

        execution_timeline = self.timeline_builder.build(
            events=detected_events,
            context=context,
        )

        events = execution_timeline.events

        verification = self.verdict_engine.evaluate(
            criteria=criteria,
            events=events,
            task_roles=task_roles,
            context=context,
        )

        return PipelineResult(
            entity_observations=perception_result.observations,
            spatial_relationships=perception_result.spatial_relationships,
            region_observations=perception_result.region_observations,
            events=events,
            verification=verification,
            context=context,
            execution_timeline=execution_timeline,
        )

    @staticmethod
    def _validate_inputs(
        perception_result: PerceptionResult,
        task_roles: TaskRoles,
        criteria: Sequence[VerificationCriteria],
    ) -> None:
        """Validate pipeline-level inputs."""

        if not isinstance(perception_result, PerceptionResult):
            raise TypeError(
                "perception_result must be a PerceptionResult instance."
            )

        if not isinstance(task_roles, TaskRoles):
            raise TypeError(
                "task_roles must be a TaskRoles instance."
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
