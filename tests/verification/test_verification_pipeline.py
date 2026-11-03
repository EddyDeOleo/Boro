
"""Tests for the verification pipeline orchestration."""

from unittest.mock import patch

import pytest

from src.models.event import EventStatus, TemporalEvent
from src.models.observation_context import (
    CoverageScope,
    ObservationContext,
)
from src.models.perception_result import PerceptionResult
from src.models.pipeline_result import PipelineResult
from src.models.task_roles import TaskRoles
from src.models.verification_criteria import (
    CriterionRequirement,
    VerificationCriteria,
)
from src.models.video_metadata import VideoMetadata
from src.verification.event_detector import EventDetector
from src.verification.execution_timeline_builder import (
    ExecutionTimelineBuilder,
)
from src.verification.verdict_engine import VerdictEngine
from src.verification.verification_pipeline import VerificationPipeline


@pytest.fixture
def context():
    return ObservationContext(
        fps=30.0,
        frame_count=10,
        frame_start=0,
        frame_end=9,
        time_offset_seconds=0.0,
        coverage_scope=CoverageScope.COMPLETE_TASK,
    )


@pytest.fixture
def video_metadata():
    return VideoMetadata(
        width=1280,
        height=960,
        fps=30.0,
        frame_count=10,
    )


@pytest.fixture
def perception_result(context, video_metadata):
    return PerceptionResult(
        context=context,
        video_metadata=video_metadata,
        observations=(),
        spatial_relationships=(),
        region_observations=(),
        provider="meta",
        model="sam-3.1",
    )


@pytest.fixture
def task_roles():
    return TaskRoles(
        entity_roles={"object": "red_cube"},
        region_roles={"target": "target_zone_01"},
    )


@pytest.fixture
def criteria():
    return (
        VerificationCriteria(
            criterion_id="object_movement",
            description="The object must move.",
            event_type="OBJECT_MOVING",
            requirement=CriterionRequirement.REQUIRED,
            entity_roles=("object",),
        ),
    )


@pytest.fixture
def pipeline():
    return VerificationPipeline()


def test_pipeline_orchestrates_all_components(
    pipeline,
    perception_result,
    task_roles,
    criteria,
):
    """Perception observations must reach event detection and verification."""

    with patch.object(
        pipeline.event_detector,
        "detect",
        return_value=(),
    ) as detector:
        with patch.object(
            pipeline.verdict_engine,
            "evaluate",
            wraps=pipeline.verdict_engine.evaluate,
        ) as verdict_engine:
            result = pipeline.run(
                perception_result=perception_result,
                task_roles=task_roles,
                criteria=criteria,
            )

    detector.assert_called_once_with(
        entity_observations=perception_result.observations,
        spatial_relationships=perception_result.spatial_relationships,
        region_observations=perception_result.region_observations,
        task_roles=task_roles,
        context=perception_result.context,
    )

    verdict_engine.assert_called_once()

    assert isinstance(result, PipelineResult)
    assert result.entity_observations == ()
    assert result.spatial_relationships == ()
    assert result.region_observations == ()
    assert result.events == ()
    assert result.context is perception_result.context
    assert result.verification is not None


def test_pipeline_uses_context_from_perception_result(
    pipeline,
    task_roles,
    criteria,
    video_metadata,
):
    """The pipeline must use the context supplied by PerceptionResult."""

    context = ObservationContext(
        fps=25.0,
        frame_count=20,
        frame_start=0,
        frame_end=19,
        time_offset_seconds=3.5,
        coverage_scope=CoverageScope.COMPLETE_TASK,
    )

    perception = PerceptionResult(
        context=context,
        video_metadata=video_metadata,
        observations=(),
        provider="meta",
        model="sam-3.1",
    )

    with patch.object(
        pipeline.event_detector,
        "detect",
        return_value=(),
    ) as detector:
        result = pipeline.run(
            perception_result=perception,
            task_roles=task_roles,
            criteria=criteria,
        )

    detector_kwargs = detector.call_args.kwargs

    assert detector_kwargs["context"] is context
    assert result.context is context


def test_pipeline_preserves_perception_observations(
    pipeline,
    perception_result,
    task_roles,
    criteria,
):
    """The pipeline must preserve the normalized perception output."""

    with patch.object(
        pipeline.event_detector,
        "detect",
        return_value=(),
    ):
        result = pipeline.run(
            perception_result=perception_result,
            task_roles=task_roles,
            criteria=criteria,
        )

    assert result.entity_observations == perception_result.observations
    assert result.spatial_relationships == (
        perception_result.spatial_relationships
    )
    assert result.region_observations == (
        perception_result.region_observations
    )


def test_partial_video_coverage_cannot_produce_verified(
    task_roles,
    criteria,
    video_metadata,
):
    """A partial segment must not be reported as a verified complete task."""

    context = ObservationContext(
        fps=30.0,
        frame_count=100,
        frame_start=20,
        frame_end=79,
        time_offset_seconds=0.0,
        coverage_scope=CoverageScope.PARTIAL_SEGMENT,
    )

    perception = PerceptionResult(
        context=context,
        video_metadata=video_metadata,
        observations=(),
        provider="meta",
        model="sam-3.1",
    )

    pipeline = VerificationPipeline()

    with patch.object(
        pipeline.event_detector,
        "detect",
        return_value=(),
    ):
        result = pipeline.run(
            perception_result=perception,
            task_roles=task_roles,
            criteria=criteria,
        )

    assert result.verification.verdict != "VERIFIED"
    assert (
        result.verification.coverage_scope
        == CoverageScope.PARTIAL_SEGMENT
    )
    assert result.verification.covers_entire_video is False


def test_pipeline_rejects_invalid_perception_result(
    pipeline,
    task_roles,
    criteria,
):
    """The pipeline must require a valid PerceptionResult."""

    with pytest.raises(
        TypeError,
        match="perception_result",
    ):
        pipeline.run(
            perception_result=None,
            task_roles=task_roles,
            criteria=criteria,
        )


def test_pipeline_rejects_invalid_task_roles(
    pipeline,
    perception_result,
    criteria,
):
    """Invalid task roles must be rejected."""

    with pytest.raises(TypeError, match="task_roles"):
        pipeline.run(
            perception_result=perception_result,
            task_roles=None,
            criteria=criteria,
        )


def test_pipeline_rejects_empty_criteria(
    pipeline,
    perception_result,
    task_roles,
):
    """A verification run must contain at least one criterion."""

    with pytest.raises(
        ValueError,
        match="criteria must not be empty",
    ):
        pipeline.run(
            perception_result=perception_result,
            task_roles=task_roles,
            criteria=(),
        )


def test_pipeline_rejects_invalid_criteria_items(
    pipeline,
    perception_result,
    task_roles,
):
    """Every criterion must be a VerificationCriteria instance."""

    with pytest.raises(
        TypeError,
        match="VerificationCriteria",
    ):
        pipeline.run(
            perception_result=perception_result,
            task_roles=task_roles,
            criteria=("invalid",),
        )


def test_pipeline_rejects_non_sequence_criteria(
    pipeline,
    perception_result,
    task_roles,
):
    """Criteria must be provided as a sequence."""

    with pytest.raises(
        TypeError,
        match="criteria",
    ):
        pipeline.run(
            perception_result=perception_result,
            task_roles=task_roles,
            criteria="invalid",
        )


def test_pipeline_accepts_injected_components():
    """Custom pipeline components must be preserved."""

    detector = EventDetector()
    verdict_engine = VerdictEngine()
    timeline_builder = ExecutionTimelineBuilder()

    pipeline = VerificationPipeline(
        event_detector=detector,
        verdict_engine=verdict_engine,
        timeline_builder=timeline_builder,
    )

    assert pipeline.event_detector is detector
    assert pipeline.verdict_engine is verdict_engine
    assert pipeline.timeline_builder is timeline_builder


def test_pipeline_rejects_invalid_injected_detector():
    """The detector dependency must have the expected type."""

    with pytest.raises(
        TypeError,
        match="event_detector",
    ):
        VerificationPipeline(event_detector=object())


def test_pipeline_rejects_invalid_injected_verdict_engine():
    """The verdict engine dependency must have the expected type."""

    with pytest.raises(
        TypeError,
        match="verdict_engine",
    ):
        VerificationPipeline(verdict_engine=object())


def test_pipeline_rejects_invalid_injected_timeline_builder():
    """The timeline builder dependency must have the expected type."""

    with pytest.raises(
        TypeError,
        match="timeline_builder",
    ):
        VerificationPipeline(timeline_builder=object())


def test_pipeline_includes_detected_events_in_execution_timeline(
    pipeline,
    perception_result,
    task_roles,
    criteria,
):
    """The execution timeline must preserve detected events."""

    event = TemporalEvent(
        event_id="event_001",
        event_type="OBJECT_MOVING",
        status=EventStatus.NOT_DETECTED,
        frame_start=1,
        frame_end=5,
        timestamp_start=1 / 30,
        timestamp_end=5 / 30,
        entity_ids=("red_cube",),
    )

    with patch.object(
        pipeline.event_detector,
        "detect",
        return_value=(event,),
    ):
        result = pipeline.run(
            perception_result=perception_result,
            task_roles=task_roles,
            criteria=criteria,
        )

    assert result.events == (event,)
    assert result.execution_timeline.events == result.events
    assert result.execution_timeline.events[0] is event
    assert result.execution_timeline.context is perception_result.context
    assert result.context is perception_result.context

def test_pipeline_normalizes_event_order(
    pipeline,
    perception_result,
    task_roles,
    criteria,
):
    """The pipeline must use chronological event order consistently."""

    later_event = TemporalEvent(
        event_id="event_002",
        event_type="OBJECT_MOVING",
        status=EventStatus.NOT_DETECTED,
        frame_start=5,
        frame_end=8,
        timestamp_start=5 / 30,
        timestamp_end=8 / 30,
        entity_ids=("red_cube",),
    )

    earlier_event = TemporalEvent(
        event_id="event_001",
        event_type="OBJECT_MOVING",
        status=EventStatus.NOT_DETECTED,
        frame_start=1,
        frame_end=3,
        timestamp_start=1 / 30,
        timestamp_end=3 / 30,
        entity_ids=("red_cube",),
    )

    with patch.object(
        pipeline.event_detector,
        "detect",
        return_value=(later_event, earlier_event),
    ):
        result = pipeline.run(
            perception_result=perception_result,
            task_roles=task_roles,
            criteria=criteria,
        )

    assert result.events == (earlier_event, later_event)
    assert result.execution_timeline.events == (
        earlier_event,
        later_event,
    )
