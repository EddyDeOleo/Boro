
"""Tests for the verification pipeline orchestration."""

from src.models.event import EventStatus, TemporalEvent
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.models.observation_context import (
    CoverageScope,
    ObservationContext,
)
from src.models.task_roles import TaskRoles
from src.models.verification_criteria import (
    CriterionRequirement,
    VerificationCriteria,
)
from src.verification.event_detector import EventDetector
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
def adapter_bundle():
    return SimpleNamespace(
        entity_observations=(),
        spatial_relationships=(),
        region_observations=(),
    )


@pytest.fixture
def pipeline():
    return VerificationPipeline()


def make_pipeline_inputs(context, task_roles, criteria):
    """Build the raw SAM-compatible inputs required by the pipeline."""

    return {
        "object_metrics": (),
        "relative_metrics": (),
        "relationship_metrics": (),
        "masks_by_frame": {},
        "object_id_map": {
            "red_cube": "1",
            "gripper": "2",
        },
        "confidences": {},
        "relative_entity_ids": ("red_cube", "gripper"),
        "task_roles": task_roles,
        "criteria": criteria,
        "context": context,
    }


def test_pipeline_orchestrates_all_components(
    pipeline,
    context,
    task_roles,
    criteria,
    adapter_bundle,
):
    """The adapter output must be passed to event detection and verification."""

    inputs = make_pipeline_inputs(context, task_roles, criteria)

    with patch(
        "src.verification.verification_pipeline.adapt_sam_observations",
        return_value=adapter_bundle,
    ) as adapter:
        result = pipeline.run(**inputs)

    adapter.assert_called_once()

    assert result.entity_observations == ()
    assert result.spatial_relationships == ()
    assert result.region_observations == ()
    assert result.events == ()
    assert result.context is context
    assert result.verification is not None


def test_pipeline_passes_context_metadata_to_adapter(
    pipeline,
    task_roles,
    criteria,
    adapter_bundle,
):
    """FPS and temporal offset must come from ObservationContext."""

    context = ObservationContext(
        fps=25.0,
        frame_count=20,
        frame_start=0,
        frame_end=19,
        time_offset_seconds=3.5,
        coverage_scope=CoverageScope.COMPLETE_TASK,
    )

    inputs = make_pipeline_inputs(context, task_roles, criteria)

    with patch(
        "src.verification.verification_pipeline.adapt_sam_observations",
        return_value=adapter_bundle,
    ) as adapter:
        pipeline.run(**inputs)

    adapter_kwargs = adapter.call_args.kwargs

    assert adapter_kwargs["fps"] == 25.0
    assert adapter_kwargs["time_offset_seconds"] == 3.5


def test_pipeline_forwards_region_configuration(
    pipeline,
    context,
    task_roles,
    criteria,
    adapter_bundle,
):
    """An optional target region must reach the SAM observation adapter."""

    from src.models.region import RegionDefinition, RegionType

    region = RegionDefinition(
        region_id="target_zone_01",
        name="Target Zone",
        region_type=RegionType.TARGET_ZONE,
        polygon=(
            (10, 10),
            (100, 10),
            (100, 100),
            (10, 100),
        ),
        frame_width=1280,
        frame_height=960,
        reference_frame=0,
    )

    inputs = make_pipeline_inputs(context, task_roles, criteria)
    inputs["region"] = region
    inputs["region_sam_object_id"] = "3"

    with patch(
        "src.verification.verification_pipeline.adapt_sam_observations",
        return_value=adapter_bundle,
    ) as adapter:
        pipeline.run(**inputs)

    adapter_kwargs = adapter.call_args.kwargs

    assert adapter_kwargs["region"] is region
    assert adapter_kwargs["region_sam_object_id"] == "3"


def test_partial_video_coverage_cannot_produce_verified(
    task_roles,
    criteria,
    adapter_bundle,
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

    pipeline = VerificationPipeline()
    inputs = make_pipeline_inputs(context, task_roles, criteria)

    with patch(
        "src.verification.verification_pipeline.adapt_sam_observations",
        return_value=adapter_bundle,
    ):
        result = pipeline.run(**inputs)

    assert result.verification.verdict != "VERIFIED"
    assert result.verification.coverage_scope == CoverageScope.PARTIAL_SEGMENT
    assert result.verification.covers_entire_video is False


def test_pipeline_rejects_invalid_task_roles(pipeline, context, criteria):
    """Invalid task roles must be rejected before perception adaptation."""

    inputs = make_pipeline_inputs(context, None, criteria)

    with pytest.raises(TypeError, match="task_roles"):
        pipeline.run(**inputs)


def test_pipeline_rejects_empty_criteria(pipeline, context, task_roles):
    """A verification run must contain at least one criterion."""

    inputs = make_pipeline_inputs(context, task_roles, ())

    with pytest.raises(ValueError, match="criteria must not be empty"):
        pipeline.run(**inputs)


def test_pipeline_rejects_invalid_criteria_items(
    pipeline,
    context,
    task_roles,
):
    """Every criterion must be a VerificationCriteria instance."""

    inputs = make_pipeline_inputs(context, task_roles, ("invalid",))

    with pytest.raises(TypeError, match="VerificationCriteria"):
        pipeline.run(**inputs)


def test_pipeline_rejects_invalid_context(pipeline, task_roles, criteria):
    """The pipeline must require a valid ObservationContext."""

    inputs = make_pipeline_inputs(None, task_roles, criteria)

    with pytest.raises(TypeError, match="context"):
        pipeline.run(**inputs)


def test_pipeline_accepts_injected_components():
    """Custom detector and verdict engine instances must be preserved."""

    detector = EventDetector()
    verdict_engine = VerdictEngine()

    pipeline = VerificationPipeline(
        event_detector=detector,
        verdict_engine=verdict_engine,
    )

    assert pipeline.event_detector is detector
    assert pipeline.verdict_engine is verdict_engine


def test_pipeline_rejects_invalid_injected_detector():
    """The detector dependency must have the expected type."""

    with pytest.raises(TypeError, match="event_detector"):
        VerificationPipeline(event_detector=object())


def test_pipeline_rejects_invalid_injected_verdict_engine():
    """The verdict engine dependency must have the expected type."""

    with pytest.raises(TypeError, match="verdict_engine"):
        VerificationPipeline(verdict_engine=object())

def test_pipeline_includes_detected_events_in_execution_timeline(
    pipeline,
    context,
    task_roles,
    criteria,
    adapter_bundle,
):
    """The execution timeline must preserve the events produced by detection."""

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

    inputs = make_pipeline_inputs(context, task_roles, criteria)

    with patch(
        "src.verification.verification_pipeline.adapt_sam_observations",
        return_value=adapter_bundle,
    ):
        with patch.object(
            pipeline.event_detector,
            "detect",
            return_value=(event,),
        ):
            result = pipeline.run(**inputs)

    assert result.events == (event,)
    assert result.execution_timeline.events == result.events

    assert result.execution_timeline.events[0] is event
    assert result.execution_timeline.context is context
    assert result.context is context