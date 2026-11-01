
"""Tests for the ExecutionReportBuilder."""

import pytest

from src.models.event import EventStatus, TemporalEvent
from src.models.evidence import Evidence
from src.models.observation_context import (
    CoverageScope,
    ObservationContext,
)
from src.models.pipeline_result import PipelineResult
from src.models.execution_timeline import ExecutionTimeline
from src.models.verification_criteria import CriterionRequirement
from src.models.verification_result import (
    CriterionResult,
    CriterionResultStatus,
    VerificationResult,
    VerificationVerdict,
)
from src.verification.execution_report_builder import (
    ExecutionReportBuilder,
)


def make_pipeline_result() -> PipelineResult:
    context = ObservationContext(
        fps=30.0,
        frame_count=10,
        frame_start=0,
        frame_end=9,
        time_offset_seconds=0.0,
        coverage_scope=CoverageScope.COMPLETE_TASK,
    )

    evidence = Evidence(
        evidence_id="evidence-1",
        evidence_type="CENTROID_DISPLACEMENT",
        description="Object movement observed across consecutive frames.",
        frame_start=1,
        frame_end=5,
        entity_ids=("object",),
    )

    event = TemporalEvent(
        event_id="event-1",
        event_type="OBJECT_MOVING",
        status=EventStatus.DETECTED,
        frame_start=1,
        frame_end=5,
        timestamp_start=0.1,
        timestamp_end=0.5,
        entity_ids=("object",),
        confidence=0.95,
        evidence=(evidence,),
    )

    timeline = ExecutionTimeline(
        events=(event,),
        context=context,
    )

    criterion_result = CriterionResult(
        criterion_id="object_movement",
        requirement=CriterionRequirement.REQUIRED,
        is_mandatory=True,
        status=CriterionResultStatus.SATISFIED,
        explanation="The object movement criterion was satisfied.",
        event_ids=("event-1",),
        evidence_ids=("evidence-1",),
    )

    verification = VerificationResult(
        verdict=VerificationVerdict.VERIFIED,
        criterion_results=(criterion_result,),
        summary="The execution was verified.",
        coverage_scope=CoverageScope.COMPLETE_TASK,
        covers_entire_video=True,
    )

    return PipelineResult(
        entity_observations=(),
        spatial_relationships=(),
        region_observations=(),
        events=(event,),
        verification=verification,
        context=context,
        execution_timeline=timeline,
    )


def test_builder_creates_execution_report():
    pipeline_result = make_pipeline_result()

    builder = ExecutionReportBuilder()
    report = builder.build(pipeline_result)

    assert report.execution_timeline is pipeline_result.execution_timeline
    assert report.verification is pipeline_result.verification


def test_builder_preserves_events_and_evidence():
    pipeline_result = make_pipeline_result()

    report = ExecutionReportBuilder().build(pipeline_result)

    assert report.events == pipeline_result.events
    assert len(report.events) == 1
    assert report.events[0].event_id == "event-1"

    assert len(report.evidence) == 1
    assert report.evidence[0].evidence_id == "evidence-1"


def test_builder_preserves_verification_result():
    pipeline_result = make_pipeline_result()

    report = ExecutionReportBuilder().build(pipeline_result)

    assert report.verdict == VerificationVerdict.VERIFIED
    assert report.summary == "The execution was verified."
    assert len(report.criterion_results) == 1
    assert report.criterion_results[0].criterion_id == "object_movement"


def test_builder_rejects_invalid_input():
    builder = ExecutionReportBuilder()

    with pytest.raises(
        ValueError,
        match="pipeline_result must be a PipelineResult",
    ):
        builder.build(None)



def test_builder_does_not_modify_pipeline_result():
    pipeline_result = make_pipeline_result()

    original_events = pipeline_result.events
    original_verification = pipeline_result.verification
    original_timeline = pipeline_result.execution_timeline

    report = ExecutionReportBuilder().build(pipeline_result)

    # The original pipeline result remains unchanged.
    assert pipeline_result.events is original_events
    assert pipeline_result.verification is original_verification
    assert pipeline_result.execution_timeline is original_timeline

    # The report preserves the original timeline and verification.
    assert report.execution_timeline is original_timeline
    assert report.verification is original_verification

    # Both event collections contain the same event instances.
    assert report.events == original_events
    assert len(report.events) == len(original_events)
    assert all(
        report_event is original_event
        for report_event, original_event in zip(
            report.events,
            original_events,
        )
    )
