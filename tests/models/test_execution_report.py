
"""Tests for the ExecutionReport model."""

import pytest

from src.models.event import EventStatus, TemporalEvent
from src.models.evidence import Evidence
from src.models.execution_report import ExecutionReport
from src.models.execution_timeline import ExecutionTimeline
from src.models.observation_context import (
    CoverageScope,
    ObservationContext,
)
from src.models.verification_criteria import CriterionRequirement
from src.models.verification_result import (
    CriterionResult,
    CriterionResultStatus,
    VerificationResult,
    VerificationVerdict,
)


def make_context(
    coverage_scope: CoverageScope = CoverageScope.COMPLETE_TASK,
) -> ObservationContext:
    return ObservationContext(
        fps=30.0,
        frame_count=10,
        frame_start=0,
        frame_end=9,
        time_offset_seconds=0.0,
        coverage_scope=coverage_scope,
    )


def make_evidence(
    evidence_id: str = "evidence-1",
    description: str = "Object movement observed across consecutive frames.",
    limitations: tuple[str, ...] = (),
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        evidence_type="CENTROID_DISPLACEMENT",
        description=description,
        frame_start=1,
        frame_end=5,
        entity_ids=("object",),
        limitations=limitations,
    )


def make_event(
    event_id: str = "event-1",
    evidence: tuple[Evidence, ...] | None = None,
    timestamp_start: float = 0.1,
) -> TemporalEvent:
    event_evidence = (
        evidence if evidence is not None else (make_evidence(),)
    )

    return TemporalEvent(
        event_id=event_id,
        event_type="OBJECT_MOVING",
        status=EventStatus.DETECTED,
        frame_start=1,
        frame_end=5,
        timestamp_start=timestamp_start,
        timestamp_end=0.5,
        entity_ids=("object",),
        confidence=0.95,
        evidence=event_evidence,
    )


def make_criterion_result(
    event_ids: tuple[str, ...] = ("event-1",),
    evidence_ids: tuple[str, ...] = ("evidence-1",),
) -> CriterionResult:
    return CriterionResult(
        criterion_id="object_movement",
        requirement=CriterionRequirement.REQUIRED,
        is_mandatory=True,
        status=CriterionResultStatus.SATISFIED,
        explanation="The object movement criterion was satisfied.",
        event_ids=event_ids,
        evidence_ids=evidence_ids,
    )


def make_verification(
    coverage_scope: CoverageScope = CoverageScope.COMPLETE_TASK,
    criterion_results: tuple[CriterionResult, ...] | None = None,
) -> VerificationResult:
    results = (
        criterion_results
        if criterion_results is not None
        else (make_criterion_result(),)
    )

    verdict = (
        VerificationVerdict.VERIFIED
        if coverage_scope == CoverageScope.COMPLETE_TASK
        else VerificationVerdict.INSUFFICIENT_EVIDENCE
    )

    return VerificationResult(
        verdict=verdict,
        criterion_results=results,
        summary="The execution was evaluated successfully.",
        coverage_scope=coverage_scope,
        covers_entire_video=(
            coverage_scope == CoverageScope.COMPLETE_TASK
        ),
    )


def make_timeline(
    events: tuple[TemporalEvent, ...] | None = None,
    coverage_scope: CoverageScope = CoverageScope.COMPLETE_TASK,
) -> ExecutionTimeline:
    timeline_events = (
        events if events is not None else (make_event(),)
    )

    return ExecutionTimeline(
        events=timeline_events,
        context=make_context(coverage_scope),
    )


def test_execution_report_exposes_original_data():
    event = make_event()
    timeline = make_timeline(events=(event,))
    verification = make_verification()

    report = ExecutionReport(
        execution_timeline=timeline,
        verification=verification,
    )

    assert report.events is timeline.events
    assert report.verdict == verification.verdict
    assert report.summary == verification.summary
    assert report.criterion_results is verification.criterion_results


def test_execution_report_collects_evidence():
    evidence = make_evidence()
    event = make_event(evidence=(evidence,))

    report = ExecutionReport(
        execution_timeline=make_timeline(events=(event,)),
        verification=make_verification(),
    )

    assert report.evidence == (evidence,)


def test_execution_report_deduplicates_evidence_by_id():
    evidence = make_evidence()

    first_event = make_event(
        event_id="event-1",
        evidence=(evidence,),
        timestamp_start=0.1,
    )
    second_event = make_event(
        event_id="event-2",
        evidence=(evidence,),
        timestamp_start=0.2,
    )

    criterion = make_criterion_result(
        event_ids=("event-1", "event-2"),
        evidence_ids=("evidence-1",),
    )

    report = ExecutionReport(
        execution_timeline=make_timeline(
            events=(first_event, second_event),
        ),
        verification=make_verification(
            criterion_results=(criterion,),
        ),
    )

    assert report.evidence == (evidence,)


def test_execution_report_collects_unique_limitations():
    first_evidence = make_evidence(
        evidence_id="evidence-1",
        limitations=(
            "Physical contact cannot be confirmed.",
            "Object orientation is not evaluated.",
        ),
    )
    second_evidence = make_evidence(
        evidence_id="evidence-2",
        limitations=(
            "Physical contact cannot be confirmed.",
            "The camera view is two-dimensional.",
        ),
    )

    first_event = make_event(
        event_id="event-1",
        evidence=(first_evidence,),
        timestamp_start=0.1,
    )
    second_event = make_event(
        event_id="event-2",
        evidence=(second_evidence,),
        timestamp_start=0.2,
    )

    criterion = make_criterion_result(
        event_ids=("event-1", "event-2"),
        evidence_ids=("evidence-1", "evidence-2"),
    )

    report = ExecutionReport(
        execution_timeline=make_timeline(
            events=(first_event, second_event),
        ),
        verification=make_verification(
            criterion_results=(criterion,),
        ),
    )

    assert report.limitations == (
        "Physical contact cannot be confirmed.",
        "Object orientation is not evaluated.",
        "The camera view is two-dimensional.",
    )


def test_execution_report_rejects_invalid_timeline():
    with pytest.raises(
        ValueError,
        match="execution_timeline must be an ExecutionTimeline",
    ):
        ExecutionReport(
            execution_timeline=None,
            verification=make_verification(),
        )


def test_execution_report_rejects_invalid_verification():
    with pytest.raises(
        ValueError,
        match="verification must be a VerificationResult",
    ):
        ExecutionReport(
            execution_timeline=make_timeline(),
            verification=None,
        )


def test_execution_report_rejects_coverage_scope_mismatch():
    with pytest.raises(
        ValueError,
        match="coverage_scope must match",
    ):
        ExecutionReport(
            execution_timeline=make_timeline(
                coverage_scope=CoverageScope.COMPLETE_TASK,
            ),
            verification=make_verification(
                coverage_scope=CoverageScope.PARTIAL_SEGMENT,
            ),
        )


def test_execution_report_rejects_unknown_event_reference():
    criterion = make_criterion_result(
        event_ids=("missing-event",),
        evidence_ids=("evidence-1",),
    )

    with pytest.raises(
        ValueError,
        match="references unknown event IDs",
    ):
        ExecutionReport(
            execution_timeline=make_timeline(),
            verification=make_verification(
                criterion_results=(criterion,),
            ),
        )


def test_execution_report_rejects_unknown_evidence_reference():
    criterion = make_criterion_result(
        event_ids=("event-1",),
        evidence_ids=("missing-evidence",),
    )

    with pytest.raises(
        ValueError,
        match="references unknown evidence IDs",
    ):
        ExecutionReport(
            execution_timeline=make_timeline(),
            verification=make_verification(
                criterion_results=(criterion,),
            ),
        )


def test_execution_report_rejects_conflicting_evidence_ids():
    first_evidence = make_evidence(
        evidence_id="shared-evidence",
        description="First evidence description.",
    )
    second_evidence = make_evidence(
        evidence_id="shared-evidence",
        description="Different evidence description.",
    )

    first_event = make_event(
        event_id="event-1",
        evidence=(first_evidence,),
        timestamp_start=0.1,
    )
    second_event = make_event(
        event_id="event-2",
        evidence=(second_evidence,),
        timestamp_start=0.2,
    )

    criterion = make_criterion_result(
        event_ids=("event-1", "event-2"),
        evidence_ids=("shared-evidence",),
    )

    with pytest.raises(
        ValueError,
        match="Conflicting evidence items share ID",
    ):
        ExecutionReport(
            execution_timeline=make_timeline(
                events=(first_event, second_event),
            ),
            verification=make_verification(
                criterion_results=(criterion,),
            ),
        )


def test_execution_report_rejects_duplicate_event_ids():
    event = make_event()

    criterion = make_criterion_result()

    with pytest.raises(
        ValueError,
        match="unique event IDs",
    ):
        ExecutionReport(
            execution_timeline=make_timeline(
                events=(event, event),
            ),
            verification=make_verification(
                criterion_results=(criterion,),
            ),
        )


def test_execution_report_rejects_evidence_from_unreferenced_event():
    first_evidence = make_evidence(
        evidence_id="evidence-1",
    )
    second_evidence = make_evidence(
        evidence_id="evidence-2",
    )

    first_event = make_event(
        event_id="event-1",
        evidence=(first_evidence,),
        timestamp_start=0.1,
    )
    second_event = make_event(
        event_id="event-2",
        evidence=(second_evidence,),
        timestamp_start=0.2,
    )

    criterion = make_criterion_result(
        event_ids=("event-1",),
        evidence_ids=("evidence-2",),
    )

    with pytest.raises(
        ValueError,
        match="evidence not associated with its events",
    ):
        ExecutionReport(
            execution_timeline=make_timeline(
                events=(first_event, second_event),
            ),
            verification=make_verification(
                criterion_results=(criterion,),
            ),
        )


def test_execution_report_accepts_evidence_from_referenced_events():
    first_evidence = make_evidence(
        evidence_id="evidence-1",
    )
    second_evidence = make_evidence(
        evidence_id="evidence-2",
    )

    first_event = make_event(
        event_id="event-1",
        evidence=(first_evidence,),
        timestamp_start=0.1,
    )
    second_event = make_event(
        event_id="event-2",
        evidence=(second_evidence,),
        timestamp_start=0.2,
    )

    criterion = make_criterion_result(
        event_ids=("event-1", "event-2"),
        evidence_ids=("evidence-1", "evidence-2"),
    )

    report = ExecutionReport(
        execution_timeline=make_timeline(
            events=(first_event, second_event),
        ),
        verification=make_verification(
            criterion_results=(criterion,),
        ),
    )

    assert report.criterion_results[0].event_ids == (
        "event-1",
        "event-2",
    )
    assert report.criterion_results[0].evidence_ids == (
        "evidence-1",
        "evidence-2",
    )
