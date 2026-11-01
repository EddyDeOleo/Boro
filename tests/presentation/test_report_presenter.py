"""Tests for the ReportPresenter."""

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

from src.presentation.report_presenter import ReportPresenter
from src.presentation.models.presentation_report import PresentationReport


@pytest.fixture
def valid_evidence():
    return Evidence(
        evidence_id="evidence-001",
        evidence_type="OBJECT_MOTION",
        description="Object movement was observed.",
        frame_start=10,
        frame_end=20,
        observation_ids=("observation-001",),
        entity_ids=("object",),
        metrics={"displacement_pixels": 45.5},
        limitations=("Visual evidence does not prove physical contact.",),
    )


@pytest.fixture
def valid_event(valid_evidence):
    return TemporalEvent(
        event_id="event-001",
        event_type="OBJECT_MOVING",
        status=EventStatus.DETECTED,
        frame_start=10,
        frame_end=20,
        timestamp_start=5.333333,
        timestamp_end=5.666667,
        entity_ids=("object",),
        confidence=0.95,
        evidence=(valid_evidence,),
        attributes={"movement_type": "translation"},
    )


@pytest.fixture
def valid_context():
    return ObservationContext(
        fps=30.0,
        frame_count=100,
        frame_start=0,
        frame_end=99,
        time_offset_seconds=5.0,
        coverage_scope=CoverageScope.COMPLETE_TASK,
    )


@pytest.fixture
def valid_report(valid_event, valid_context):
    timeline = ExecutionTimeline(
        events=(valid_event,),
        context=valid_context,
    )

    criterion = CriterionResult(
        criterion_id="object_movement",
        requirement=CriterionRequirement.REQUIRED,
        is_mandatory=True,
        status=CriterionResultStatus.SATISFIED,
        explanation="Object movement was successfully detected.",
        event_ids=("event-001",),
        evidence_ids=("evidence-001",),
    )

    verification = VerificationResult(
        verdict=VerificationVerdict.VERIFIED,
        criterion_results=(criterion,),
        summary="The task satisfies all mandatory criteria.",
        coverage_scope=CoverageScope.COMPLETE_TASK,
        covers_entire_video=True,
    )

    return ExecutionReport(
        execution_timeline=timeline,
        verification=verification,
    )


class TestReportPresenter:

    def test_present_returns_presentation_report(self, valid_report):
        result = ReportPresenter().present(valid_report)

        assert isinstance(result, PresentationReport)

    def test_preserves_verification_verdict(self, valid_report):
        result = ReportPresenter().present(valid_report)

        assert result.overview.verdict == "VERIFIED"

    def test_preserves_context_coverage(self, valid_report):
        result = ReportPresenter().present(valid_report)

        assert result.overview.covers_entire_context is True

    def test_uses_observation_context_metadata(self, valid_report):
        result = ReportPresenter().present(valid_report)

        assert result.overview.fps == 30.0
        assert result.overview.context_frame_count == 100
        assert result.overview.observed_frame_count == 100
        assert result.overview.start_time_seconds == pytest.approx(5.0)
        assert result.overview.end_time_seconds == pytest.approx(
            5.0 + 99 / 30.0
        )

    def test_converts_criterion_results(self, valid_report):
        result = ReportPresenter().present(valid_report)

        criterion = result.criteria[0]

        assert criterion.criterion_id == "object_movement"
        assert criterion.requirement == "REQUIRED"
        assert criterion.status == "SATISFIED"
        assert criterion.is_mandatory is True
        assert criterion.event_ids == ("event-001",)
        assert criterion.evidence_ids == ("evidence-001",)

    def test_converts_timeline_events(self, valid_report):
        result = ReportPresenter().present(valid_report)

        event = result.timeline[0]

        assert event.event_id == "event-001"
        assert event.event_type == "OBJECT_MOVING"
        assert event.status == EventStatus.DETECTED
        assert event.confidence == pytest.approx(0.95)
        assert event.evidence_ids == ("evidence-001",)

    def test_derives_evidence_timestamps(self, valid_report):
        result = ReportPresenter().present(valid_report)

        evidence = result.evidence[0]

        assert evidence.timestamp_start_seconds == pytest.approx(
            5.0 + 10 / 30.0
        )
        assert evidence.timestamp_end_seconds == pytest.approx(
            5.0 + 20 / 30.0
        )

    def test_preserves_evidence_metadata(self, valid_report):
        result = ReportPresenter().present(valid_report)

        evidence = result.evidence[0]

        assert evidence.evidence_id == "evidence-001"
        assert evidence.observation_ids == ("observation-001",)
        assert evidence.entity_ids == ("object",)
        assert evidence.metrics["displacement_pixels"] == 45.5

    def test_preserves_limitations(self, valid_report):
        result = ReportPresenter().present(valid_report)

        assert result.limitations == (
            "Visual evidence does not prove physical contact.",
        )

    def test_uses_absolute_frame_indices_for_evidence_timestamps(
        self,
        valid_event,
    ):
        context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=10,
            frame_end=70,
            time_offset_seconds=5.0,
            coverage_scope=CoverageScope.PARTIAL_SEGMENT,
        )

        evidence = ReportPresenter._present_evidence(
            valid_event.evidence[0],
            context,
        )

        assert evidence.timestamp_start_seconds == pytest.approx(
            5.0 + 10 / 30.0
        )
        assert evidence.timestamp_end_seconds == pytest.approx(
            5.0 + 20 / 30.0
        )

    def test_preserves_insufficient_evidence_verdict(self, valid_report):
        verification = VerificationResult(
            verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
            criterion_results=(
                CriterionResult(
                    criterion_id="object_movement",
                    requirement=CriterionRequirement.REQUIRED,
                    is_mandatory=True,
                    status=CriterionResultStatus.INSUFFICIENT_EVIDENCE,
                    explanation="There is insufficient evidence.",
                    event_ids=("event-001",),
                    evidence_ids=("evidence-001",),
                ),
            ),
            summary="The task could not be fully verified.",
            coverage_scope=CoverageScope.PARTIAL_SEGMENT,
            covers_entire_video=False,
        )

        partial_context = ObservationContext(
            fps=30.0,
            frame_count=100,
            frame_start=10,
            frame_end=70,
            time_offset_seconds=5.0,
            coverage_scope=CoverageScope.PARTIAL_SEGMENT,
        )

        partial_timeline = ExecutionTimeline(
            events=valid_report.events,
            context=partial_context,
        )

        partial_report = ExecutionReport(
            execution_timeline=partial_timeline,
            verification=verification,
        )

        result = ReportPresenter().present(partial_report)

        assert result.overview.verdict == "INSUFFICIENT_EVIDENCE"
        assert result.overview.covers_entire_context is False
        assert result.overview.coverage_scope == "PARTIAL_SEGMENT"
        assert result.overview.context_frame_count == 100
        assert result.overview.observed_frame_count == 61

    def test_rejects_invalid_report(self):
        with pytest.raises(TypeError):
            ReportPresenter().present("invalid")