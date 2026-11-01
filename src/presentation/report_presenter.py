
"""Transform execution reports into presentation-ready models."""

from __future__ import annotations

from src.models.event import TemporalEvent
from src.models.evidence import Evidence
from src.models.execution_report import ExecutionReport
from src.models.observation_context import ObservationContext
from src.models.verification_result import CriterionResult

from src.presentation.models.criterion_view import CriterionView
from src.presentation.models.evidence_view import EvidenceView
from src.presentation.models.presentation_report import PresentationReport
from src.presentation.models.report_overview import ReportOverview
from src.presentation.models.timeline_event_view import TimelineEventView


class ReportPresenter:
    """Convert domain execution reports into presentation reports."""

    def present(
        self,
        report: ExecutionReport,
    ) -> PresentationReport:
        """Build a presentation report from an execution report."""

        if not isinstance(report, ExecutionReport):
            raise TypeError("report must be an ExecutionReport instance.")

        context = report.execution_timeline.context

        overview = ReportOverview(
            verdict=report.verification.verdict.value,
            summary=report.verification.summary,
            coverage_scope=report.verification.coverage_scope.value,
            covers_entire_context=context.covers_entire_video,
            start_time_seconds=context.start_time_seconds,
            end_time_seconds=context.end_time_seconds,
            duration_seconds=context.duration_seconds,
            fps=context.fps,
            context_frame_count=context.frame_count,
            observed_frame_count=context.observed_frame_count,
        )

        criteria = tuple(
            self._present_criterion(criterion)
            for criterion in report.criterion_results
        )

        timeline = tuple(
            self._present_event(event)
            for event in report.events
        )

        evidence = tuple(
            sorted(
                (
                    self._present_evidence(item, context)
                    for item in report.evidence
                ),
                key=lambda item: (
                    item.frame_start,
                    item.frame_end,
                    item.evidence_id,
                ),
            )
        )

        return PresentationReport(
            overview=overview,
            criteria=criteria,
            timeline=timeline,
            evidence=evidence,
            limitations=report.limitations,
        )

    @staticmethod
    def _present_criterion(
        criterion: CriterionResult,
    ) -> CriterionView:
        """Convert a verification criterion result."""

        return CriterionView(
            criterion_id=criterion.criterion_id,
            requirement=criterion.requirement.value,
            status=criterion.status.value,
            is_mandatory=criterion.is_mandatory,
            explanation=criterion.explanation,
            event_ids=criterion.event_ids,
            evidence_ids=criterion.evidence_ids,
        )

    @staticmethod
    def _present_event(
        event: TemporalEvent,
    ) -> TimelineEventView:
        """Convert a temporal event."""

        return TimelineEventView(
            event_id=event.event_id,
            event_type=event.event_type,
            status=event.status,
            frame_start=event.frame_start,
            frame_end=event.frame_end,
            timestamp_start=event.timestamp_start,
            timestamp_end=event.timestamp_end,
            entity_ids=event.entity_ids,
            confidence=event.confidence,
            evidence_ids=tuple(
                item.evidence_id for item in event.evidence
            ),
            attributes=event.attributes,
        )

    @staticmethod
    def _present_evidence(
        evidence: Evidence,
        context: ObservationContext,
    ) -> EvidenceView:
        """Convert evidence and derive its timestamps from video context."""

        timestamp_start = (
            context.time_offset_seconds
            + evidence.frame_start / context.fps
        )

        timestamp_end = (
            context.time_offset_seconds
            + evidence.frame_end / context.fps
        )

        return EvidenceView(
            evidence_id=evidence.evidence_id,
            evidence_type=evidence.evidence_type,
            description=evidence.description,
            frame_start=evidence.frame_start,
            frame_end=evidence.frame_end,
            timestamp_start_seconds=timestamp_start,
            timestamp_end_seconds=timestamp_end,
            observation_ids=evidence.observation_ids,
            entity_ids=evidence.entity_ids,
            metrics=evidence.metrics,
            limitations=evidence.limitations,
        )

