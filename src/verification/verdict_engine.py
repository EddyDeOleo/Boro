
"""Evaluate verification criteria against detected temporal events."""

from collections.abc import Sequence

from src.models.event import EventStatus, TemporalEvent
from src.models.observation_context import (
    CoverageScope,
    ObservationContext,
)
from src.models.task_roles import TaskRoles
from src.models.verification_criteria import (
    CriterionRequirement,
    VerificationCriteria,
)
from src.models.verification_result import (
    CriterionResult,
    CriterionResultStatus,
    VerificationResult,
    VerificationVerdict,
)


class VerdictEngine:
    """Evaluate task criteria and produce an explainable verdict."""

    OBSERVABILITY_EVIDENCE_TYPE = "OBSERVABILITY_CONFIRMED"

    def evaluate(
        self,
        criteria: Sequence[VerificationCriteria],
        events: Sequence[TemporalEvent],
        task_roles: TaskRoles,
        context: ObservationContext,
    ) -> VerificationResult:
        """Evaluate criteria against events and observation coverage."""

        criteria = self._validate_criteria(criteria)
        events = self._validate_events(events)

        if not isinstance(task_roles, TaskRoles):
            raise ValueError("task_roles must be a TaskRoles instance.")

        if not isinstance(context, ObservationContext):
            raise ValueError(
                "context must be an ObservationContext instance."
            )

        criterion_results = tuple(
            self._evaluate_criterion(
                criterion=criterion,
                events=events,
                task_roles=task_roles,
                context=context,
            )
            for criterion in criteria
        )

        verdict = self._calculate_verdict(
            criterion_results=criterion_results,
            context=context,
        )

        return VerificationResult(
            verdict=verdict,
            criterion_results=criterion_results,
            summary=self._build_summary(verdict),
            coverage_scope=context.coverage_scope,
            covers_entire_video=context.covers_entire_video,
        )

    @staticmethod
    def _validate_criteria(
        criteria: Sequence[VerificationCriteria],
    ) -> tuple[VerificationCriteria, ...]:
        """Validate criteria and ensure their identifiers are unique."""

        if (
            not isinstance(criteria, Sequence)
            or isinstance(criteria, (str, bytes))
        ):
            raise ValueError("criteria must be a sequence.")

        criteria = tuple(criteria)

        if not criteria:
            raise ValueError("criteria must not be empty.")

        if not all(
            isinstance(criterion, VerificationCriteria)
            for criterion in criteria
        ):
            raise ValueError(
                "Every item in criteria must be a VerificationCriteria."
            )

        criterion_ids = [
            criterion.criterion_id for criterion in criteria
        ]

        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("Criteria must have unique criterion IDs.")

        return criteria

    @staticmethod
    def _validate_events(
        events: Sequence[TemporalEvent],
    ) -> tuple[TemporalEvent, ...]:
        """Validate events and ensure their identifiers are unique."""

        if (
            not isinstance(events, Sequence)
            or isinstance(events, (str, bytes))
        ):
            raise ValueError("events must be a sequence.")

        events = tuple(events)

        if not all(isinstance(event, TemporalEvent) for event in events):
            raise ValueError(
                "Every item in events must be a TemporalEvent."
            )

        event_ids = [event.event_id for event in events]

        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Events must have unique event IDs.")

        return events

    def _evaluate_criterion(
        self,
        criterion: VerificationCriteria,
        events: tuple[TemporalEvent, ...],
        task_roles: TaskRoles,
        context: ObservationContext,
    ) -> CriterionResult:
        """Evaluate one criterion and retain its supporting references."""

        structural_matches = tuple(
            event
            for event in events
            if self._matches_structure(
                event=event,
                criterion=criterion,
                task_roles=task_roles,
            )
        )

        detected_matches = tuple(
            event
            for event in structural_matches
            if event.status == EventStatus.DETECTED
            and self._has_required_evidence(event, criterion)
        )

        incomplete_detected_matches = tuple(
            event
            for event in structural_matches
            if event.status == EventStatus.DETECTED
            and not self._has_required_evidence(event, criterion)
        )

        not_detected_matches = tuple(
            event
            for event in structural_matches
            if event.status == EventStatus.NOT_DETECTED
        )

        not_observable_matches = tuple(
            event
            for event in structural_matches
            if event.status == EventStatus.NOT_OBSERVABLE
        )

        if criterion.requirement == CriterionRequirement.REQUIRED:
            return self._evaluate_required_criterion(
                criterion=criterion,
                detected_matches=detected_matches,
                incomplete_detected_matches=incomplete_detected_matches,
                not_detected_matches=not_detected_matches,
                not_observable_matches=not_observable_matches,
                structural_matches=structural_matches,
                context=context,
            )

        return self._evaluate_forbidden_criterion(
            criterion=criterion,
            detected_matches=detected_matches,
            incomplete_detected_matches=incomplete_detected_matches,
            not_detected_matches=not_detected_matches,
            not_observable_matches=not_observable_matches,
            structural_matches=structural_matches,
            context=context,
        )

    def _matches_structure(
        self,
        event: TemporalEvent,
        criterion: VerificationCriteria,
        task_roles: TaskRoles,
    ) -> bool:
        """Check event type, semantic entities, and optional region."""

        if event.event_type != criterion.event_type:
            return False

        required_entity_ids: set[str] = set()

        for role in criterion.entity_roles:
            if not task_roles.has_entity_role(role):
                return False

            required_entity_ids.add(task_roles.get_entity_id(role))

        if not required_entity_ids.issubset(set(event.entity_ids)):
            return False

        if criterion.region_role is not None:
            if not task_roles.has_region_role(criterion.region_role):
                return False

            expected_region_id = task_roles.get_region_id(
                criterion.region_role
            )

            actual_region_id = event.attributes.get("region_id")

            if actual_region_id != expected_region_id:
                return False

        return True

    @staticmethod
    def _has_required_evidence(
        event: TemporalEvent,
        criterion: VerificationCriteria,
    ) -> bool:
        """Check whether a detected event contains required evidence types."""

        available_types = {
            evidence.evidence_type for evidence in event.evidence
        }

        return set(criterion.required_evidence_types).issubset(
            available_types
        )

    def _evaluate_required_criterion(
        self,
        criterion: VerificationCriteria,
        detected_matches: tuple[TemporalEvent, ...],
        incomplete_detected_matches: tuple[TemporalEvent, ...],
        not_detected_matches: tuple[TemporalEvent, ...],
        not_observable_matches: tuple[TemporalEvent, ...],
        structural_matches: tuple[TemporalEvent, ...],
        context: ObservationContext,
    ) -> CriterionResult:
        """Evaluate a required event."""

        if detected_matches:
            return self._build_criterion_result(
                criterion=criterion,
                status=CriterionResultStatus.SATISFIED,
                explanation=(
                    "The required event was detected with the configured "
                    "entity, region, and evidence requirements."
                ),
                events=detected_matches,
            )

        if incomplete_detected_matches:
            return self._build_criterion_result(
                criterion=criterion,
                status=CriterionResultStatus.INSUFFICIENT_EVIDENCE,
                explanation=(
                    "A matching event was detected, but its evidence "
                    "does not satisfy the configured requirements. "
                    "Its occurrence or absence cannot be established."
                ),
                events=structural_matches,
            )

        reliable_absence = tuple(
            event
            for event in not_detected_matches
            if self._has_complete_observability(event, context)
        )

        if reliable_absence:
            return self._build_criterion_result(
                criterion=criterion,
                status=CriterionResultStatus.VIOLATED,
                explanation=(
                    "The required event was not detected despite explicit "
                    "observability evidence covering the complete task."
                ),
                events=reliable_absence,
            )

        if not_observable_matches:
            return self._build_criterion_result(
                criterion=criterion,
                status=CriterionResultStatus.INSUFFICIENT_EVIDENCE,
                explanation=(
                    "The event could not be observed reliably, so its "
                    "occurrence cannot be established."
                ),
                events=structural_matches,
            )

        if not_detected_matches:
            return self._build_criterion_result(
                criterion=criterion,
                status=CriterionResultStatus.INSUFFICIENT_EVIDENCE,
                explanation=(
                    "The event was reported as not detected, but there is "
                    "no explicit observability evidence covering the "
                    "complete task."
                ),
                events=structural_matches,
            )

        return self._build_criterion_result(
            criterion=criterion,
            status=CriterionResultStatus.INSUFFICIENT_EVIDENCE,
            explanation=(
                "No matching event with sufficient evidence was found. "
                "The absence of a detection does not prove that the event "
                "did not occur."
            ),
            events=structural_matches,
        )

    def _evaluate_forbidden_criterion(
        self,
        criterion: VerificationCriteria,
        detected_matches: tuple[TemporalEvent, ...],
        incomplete_detected_matches: tuple[TemporalEvent, ...],
        not_detected_matches: tuple[TemporalEvent, ...],
        not_observable_matches: tuple[TemporalEvent, ...],
        structural_matches: tuple[TemporalEvent, ...],
        context: ObservationContext,
    ) -> CriterionResult:
        """Evaluate an event or condition that must not occur."""

        if detected_matches:
            return self._build_criterion_result(
                criterion=criterion,
                status=CriterionResultStatus.VIOLATED,
                explanation=(
                    "A forbidden event was detected with the configured "
                    "entity, region, and evidence requirements."
                ),
                events=detected_matches,
            )

        if incomplete_detected_matches:
            return self._build_criterion_result(
                criterion=criterion,
                status=CriterionResultStatus.INSUFFICIENT_EVIDENCE,
                explanation=(
                    "A matching event was detected, but its evidence "
                    "does not satisfy the configured requirements. "
                    "Its occurrence or absence cannot be established."
                ),
                events=structural_matches,
            )

        reliable_absence = tuple(
            event
            for event in not_detected_matches
            if self._has_complete_observability(event, context)
        )

        if reliable_absence:
            return self._build_criterion_result(
                criterion=criterion,
                status=CriterionResultStatus.SATISFIED,
                explanation=(
                    "The forbidden event was not detected, and explicit "
                    "observability evidence covers the complete task."
                ),
                events=reliable_absence,
            )

        if not_observable_matches:
            return self._build_criterion_result(
                criterion=criterion,
                status=CriterionResultStatus.INSUFFICIENT_EVIDENCE,
                explanation=(
                    "The forbidden event could not be observed reliably. "
                    "Its absence cannot be established."
                ),
                events=structural_matches,
            )

        if not_detected_matches:
            return self._build_criterion_result(
                criterion=criterion,
                status=CriterionResultStatus.INSUFFICIENT_EVIDENCE,
                explanation=(
                    "The event was reported as not detected, but the "
                    "available evidence does not establish that it was "
                    "observable throughout the complete task."
                ),
                events=structural_matches,
            )

        return self._build_criterion_result(
            criterion=criterion,
            status=CriterionResultStatus.INSUFFICIENT_EVIDENCE,
            explanation=(
                "No matching event was found, but there is no explicit "
                "evidence proving that the forbidden event was observable "
                "throughout the task."
            ),
            events=structural_matches,
        )

    def _has_complete_observability(
        self,
        event: TemporalEvent,
        context: ObservationContext,
    ) -> bool:
        """Determine whether an event's absence is reliably observable."""

        if context.coverage_scope != CoverageScope.COMPLETE_TASK:
            return False

        if not context.covers_entire_video:
            return False

        if (
            event.frame_start > context.frame_start
            or event.frame_end < context.frame_end
        ):
            return False

        return any(
            evidence.evidence_type == self.OBSERVABILITY_EVIDENCE_TYPE
            and evidence.frame_start <= context.frame_start
            and evidence.frame_end >= context.frame_end
            for evidence in event.evidence
        )

    @staticmethod
    def _build_criterion_result(
        criterion: VerificationCriteria,
        status: CriterionResultStatus,
        explanation: str,
        events: tuple[TemporalEvent, ...],
    ) -> CriterionResult:
        """Build a criterion result with traceable event and evidence IDs."""

        event_ids = tuple(
            dict.fromkeys(event.event_id for event in events)
        )

        evidence_ids = tuple(
            dict.fromkeys(
                evidence.evidence_id
                for event in events
                for evidence in event.evidence
            )
        )

        return CriterionResult(
            criterion_id=criterion.criterion_id,
            requirement=criterion.requirement,
            is_mandatory=criterion.is_mandatory,
            status=status,
            explanation=explanation,
            event_ids=event_ids,
            evidence_ids=evidence_ids,
        )

    @staticmethod
    def _calculate_verdict(
        criterion_results: tuple[CriterionResult, ...],
        context: ObservationContext,
    ) -> VerificationVerdict:
        """Calculate the task verdict from criterion outcomes and coverage."""

        has_failure = any(
            result.status == CriterionResultStatus.VIOLATED
            and (
                result.is_mandatory
                or result.requirement == CriterionRequirement.FORBIDDEN
            )
            for result in criterion_results
        )

        if has_failure:
            return VerificationVerdict.FAILED

        has_unresolved_mandatory_criteria = any(
            result.is_mandatory
            and result.status == CriterionResultStatus.INSUFFICIENT_EVIDENCE
            for result in criterion_results
        )

        if has_unresolved_mandatory_criteria:
            return VerificationVerdict.INSUFFICIENT_EVIDENCE

        has_mandatory_criteria = any(
            result.is_mandatory for result in criterion_results
        )

        if not has_mandatory_criteria:
            return VerificationVerdict.INSUFFICIENT_EVIDENCE

        has_complete_task_coverage = (
            context.coverage_scope == CoverageScope.COMPLETE_TASK
            and context.covers_entire_video
        )

        if not has_complete_task_coverage:
            return VerificationVerdict.INSUFFICIENT_EVIDENCE

        return VerificationVerdict.VERIFIED

    @staticmethod
    def _build_summary(verdict: VerificationVerdict) -> str:
        """Build a concise explanation of the final verdict."""

        summaries = {
            VerificationVerdict.VERIFIED: (
                "All mandatory criteria are satisfied and the available "
                "video covers the complete task."
            ),
            VerificationVerdict.FAILED: (
                "At least one mandatory criterion was violated or a "
                "forbidden condition was detected."
            ),
            VerificationVerdict.INSUFFICIENT_EVIDENCE: (
                "The available evidence or task coverage is insufficient "
                "to verify the complete task."
            ),
        }

        return summaries[verdict]
