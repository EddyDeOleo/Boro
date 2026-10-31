"""Models representing task verification results."""

from dataclasses import dataclass
from enum import Enum

from src.models.observation_context import CoverageScope
from src.models.verification_criteria import CriterionRequirement


class CriterionResultStatus(str, Enum):
    """Possible outcomes for an individual verification criterion."""

    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class VerificationVerdict(str, Enum):
    """Possible outcomes for the complete task verification."""

    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class CriterionResult:
    """Result of evaluating one verification criterion."""

    criterion_id: str
    requirement: CriterionRequirement
    is_mandatory: bool
    status: CriterionResultStatus
    explanation: str
    event_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the criterion result."""

        if not isinstance(self.criterion_id, str) or not self.criterion_id.strip():
            raise ValueError("criterion_id must be a non-empty string.")

        if not isinstance(self.requirement, CriterionRequirement):
            raise ValueError("requirement must be a CriterionRequirement.")

        if not isinstance(self.is_mandatory, bool):
            raise ValueError("is_mandatory must be a boolean.")

        if not isinstance(self.status, CriterionResultStatus):
            raise ValueError("status must be a CriterionResultStatus.")

        if not isinstance(self.explanation, str) or not self.explanation.strip():
            raise ValueError("explanation must be a non-empty string.")

        self._validate_identifiers(self.event_ids, "event_ids")
        self._validate_identifiers(self.evidence_ids, "evidence_ids")

    @staticmethod
    def _validate_identifiers(
        identifiers: tuple[str, ...],
        field_name: str,
    ) -> None:
        """Validate a tuple of unique, non-empty identifiers."""

        if not isinstance(identifiers, tuple):
            raise ValueError(f"{field_name} must be a tuple.")

        for identifier in identifiers:
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(
                    f"Every value in {field_name} must be a non-empty string."
                )

        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{field_name} must not contain duplicates.")


@dataclass(frozen=True)
class VerificationResult:
    """Final verification result for a robotic task."""

    verdict: VerificationVerdict
    criterion_results: tuple[CriterionResult, ...]
    summary: str
    coverage_scope: CoverageScope = CoverageScope.UNKNOWN
    covers_entire_video: bool = False

    def __post_init__(self) -> None:
        """Validate the complete verification result."""

        if not isinstance(self.verdict, VerificationVerdict):
            raise ValueError("verdict must be a VerificationVerdict.")

        if not isinstance(self.criterion_results, tuple):
            raise ValueError("criterion_results must be a tuple.")

        if not self.criterion_results:
            raise ValueError("criterion_results must not be empty.")

        if not all(
            isinstance(result, CriterionResult)
            for result in self.criterion_results
        ):
            raise ValueError(
                "Every item in criterion_results must be a CriterionResult."
            )

        criterion_ids = [
            result.criterion_id for result in self.criterion_results
        ]

        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("criterion_results must have unique criterion IDs.")

        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("summary must be a non-empty string.")

        if not isinstance(self.coverage_scope, CoverageScope):
            raise ValueError("coverage_scope must be a CoverageScope.")

        if not isinstance(self.covers_entire_video, bool):
            raise ValueError("covers_entire_video must be a boolean.")

        expected_verdict = self._calculate_expected_verdict()

        if self.verdict != expected_verdict:
            raise ValueError(
                f"Inconsistent verification verdict: expected "
                f"{expected_verdict.value}, received {self.verdict.value}."
            )

    def _calculate_expected_verdict(self) -> VerificationVerdict:
        """Calculate the verdict implied by criterion results and coverage."""

        has_failure = any(
            result.status == CriterionResultStatus.VIOLATED
            and (
                result.is_mandatory
                or result.requirement == CriterionRequirement.FORBIDDEN
            )
            for result in self.criterion_results
        )

        if has_failure:
            return VerificationVerdict.FAILED

        has_unresolved_mandatory_criteria = any(
            result.is_mandatory
            and result.status == CriterionResultStatus.INSUFFICIENT_EVIDENCE
            for result in self.criterion_results
        )

        if has_unresolved_mandatory_criteria:
            return VerificationVerdict.INSUFFICIENT_EVIDENCE

        has_mandatory_criteria = any(
            result.is_mandatory for result in self.criterion_results
        )

        if not has_mandatory_criteria:
            return VerificationVerdict.INSUFFICIENT_EVIDENCE

        has_complete_task_coverage = (
            self.coverage_scope == CoverageScope.COMPLETE_TASK
            and self.covers_entire_video
        )

        if not has_complete_task_coverage:
            return VerificationVerdict.INSUFFICIENT_EVIDENCE

        return VerificationVerdict.VERIFIED
