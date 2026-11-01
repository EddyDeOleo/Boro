
"""Models representing an explainable robotic execution report."""

from dataclasses import dataclass

from src.models.evidence import Evidence
from src.models.execution_timeline import ExecutionTimeline
from src.models.verification_result import VerificationResult


@dataclass(frozen=True)
class ExecutionReport:
    """Explainable report combining execution events and verification results.

    The report preserves references to the original timeline and verification
    result instead of duplicating their underlying data.
    """

    execution_timeline: ExecutionTimeline
    verification: VerificationResult

    def __post_init__(self) -> None:
        """Validate report consistency and cross-component references."""

        if not isinstance(self.execution_timeline, ExecutionTimeline):
            raise ValueError(
                "execution_timeline must be an ExecutionTimeline."
            )

        if not isinstance(self.verification, VerificationResult):
            raise ValueError(
                "verification must be a VerificationResult."
            )

        if (
            self.verification.coverage_scope
            != self.execution_timeline.coverage_scope
        ):
            raise ValueError(
                "Verification coverage_scope must match "
                "execution timeline coverage_scope."
            )

        self._validate_event_references()
        self._validate_evidence_references()

    def _validate_event_references(self) -> None:
        """Ensure criterion event references exist in the timeline."""

        event_ids = {
            event.event_id
            for event in self.execution_timeline.events
        }

        if len(event_ids) != len(self.execution_timeline.events):
            raise ValueError(
                "Execution timeline must contain unique event IDs."
            )

        for criterion in self.verification.criterion_results:
            missing_event_ids = (
                set(criterion.event_ids) - event_ids
            )

            if missing_event_ids:
                missing = ", ".join(sorted(missing_event_ids))

                raise ValueError(
                    f"Criterion {criterion.criterion_id!r} "
                    f"references unknown event IDs: {missing}."
                )


    def _validate_evidence_references(self) -> None:
        """Ensure criterion evidence belongs to its referenced events."""

        evidence_by_id = {
            item.evidence_id: item
            for item in self.evidence
        }

        event_evidence_by_id = {
            event.event_id: {
                item.evidence_id
                for item in event.evidence
            }
            for event in self.execution_timeline.events
        }

        for criterion in self.verification.criterion_results:
            referenced_evidence_ids = set(criterion.evidence_ids)

            missing_evidence_ids = (
                referenced_evidence_ids - evidence_by_id.keys()
            )

            if missing_evidence_ids:
                missing = ", ".join(sorted(missing_evidence_ids))

                raise ValueError(
                    f"Criterion {criterion.criterion_id!r} "
                    f"references unknown evidence IDs: {missing}."
                )

            associated_evidence_ids = {
                evidence_id
                for event_id in criterion.event_ids
                for evidence_id in event_evidence_by_id[event_id]
            }

            unrelated_evidence_ids = (
                referenced_evidence_ids - associated_evidence_ids
            )

            if unrelated_evidence_ids:
                unrelated = ", ".join(sorted(unrelated_evidence_ids))

                raise ValueError(
                    f"Criterion {criterion.criterion_id!r} "
                    f"references evidence not associated with its "
                    f"events: {unrelated}."
                )


    @property
    def evidence(self) -> tuple[Evidence, ...]:
        """Return unique evidence items in timeline order."""

        evidence_by_id: dict[str, Evidence] = {}

        for event in self.execution_timeline.events:
            for item in event.evidence:
                existing = evidence_by_id.get(item.evidence_id)

                if existing is not None and existing != item:
                    raise ValueError(
                        f"Conflicting evidence items share ID "
                        f"{item.evidence_id!r}."
                    )

                evidence_by_id[item.evidence_id] = item

        return tuple(evidence_by_id.values())

    @property
    def limitations(self) -> tuple[str, ...]:
        """Return unique limitations declared by supporting evidence."""

        limitations: list[str] = []
        seen: set[str] = set()

        for item in self.evidence:
            for limitation in item.limitations:
                if limitation not in seen:
                    seen.add(limitation)
                    limitations.append(limitation)

        return tuple(limitations)

    @property
    def events(self):
        """Return the chronological events from the execution timeline."""

        return self.execution_timeline.events

    @property
    def verdict(self):
        """Return the verification verdict."""

        return self.verification.verdict

    @property
    def summary(self) -> str:
        """Return the verification summary."""

        return self.verification.summary

    @property
    def criterion_results(self):
        """Return the evaluated verification criteria."""

        return self.verification.criterion_results
