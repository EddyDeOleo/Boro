
"""Presentation model for an individual verification criterion."""

from dataclasses import dataclass

from src.models.verification_criteria import CriterionRequirement
from src.models.verification_result import CriterionResultStatus


@dataclass(frozen=True)
class CriterionView:
    """Presentation-ready representation of a verification criterion."""

    criterion_id: str
    requirement: str
    status: str
    is_mandatory: bool
    explanation: str

    event_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate criterion presentation data."""

        for field_name, value in (
            ("criterion_id", self.criterion_id),
            ("requirement", self.requirement),
            ("status", self.status),
            ("explanation", self.explanation),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string."
                )

        if self.requirement not in {
            requirement.value
            for requirement in CriterionRequirement
        }:
            raise ValueError("Invalid criterion requirement.")

        if self.status not in {
            status.value
            for status in CriterionResultStatus
        }:
            raise ValueError("Invalid criterion result status.")

        if not isinstance(self.is_mandatory, bool):
            raise ValueError("is_mandatory must be a boolean.")

        self._validate_identifiers(self.event_ids, "event_ids")
        self._validate_identifiers(self.evidence_ids, "evidence_ids")

    @staticmethod
    def _validate_identifiers(
        identifiers: tuple[str, ...],
        field_name: str,
    ) -> None:
        """Validate unique, non-empty identifier tuples."""

        if not isinstance(identifiers, tuple):
            raise ValueError(f"{field_name} must be a tuple.")

        for identifier in identifiers:
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(
                    f"Every value in {field_name} must be a non-empty string."
                )

        if len(identifiers) != len(set(identifiers)):
            raise ValueError(
                f"{field_name} must not contain duplicates."
            )
