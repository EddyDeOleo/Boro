
"""Declarative criteria for task verification."""

from dataclasses import dataclass
from enum import Enum


class CriterionRequirement(str, Enum):
    """Defines whether an event is expected or prohibited."""

    REQUIRED = "REQUIRED"
    FORBIDDEN = "FORBIDDEN"


@dataclass(frozen=True)
class VerificationCriteria:
    """Defines one declarative condition to evaluate during verification.

    Criteria refer to semantic roles rather than perception-provider IDs.
    This keeps task definitions independent of specific robots and
    perception implementations.
    """

    criterion_id: str
    description: str
    event_type: str
    requirement: CriterionRequirement

    entity_roles: tuple[str, ...] = ()
    region_role: str | None = None

    required_evidence_types: tuple[str, ...] = ()
    one_of_evidence_types: tuple[str, ...] = ()
    is_mandatory: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.criterion_id, str) or not self.criterion_id.strip():
            raise ValueError("criterion_id must be a non-empty string.")

        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string.")

        if not isinstance(self.event_type, str) or not self.event_type.strip():
            raise ValueError("event_type must be a non-empty string.")

        if not isinstance(self.requirement, CriterionRequirement):
            raise ValueError(
                "requirement must be a CriterionRequirement."
            )

        if not isinstance(self.is_mandatory, bool):
            raise ValueError("is_mandatory must be a boolean.")

        if not isinstance(self.entity_roles, tuple):
            raise ValueError("entity_roles must be a tuple.")

        if not isinstance(self.required_evidence_types, tuple):
            raise ValueError(
                "required_evidence_types must be a tuple."
            )

        if not isinstance(self.one_of_evidence_types, tuple):
            raise ValueError(
                "one_of_evidence_types must be a tuple."
            )

        if any(
            not isinstance(role, str) or not role.strip()
            for role in self.entity_roles
        ):
            raise ValueError(
                "Every entity role must be a non-empty string."
            )

        if len(set(self.entity_roles)) != len(self.entity_roles):
            raise ValueError("entity_roles cannot contain duplicates.")

        if self.region_role is not None:
            if (
                not isinstance(self.region_role, str)
                or not self.region_role.strip()
            ):
                raise ValueError(
                    "region_role must be None or a non-empty string."
                )

        if any(
            not isinstance(evidence_type, str) or not evidence_type.strip()
            for evidence_type in self.required_evidence_types
        ):
            raise ValueError(
                "Every required evidence type must be a non-empty string."
            )

        if (
            len(set(self.required_evidence_types))
            != len(self.required_evidence_types)
        ):
            raise ValueError(
                "required_evidence_types cannot contain duplicates."
            )

        if any(
            not isinstance(evidence_type, str) or not evidence_type.strip()
            for evidence_type in self.one_of_evidence_types
        ):
            raise ValueError(
                "Every one-of evidence type must be a non-empty string."
            )

        if len(set(self.one_of_evidence_types)) != len(
            self.one_of_evidence_types
        ):
            raise ValueError(
                "one_of_evidence_types cannot contain duplicates."
            )

