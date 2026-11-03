
"""Adapt task definitions into Boro verification domain models."""

from collections.abc import Mapping
from dataclasses import dataclass

from src.models.task_definition import TaskDefinition
from src.models.task_roles import TaskRoles
from src.models.verification_criteria import (
    CriterionRequirement,
    VerificationCriteria,
)


EVENT_REQUIRED_EVIDENCE_TYPES = {
    "OBJECT_MOVING": ("CENTROID_DISPLACEMENT",),
    "OBJECT_MOVING_WITH_EFFECTOR": ("COORDINATED_ENTITY_MOTION",),
    "RELATIVE_SEPARATION": (),
    "REGION_ENTRY": ("ENTITY_ENTERED_REGION",),
    "STABLE_PLACEMENT": ("STABLE_IN_REGION",),
}

EVENT_ONE_OF_EVIDENCE_TYPES = {
    "RELATIVE_SEPARATION": (
        "MASK_DISTANCE_SEPARATION",
        "RELATIVE_CENTROID_SEPARATION",
    ),
}


class TaskDefinitionAdapterError(ValueError):
    """Raised when a task definition cannot be adapted safely."""


@dataclass(frozen=True)
class AdaptedTaskDefinition:
    """Domain configuration derived from a structured task definition."""

    task_roles: TaskRoles
    criteria: tuple[VerificationCriteria, ...]


def _validate_id_mapping(
    mapping: Mapping[str, str],
    field_name: str,
) -> dict[str, str]:
    if not isinstance(mapping, Mapping):
        raise TaskDefinitionAdapterError(
            f"{field_name} must be a mapping."
        )

    normalized: dict[str, str] = {}

    for role, identifier in mapping.items():
        if not isinstance(role, str) or not role.strip():
            raise TaskDefinitionAdapterError(
                f"{field_name} contains an invalid role."
            )

        if not isinstance(identifier, str) or not identifier.strip():
            raise TaskDefinitionAdapterError(
                f"{field_name}[{role!r}] must be a non-empty string."
            )

        normalized[role] = identifier.strip()

    if len(set(normalized.values())) != len(normalized):
        raise TaskDefinitionAdapterError(
            f"{field_name} cannot map multiple roles to the same ID."
        )

    return normalized


class TaskDefinitionAdapter:
    """Convert validated task definitions into verification inputs."""

    def adapt(
        self,
        task_definition: TaskDefinition,
        entity_id_map: Mapping[str, str],
        region_id_map: Mapping[str, str],
    ) -> AdaptedTaskDefinition:
        """Build TaskRoles and VerificationCriteria.

        The ID maps connect semantic task roles to IDs already assigned
        by the perception and region configuration layers.

        This adapter does not create tracking prompts or region geometry.
        """

        if not isinstance(task_definition, TaskDefinition):
            raise TypeError(
                "task_definition must be a TaskDefinition instance."
            )

        if task_definition.unresolved_items:
            raise TaskDefinitionAdapterError(
                "Task definition contains unresolved items: "
                f"{list(task_definition.unresolved_items)}."
            )

        normalized_entity_ids = _validate_id_mapping(
            entity_id_map,
            "entity_id_map",
        )
        normalized_region_ids = _validate_id_mapping(
            region_id_map,
            "region_id_map",
        )

        declared_entity_roles = {
            entity.role for entity in task_definition.entities
        }
        declared_region_roles = {
            region.role for region in task_definition.regions
        }

        unknown_entity_roles = (
            set(normalized_entity_ids) - declared_entity_roles
        )
        unknown_region_roles = (
            set(normalized_region_ids) - declared_region_roles
        )

        if unknown_entity_roles:
            raise TaskDefinitionAdapterError(
                "entity_id_map contains undeclared roles: "
                f"{sorted(unknown_entity_roles)}."
            )

        if unknown_region_roles:
            raise TaskDefinitionAdapterError(
                "region_id_map contains undeclared roles: "
                f"{sorted(unknown_region_roles)}."
            )

        required_entity_roles = {
            role
            for criterion in task_definition.criteria
            for role in criterion.entity_roles
        }
        required_region_roles = {
            criterion.region_role
            for criterion in task_definition.criteria
            if criterion.region_role is not None
        }

        missing_entity_roles = (
            required_entity_roles - set(normalized_entity_ids)
        )
        missing_region_roles = (
            required_region_roles - set(normalized_region_ids)
        )

        if missing_entity_roles:
            raise TaskDefinitionAdapterError(
                "Missing entity IDs for task roles: "
                f"{sorted(missing_entity_roles)}."
            )

        if missing_region_roles:
            raise TaskDefinitionAdapterError(
                "Missing region IDs for task roles: "
                f"{sorted(missing_region_roles)}."
            )

        entities_by_role = {
            entity.role: entity
            for entity in task_definition.entities
        }
        regions_by_role = {
            region.role: region
            for region in task_definition.regions
        }

        for role in required_entity_roles:
            if not entities_by_role[role].required:
                raise TaskDefinitionAdapterError(
                    f"Criterion references optional entity role: {role}."
                )

        for role in required_region_roles:
            if not regions_by_role[role].required:
                raise TaskDefinitionAdapterError(
                    f"Criterion references optional region role: {role}."
                )

        task_roles = TaskRoles(
            entity_roles={
                role: normalized_entity_ids[role]
                for role in declared_entity_roles
                if role in normalized_entity_ids
            },
            region_roles={
                role: normalized_region_ids[role]
                for role in declared_region_roles
                if role in normalized_region_ids
            },
        )

        adapted_criteria: list[VerificationCriteria] = []

        for criterion in task_definition.criteria:
            evidence_types = EVENT_REQUIRED_EVIDENCE_TYPES.get(
                criterion.event_type
            )

            if evidence_types is None:
                raise TaskDefinitionAdapterError(
                    "No evidence mapping exists for event type: "
                    f"{criterion.event_type}."
                )

            one_of_evidence_types = EVENT_ONE_OF_EVIDENCE_TYPES.get(
                criterion.event_type,
                (),
            )

            adapted_criteria.append(
                VerificationCriteria(
                    criterion_id=criterion.criterion_id,
                    description=criterion.description,
                    event_type=criterion.event_type,
                    requirement=CriterionRequirement.REQUIRED,
                    entity_roles=criterion.entity_roles,
                    region_role=criterion.region_role,
                    required_evidence_types=evidence_types,
                    is_mandatory=criterion.is_mandatory,
                    one_of_evidence_types=one_of_evidence_types,
                )
            )

        return AdaptedTaskDefinition(
            task_roles=task_roles,
            criteria=tuple(adapted_criteria),
        )
