
"""Structured task definitions for robotic execution verification."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class TaskEntityRole(str, Enum):
    """Supported semantic roles for task entities."""

    OBJECT = "object"
    EFFECTOR = "effector"


class TaskRegionRole(str, Enum):
    """Supported semantic roles for task regions."""

    TARGET_REGION = "target_region"


class SupportedEventType(str, Enum):
    """Event types currently supported by the verification engine."""

    OBJECT_MOVING = "OBJECT_MOVING"
    OBJECT_MOVING_WITH_EFFECTOR = "OBJECT_MOVING_WITH_EFFECTOR"
    RELATIVE_SEPARATION = "RELATIVE_SEPARATION"
    REGION_ENTRY = "REGION_ENTRY"
    STABLE_PLACEMENT = "STABLE_PLACEMENT"


SUPPORTED_REGION_TYPES = frozenset({
    "TARGET_ZONE",
})

EVENT_REQUIRED_ENTITY_ROLES = {
    SupportedEventType.OBJECT_MOVING: frozenset({
        TaskEntityRole.OBJECT.value,
    }),
    SupportedEventType.OBJECT_MOVING_WITH_EFFECTOR: frozenset({
        TaskEntityRole.OBJECT.value,
        TaskEntityRole.EFFECTOR.value,
    }),
    SupportedEventType.RELATIVE_SEPARATION: frozenset({
        TaskEntityRole.OBJECT.value,
        TaskEntityRole.EFFECTOR.value,
    }),
    SupportedEventType.REGION_ENTRY: frozenset({
        TaskEntityRole.OBJECT.value,
    }),
    SupportedEventType.STABLE_PLACEMENT: frozenset({
        TaskEntityRole.OBJECT.value,
    }),
}

EVENT_REQUIRES_REGION = frozenset({
    SupportedEventType.REGION_ENTRY,
    SupportedEventType.STABLE_PLACEMENT,
})


def _validate_non_empty_string(value: object, field_name: str) -> None:
    """Validate a non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _normalize_strings(
    values: object,
    field_name: str,
) -> tuple[str, ...]:
    """Normalize an iterable of non-empty strings into a tuple."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be an iterable of strings.")

    try:
        normalized = tuple(values)
    except TypeError:
        raise ValueError(f"{field_name} must be iterable.") from None

    for value in normalized:
        _validate_non_empty_string(value, field_name)

    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} cannot contain duplicates.")

    return normalized


@dataclass(frozen=True)
class TaskEntity:
    """Semantic description of an entity involved in a task."""

    role: str
    description: str
    visual_attributes: Mapping[str, str]
    required: bool = True

    def __post_init__(self) -> None:
        allowed_roles = {
            role.value for role in TaskEntityRole
        }

        if not isinstance(self.role, str) or self.role not in allowed_roles:
            raise ValueError(f"Unsupported entity role: {self.role}")

        _validate_non_empty_string(self.description, "description")

        if not isinstance(self.visual_attributes, Mapping):
            raise ValueError("visual_attributes must be a mapping.")

        normalized_attributes: dict[str, str] = {}

        for key, value in self.visual_attributes.items():
            _validate_non_empty_string(key, "visual attribute key")
            _validate_non_empty_string(value, "visual attribute value")
            normalized_attributes[key] = value

        if not isinstance(self.required, bool):
            raise ValueError("required must be a boolean.")

        object.__setattr__(
            self,
            "visual_attributes",
            MappingProxyType(normalized_attributes),
        )


@dataclass(frozen=True)
class TaskRegion:
    """Semantic description of a region required by a task."""

    role: str
    description: str
    region_type: str
    required: bool = True

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.role, "role")
        _validate_non_empty_string(self.description, "description")
        _validate_non_empty_string(self.region_type, "region_type")

        allowed_roles = {
            role.value for role in TaskRegionRole
        }

        if self.role not in allowed_roles:
            raise ValueError(
                f"Unsupported region role: {self.role}."
            )

        if self.region_type not in SUPPORTED_REGION_TYPES:
            raise ValueError(
                f"Unsupported region type: {self.region_type}."
            )

        if (
            self.role == TaskRegionRole.TARGET_REGION.value
            and self.region_type != "TARGET_ZONE"
        ):
            raise ValueError(
                "target_region must use the TARGET_ZONE region type."
            )

        if not isinstance(self.required, bool):
            raise ValueError("required must be a boolean.")


@dataclass(frozen=True)
class TaskCriterion:
    """A verification criterion proposed for a robotic task."""

    criterion_id: str
    description: str
    event_type: str
    entity_roles: tuple[str, ...]
    region_role: str | None = None
    is_mandatory: bool = True

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.criterion_id, "criterion_id")
        _validate_non_empty_string(self.description, "description")
        _validate_non_empty_string(self.event_type, "event_type")

        try:
            supported_event = SupportedEventType(self.event_type)
        except ValueError:
            raise ValueError(
                f"Unsupported event type: {self.event_type}."
            ) from None

        normalized_roles = _normalize_strings(
            self.entity_roles,
            "entity_roles",
        )

        allowed_entity_roles = {
            role.value for role in TaskEntityRole
        }

        unsupported_roles = set(normalized_roles) - allowed_entity_roles

        if unsupported_roles:
            raise ValueError(
                f"Unsupported entity roles: {sorted(unsupported_roles)}."
            )

        required_roles = EVENT_REQUIRED_ENTITY_ROLES[supported_event]

        if not required_roles.issubset(normalized_roles):
            raise ValueError(
                f"{supported_event.value} requires entity roles: "
                f"{sorted(required_roles)}."
            )

        if self.region_role is not None:
            _validate_non_empty_string(self.region_role, "region_role")

            allowed_region_roles = {
                role.value for role in TaskRegionRole
            }

            if self.region_role not in allowed_region_roles:
                raise ValueError(
                    f"Unsupported region role: {self.region_role}."
                )

        if (
            supported_event in EVENT_REQUIRES_REGION
            and self.region_role is None
        ):
            raise ValueError(
                f"{supported_event.value} requires a region_role."
            )

        if (
            supported_event not in EVENT_REQUIRES_REGION
            and self.region_role is not None
        ):
            raise ValueError(
                f"{supported_event.value} does not support a region_role."
            )

        if not isinstance(self.is_mandatory, bool):
            raise ValueError("is_mandatory must be a boolean.")

        object.__setattr__(self, "entity_roles", normalized_roles)


@dataclass(frozen=True)
class TaskDefinition:
    """Immutable, structured definition of a robotic task."""

    task_summary: str
    entities: tuple[TaskEntity, ...]
    regions: tuple[TaskRegion, ...]
    criteria: tuple[TaskCriterion, ...]
    unresolved_items: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.task_summary, "task_summary")

        for field_name, expected_type in (
            ("entities", TaskEntity),
            ("regions", TaskRegion),
            ("criteria", TaskCriterion),
        ):
            values = getattr(self, field_name)

            if isinstance(values, (str, bytes)):
                raise ValueError(
                    f"{field_name} must be an iterable of valid objects."
                )

            try:
                normalized = tuple(values)
            except TypeError:
                raise ValueError(
                    f"{field_name} must be iterable."
                ) from None

            if not all(
                isinstance(value, expected_type)
                for value in normalized
            ):
                raise ValueError(
                    f"Every item in {field_name} must be a "
                    f"{expected_type.__name__} instance."
                )

            object.__setattr__(self, field_name, normalized)

        unresolved_items = _normalize_strings(
            self.unresolved_items,
            "unresolved_items",
        )
        object.__setattr__(self, "unresolved_items", unresolved_items)

        if not self.entities:
            raise ValueError("entities must not be empty.")

        if not self.criteria:
            raise ValueError("criteria must not be empty.")

        entity_roles = [entity.role for entity in self.entities]

        if len(entity_roles) != len(set(entity_roles)):
            raise ValueError("Entity roles must be unique.")

        region_roles = [region.role for region in self.regions]

        if len(region_roles) != len(set(region_roles)):
            raise ValueError("Region roles must be unique.")

        criterion_ids = [
            criterion.criterion_id for criterion in self.criteria
        ]

        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("Criterion IDs must be unique.")

        available_entity_roles = set(entity_roles)
        available_region_roles = set(region_roles)

        if TaskEntityRole.OBJECT.value not in available_entity_roles:
            raise ValueError(
                "At least one object entity is required."
            )

        for criterion in self.criteria:
            missing_entity_roles = (
                set(criterion.entity_roles) - available_entity_roles
            )

            if missing_entity_roles:
                raise ValueError(
                    f"Criterion '{criterion.criterion_id}' references "
                    f"undefined entity roles: "
                    f"{sorted(missing_entity_roles)}."
                )

            if (
                criterion.region_role is not None
                and criterion.region_role not in available_region_roles
            ):
                raise ValueError(
                    f"Criterion '{criterion.criterion_id}' references "
                    f"undefined region role: {criterion.region_role}."
                )
