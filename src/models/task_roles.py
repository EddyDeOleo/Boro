
"""Generic semantic role configuration for robotic tasks."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


def _validate_identifier(value: str, field_name: str) -> None:
    """Validate a non-empty string identifier."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _freeze_role_mapping(
    mapping: Mapping[str, str],
    field_name: str,
) -> Mapping[str, str]:
    """Validate and freeze a semantic role mapping."""

    if not isinstance(mapping, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")

    copied_mapping = dict(mapping)

    for role, identifier in copied_mapping.items():
        _validate_identifier(role, f"{field_name} role")
        _validate_identifier(identifier, f"{field_name}[{role}]")

    return MappingProxyType(copied_mapping)


@dataclass(frozen=True)
class TaskRoles:
    """Maps semantic task roles to observed entity and region identifiers.

    Entity roles describe the purpose of an entity in a task.
    Region roles describe the purpose of a spatial region.

    The mappings are intentionally generic and do not assume
    a particular robot type or task category.
    """

    entity_roles: Mapping[str, str] = field(default_factory=dict)
    region_roles: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "entity_roles",
            _freeze_role_mapping(
                self.entity_roles,
                "entity_roles",
            ),
        )

        object.__setattr__(
            self,
            "region_roles",
            _freeze_role_mapping(
                self.region_roles,
                "region_roles",
            ),
        )

    def get_entity_id(self, role: str) -> str:
        """Resolve an entity identifier from its semantic role."""

        _validate_identifier(role, "role")

        try:
            return self.entity_roles[role]
        except KeyError:
            raise KeyError(
                f"No entity is configured for role {role!r}."
            ) from None

    def get_region_id(self, role: str) -> str:
        """Resolve a region identifier from its semantic role."""

        _validate_identifier(role, "role")

        try:
            return self.region_roles[role]
        except KeyError:
            raise KeyError(
                f"No region is configured for role {role!r}."
            ) from None

    def has_entity_role(self, role: str) -> bool:
        """Return whether an entity role is configured."""

        return role in self.entity_roles

    def has_region_role(self, role: str) -> bool:
        """Return whether a region role is configured."""

        return role in self.region_roles
