
"""Generic configuration for a provider-independent perception execution."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


def _validate_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


@dataclass(frozen=True)
class PerceptionConfig:
    """Configuration for a provider-independent perception execution."""

    provider: str
    model: str
    entity_id_map: Mapping[str, str]

    include_spatial_relationships: bool = True
    include_region_observations: bool = True

    def __post_init__(self) -> None:
        _validate_non_empty(self.provider, "provider")
        _validate_non_empty(self.model, "model")

        if not isinstance(self.entity_id_map, Mapping):
            raise ValueError("entity_id_map must be a mapping.")

        normalized_map = dict(self.entity_id_map)

        if not normalized_map:
            raise ValueError("At least one entity must be configured.")

        for provider_id, entity_id in normalized_map.items():
            _validate_non_empty(provider_id, "provider entity ID")
            _validate_non_empty(entity_id, "semantic entity ID")

        if len(set(normalized_map.values())) != len(normalized_map):
            raise ValueError("Semantic entity IDs cannot contain duplicates.")

        if not isinstance(self.include_spatial_relationships, bool):
            raise ValueError(
                "include_spatial_relationships must be a boolean."
            )

        if not isinstance(self.include_region_observations, bool):
            raise ValueError(
                "include_region_observations must be a boolean."
            )

        object.__setattr__(
            self,
            "entity_id_map",
            MappingProxyType(normalized_map),
        )

    @property
    def entity_ids(self) -> tuple[str, ...]:
        """Return the configured semantic entity IDs."""
        return tuple(self.entity_id_map.values())
