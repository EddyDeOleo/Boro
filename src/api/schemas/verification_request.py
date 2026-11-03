
"""HTTP request schemas for robotic execution verification."""

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.models.observation_context import CoverageScope
from src.models.perception_config import PerceptionConfig
from src.models.region import RegionDefinition, RegionType
from src.models.task_roles import TaskRoles
from src.models.verification_criteria import (
    CriterionRequirement,
    VerificationCriteria,
)


class APIRequestModel(BaseModel):
    """Base configuration for API request models."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class PerceptionConfigRequest(APIRequestModel):
    """Provider-independent perception configuration."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    entity_id_map: dict[str, str] = Field(min_length=1)

    include_spatial_relationships: bool = True
    include_region_observations: bool = True

    @field_validator("entity_id_map")
    @classmethod
    def validate_entity_id_map(
        cls,
        value: dict[str, str],
    ) -> dict[str, str]:
        if any(not key.strip() for key in value):
            raise ValueError("Provider entity IDs cannot be empty.")

        if any(not entity_id.strip() for entity_id in value.values()):
            raise ValueError("Semantic entity IDs cannot be empty.")

        if len(set(value.values())) != len(value):
            raise ValueError("Semantic entity IDs must be unique.")

        return value

    def to_domain(self) -> PerceptionConfig:
        """Convert the request schema into a domain model."""

        return PerceptionConfig(
            provider=self.provider,
            model=self.model,
            entity_id_map=self.entity_id_map,
            include_spatial_relationships=self.include_spatial_relationships,
            include_region_observations=self.include_region_observations,
        )


class TaskRolesRequest(APIRequestModel):
    """Semantic entity and region roles for a task."""

    entity_roles: dict[str, str] = Field(default_factory=dict)
    region_roles: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_role_mappings(self) -> TaskRolesRequest:
        for mapping_name, mapping in (
            ("entity_roles", self.entity_roles),
            ("region_roles", self.region_roles),
        ):
            for role, identifier in mapping.items():
                if not role.strip():
                    raise ValueError(
                        f"{mapping_name} role names cannot be empty."
                    )

                if not identifier.strip():
                    raise ValueError(
                        f"{mapping_name} identifiers cannot be empty."
                    )

        return self

    def to_domain(self) -> TaskRoles:
        """Convert the request schema into a domain model."""

        return TaskRoles(
            entity_roles=self.entity_roles,
            region_roles=self.region_roles,
        )


class VerificationCriterionRequest(APIRequestModel):
    """One declarative verification criterion."""

    criterion_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    event_type: str = Field(min_length=1)

    requirement: CriterionRequirement

    entity_roles: list[str] = Field(default_factory=list)
    region_role: str | None = None

    required_evidence_types: list[str] = Field(default_factory=list)
    is_mandatory: bool = True

    @field_validator("entity_roles", "required_evidence_types")
    @classmethod
    def validate_unique_non_empty_values(
        cls,
        values: list[str],
    ) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("Values cannot contain empty strings.")

        if len(set(values)) != len(values):
            raise ValueError("Duplicate values are not allowed.")

        return values

    @field_validator("region_role")
    @classmethod
    def validate_region_role(
        cls,
        value: str | None,
    ) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("region_role cannot be empty.")

        return value

    def to_domain(self) -> VerificationCriteria:
        """Convert the request schema into a domain model."""

        return VerificationCriteria(
            criterion_id=self.criterion_id,
            description=self.description,
            event_type=self.event_type,
            requirement=self.requirement,
            entity_roles=tuple(self.entity_roles),
            region_role=self.region_role,
            required_evidence_types=tuple(
                self.required_evidence_types
            ),
            is_mandatory=self.is_mandatory,
        )


class PointRequest(APIRequestModel):
    """A two-dimensional point in image coordinates."""

    x: float
    y: float


class BoundingBoxRequest(APIRequestModel):
    """An initial entity bounding box in image coordinates."""

    x1: int = Field(ge=0, strict=True)
    y1: int = Field(ge=0, strict=True)
    x2: int = Field(ge=0, strict=True)
    y2: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_coordinates(self) -> BoundingBoxRequest:
        if self.x2 < self.x1:
            raise ValueError(
                "Bounding box x2 must be greater than or equal to x1."
            )

        if self.y2 < self.y1:
            raise ValueError(
                "Bounding box y2 must be greater than or equal to y1."
            )

        return self

    def to_domain(self) -> dict[str, int]:
        """Convert the bounding box into primitive coordinates."""

        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


class RegionRequest(APIRequestModel):
    """Polygonal region definition in image coordinates."""

    region_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    region_type: RegionType

    polygon: list[PointRequest] = Field(min_length=3)

    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    reference_frame: int = Field(default=0, ge=0)

    coordinate_system: str = "pixel"

    @field_validator("coordinate_system")
    @classmethod
    def validate_coordinate_system(cls, value: str) -> str:
        if value != "pixel":
            raise ValueError(
                "Only pixel coordinates are currently supported."
            )

        return value

    @model_validator(mode="after")
    def validate_polygon_bounds(self) -> RegionRequest:
        for point in self.polygon:
            if not 0 <= point.x < self.frame_width:
                raise ValueError(
                    "Polygon point exceeds frame width."
                )

            if not 0 <= point.y < self.frame_height:
                raise ValueError(
                    "Polygon point exceeds frame height."
                )

        return self

    def to_domain(self) -> RegionDefinition:
        """Convert the request schema into a domain model."""

        return RegionDefinition(
            region_id=self.region_id,
            name=self.name,
            region_type=self.region_type,
            polygon=tuple(
                (point.x, point.y)
                for point in self.polygon
            ),
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            reference_frame=self.reference_frame,
            coordinate_system=self.coordinate_system,
        )


class ObservationRequest(APIRequestModel):
    """Video observation and coverage configuration."""

    coverage_scope: CoverageScope = CoverageScope.UNKNOWN

    frame_start: int = Field(default=0, ge=0)
    frame_end: int | None = Field(default=None, ge=0)

    time_offset_seconds: float = Field(default=0.0, ge=0)


class VerificationRequest(APIRequestModel):
    """Complete verification configuration submitted with a video."""

    perception: PerceptionConfigRequest
    task_roles: TaskRolesRequest

    tracking_boxes: dict[str, BoundingBoxRequest] = Field(
        min_length=1
    )

    criteria: list[VerificationCriterionRequest] = Field(
        min_length=1
    )

    regions: list[RegionRequest] = Field(default_factory=list)

    observation: ObservationRequest = Field(
        default_factory=ObservationRequest
    )

    @model_validator(mode="after")
    def validate_configuration_consistency(
        self,
    ) -> VerificationRequest:
        criterion_ids = [
            criterion.criterion_id
            for criterion in self.criteria
        ]

        if len(set(criterion_ids)) != len(criterion_ids):
            raise ValueError("Criterion IDs must be unique.")

        region_ids = [
            region.region_id
            for region in self.regions
        ]

        if len(set(region_ids)) != len(region_ids):
            raise ValueError("Region IDs must be unique.")

        configured_entities = set(
            self.perception.entity_id_map.values()
        )

        configured_provider_ids = set(
            self.perception.entity_id_map.keys()
        )

        tracking_provider_ids = set(self.tracking_boxes.keys())

        if tracking_provider_ids != configured_provider_ids:
            raise ValueError(
                "Tracking box IDs must match the provider entity IDs "
                "configured in perception.entity_id_map."
            )

        configured_regions = set(region_ids)

        for role, entity_id in self.task_roles.entity_roles.items():
            if entity_id not in configured_entities:
                raise ValueError(
                    f"Entity role {role!r} references an entity "
                    f"that is not configured in perception.entity_id_map."
                )

        for role, region_id in self.task_roles.region_roles.items():
            if region_id not in configured_regions:
                raise ValueError(
                    f"Region role {role!r} references an undefined region."
                )

        for criterion in self.criteria:
            for role in criterion.entity_roles:
                if role not in self.task_roles.entity_roles:
                    raise ValueError(
                        f"Criterion {criterion.criterion_id!r} references "
                        f"an undefined entity role {role!r}."
                    )

            if (
                criterion.region_role is not None
                and criterion.region_role
                not in self.task_roles.region_roles
            ):
                raise ValueError(
                    f"Criterion {criterion.criterion_id!r} references "
                    f"an undefined region role "
                    f"{criterion.region_role!r}."
                )

        return self

    def to_domain(self) -> dict[str, Any]:
        """Convert the HTTP request into validated domain objects."""

        return {
            "perception_config": self.perception.to_domain(),
            "task_roles": self.task_roles.to_domain(),
            "tracking_boxes": {
                provider_id: box.to_domain()
                for provider_id, box in self.tracking_boxes.items()
            },
            "criteria": tuple(
                criterion.to_domain()
                for criterion in self.criteria
            ),
            "regions": tuple(
                region.to_domain()
                for region in self.regions
            ),
            "coverage_scope": self.observation.coverage_scope,
            "frame_start": self.observation.frame_start,
            "frame_end": self.observation.frame_end,
            "time_offset_seconds": (
                self.observation.time_offset_seconds
            ),
        }
