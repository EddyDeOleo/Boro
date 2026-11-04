
"""HTTP request schema for integrated robotic execution."""

from __future__ import annotations

from pydantic import Field, model_validator

from src.api.schemas.verification_request import (
    APIRequestModel,
    BoundingBoxRequest,
    ObservationRequest,
    PointRequest,
)
from src.models.region import RegionDefinition, RegionType


class RoleRegionRequest(APIRequestModel):
    """Polygonal region associated with a semantic task role."""

    name: str = Field(min_length=1)
    region_type: RegionType

    polygon: list[PointRequest] = Field(min_length=3)

    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    reference_frame: int = Field(default=0, ge=0)

    coordinate_system: str = "pixel"

    @model_validator(mode="after")
    def validate_region(self) -> RoleRegionRequest:
        if self.coordinate_system != "pixel":
            raise ValueError(
                "Only pixel coordinates are currently supported."
            )

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
    
    def to_domain(self, region_id: str) -> RegionDefinition:
        """Convert the request into a domain region."""

        polygon = [
            (point.x, point.y)
            for point in self.polygon
        ]

        # Remove the closing vertex if it duplicates the first vertex.
        if len(polygon) > 1 and polygon[0] == polygon[-1]:
            polygon.pop()

        return RegionDefinition(
            region_id=region_id,
            name=self.name,
            region_type=self.region_type,
            polygon=tuple(polygon),
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            reference_frame=self.reference_frame,
            coordinate_system=self.coordinate_system,
        )

class RoboticExecutionRequest(APIRequestModel):
    """Integrated robotic execution request."""

    task_instruction: str = Field(min_length=1)

    tracking_boxes: dict[str, BoundingBoxRequest] = Field(
        min_length=1
    )

    regions: dict[str, RoleRegionRequest] = Field(
        default_factory=dict
    )

    observation: ObservationRequest = Field(
        default_factory=ObservationRequest
    )

    @model_validator(mode="after")
    def validate_role_mappings(self) -> RoboticExecutionRequest:
        for role in self.tracking_boxes:
            if not role or not role.strip():
                raise ValueError(
                    "Tracking box role names cannot be empty."
                )

            if role != role.strip():
                raise ValueError(
                    "Tracking box role names cannot contain "
                    "surrounding whitespace."
                )

        for role in self.regions:
            if not role or not role.strip():
                raise ValueError(
                    "Region role names cannot be empty."
                )

            if role != role.strip():
                raise ValueError(
                    "Region role names cannot contain "
                    "surrounding whitespace."
                )

        for role, box in self.tracking_boxes.items():
            if box.x2 <= box.x1:
                raise ValueError(
                    f"Tracking box for role {role!r} must have "
                    "a positive width."
                )

            if box.y2 <= box.y1:
                raise ValueError(
                    f"Tracking box for role {role!r} must have "
                    "a positive height."
                )

        return self

    def to_domain(self) -> dict:
        """Convert the HTTP request into orchestration inputs."""

        return {
            "task_instruction": self.task_instruction,
            "tracking_boxes_by_role": {
                role: box.to_domain()
                for role, box in self.tracking_boxes.items()
            },
            "regions_by_role": {
                role: region.to_domain(
                    region_id=f"pending-{role}"
                )
                for role, region in self.regions.items()
            },
            "coverage_scope": self.observation.coverage_scope,
            "frame_start": self.observation.frame_start,
            "frame_end": self.observation.frame_end,
            "time_offset_seconds": (
                self.observation.time_offset_seconds
            ),
        }
