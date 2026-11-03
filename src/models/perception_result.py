
"""Normalized output model for visual perception."""

from dataclasses import dataclass
from typing import TypeVar

from src.models.evidence import (
    EntityObservation,
    Evidence,
    SpatialRelationship,
)
from src.models.observation_context import ObservationContext
from src.models.region import RegionObservation
from src.models.video_metadata import VideoMetadata


T = TypeVar("T")


def _normalize_collection(
    values: object,
    expected_type: type[T],
    field_name: str,
) -> tuple[T, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be an iterable of valid objects.")

    try:
        normalized = tuple(values)
    except TypeError:
        raise ValueError(f"{field_name} must be iterable.") from None

    if any(not isinstance(value, expected_type) for value in normalized):
        raise ValueError(
            f"Every item in {field_name} must be a "
            f"{expected_type.__name__} instance."
        )

    return normalized


@dataclass(frozen=True)
class PerceptionResult:
    """Immutable, provider-independent perception output."""

    context: ObservationContext
    video_metadata: VideoMetadata

    observations: tuple[EntityObservation, ...]
    spatial_relationships: tuple[SpatialRelationship, ...] = ()
    region_observations: tuple[RegionObservation, ...] = ()
    evidence: tuple[Evidence, ...] = ()

    provider: str = ""
    model: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.context, ObservationContext):
            raise ValueError("context must be an ObservationContext.")

        if not isinstance(self.video_metadata, VideoMetadata):
            raise ValueError("video_metadata must be a VideoMetadata.")

        for field_name, expected_type in (
            ("observations", EntityObservation),
            ("spatial_relationships", SpatialRelationship),
            ("region_observations", RegionObservation),
            ("evidence", Evidence),
        ):
            normalized = _normalize_collection(
                getattr(self, field_name),
                expected_type,
                field_name,
            )
            object.__setattr__(self, field_name, normalized)

        # Validate frame boundaries before checking cross-references.
        for observation in self.observations:
            if observation.frame_index >= self.context.frame_count:
                raise ValueError(
                    "Observation frame index exceeds the video context."
                )

        for relationship in self.spatial_relationships:
            if relationship.frame_index >= self.context.frame_count:
                raise ValueError(
                    "Spatial relationship frame index exceeds "
                    "the video context."
                )

        for region_observation in self.region_observations:
            if region_observation.frame_index >= self.context.frame_count:
                raise ValueError(
                    "Region observation frame index exceeds "
                    "the video context."
                )

        for item in self.evidence:
            if item.frame_end >= self.context.frame_count:
                raise ValueError(
                    "Evidence frame range exceeds the video context."
                )

        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string.")

        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string.")

        observation_ids = [
            observation.observation_id
            for observation in self.observations
        ]

        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("Observation IDs must be unique.")

        observed_entities_by_frame = {
            (observation.frame_index, observation.entity_id)
            for observation in self.observations
        }

        for relationship in self.spatial_relationships:
            if (
                relationship.frame_index,
                relationship.entity_a_id,
            ) not in observed_entities_by_frame:
                raise ValueError(
                    "Spatial relationship references an entity "
                    "without an observation in the same frame."
                )

            if (
                relationship.frame_index,
                relationship.entity_b_id,
            ) not in observed_entities_by_frame:
                raise ValueError(
                    "Spatial relationship references an entity "
                    "without an observation in the same frame."
                )

        for region_observation in self.region_observations:
            if (
                region_observation.frame_index,
                region_observation.entity_id,
            ) not in observed_entities_by_frame:
                raise ValueError(
                    "Region observation references an entity "
                    "without an observation in the same frame."
                )
