
"""Convert SAM 3.1 perception outputs into Boro observation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from src.models.evidence import EntityObservation, SpatialRelationship
from src.models.region import RegionDefinition, RegionObservation, RegionRelation
from src.perception.mask_analyzer import (
    MaskMetrics,
    MaskRelationshipMetrics,
    RelativeMetrics,
)


@dataclass(frozen=True)
class SAMObservationBundle:
    """Collection of normalized observations produced by the SAM adapter."""

    entity_observations: tuple[EntityObservation, ...]
    spatial_relationships: tuple[SpatialRelationship, ...]
    region_observations: tuple[RegionObservation, ...]


def _validate_fps(fps: float) -> None:
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("fps must be a finite positive number.")


def _validate_time_offset(offset: float) -> None:
    if (
        isinstance(offset, bool)
        or not isinstance(offset, (int, float))
        or not isfinite(offset)
    ):
        raise ValueError("time_offset_seconds must be finite.")


def _validate_source(source: str) -> None:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string.")


def _validate_object_id_map(
    object_id_map: Mapping[str, str],
) -> None:
    if not isinstance(object_id_map, Mapping) or not object_id_map:
        raise ValueError("object_id_map must be a non-empty mapping.")

    for sam_id, entity_id in object_id_map.items():
        if not isinstance(sam_id, str) or not sam_id.strip():
            raise ValueError("SAM object IDs must be non-empty strings.")

        if not isinstance(entity_id, str) or not entity_id.strip():
            raise ValueError("Semantic entity IDs must be non-empty strings.")

    semantic_ids = list(object_id_map.values())

    if len(semantic_ids) != len(set(semantic_ids)):
        raise ValueError("Semantic entity IDs must be unique.")


def build_entity_observations(
    object_metrics: Sequence[MaskMetrics],
    object_id_map: Mapping[str, str],
    confidences: Mapping[tuple[int, str], float | None],
    fps: float,
    time_offset_seconds: float = 0.0,
    source: str = "sam-3.1",
) -> tuple[EntityObservation, ...]:
    """Convert per-frame mask metrics into entity observations.

    Bounding boxes use the half-open convention:
    (x_min, y_min, x_max, y_max).

    Timestamps are relative to the clip unless an offset is supplied.
    """

    _validate_fps(fps)
    _validate_time_offset(time_offset_seconds)
    _validate_object_id_map(object_id_map)
    _validate_source(source)

    observations: list[EntityObservation] = []
    seen_keys: set[tuple[int, str]] = set()

    for metrics in object_metrics:
        key = (metrics.frame, metrics.object_id)

        if key in seen_keys:
            raise ValueError(
                f"Duplicate object metrics for frame {metrics.frame}, "
                f"SAM object {metrics.object_id}."
            )

        seen_keys.add(key)

        if metrics.object_id not in object_id_map:
            raise ValueError(
                f"SAM object ID {metrics.object_id!r} "
                "has no semantic mapping."
            )

        entity_id = object_id_map[metrics.object_id]
        timestamp = metrics.frame / fps + time_offset_seconds

        attributes = {
            "sam_object_id": metrics.object_id,
            "temporal_iou_previous": metrics.iou_previous,
            "area_ratio": metrics.area_ratio,
            "bbox_coordinate_convention": "half_open",
            "centroid_coordinate_system": "image_pixels",
        }

        observations.append(
            EntityObservation(
                observation_id=f"sam:{entity_id}:{metrics.frame}",
                entity_id=entity_id,
                frame_index=metrics.frame,
                timestamp_seconds=timestamp,
                bbox=metrics.bbox,
                centroid=(metrics.centroid_x, metrics.centroid_y),
                area_pixels=float(metrics.area),
                confidence=confidences.get(key),
                attributes=attributes,
                source=source,
            )
        )

    return tuple(observations)


def build_spatial_relationships(
    relationship_metrics: Sequence[MaskRelationshipMetrics],
    relative_metrics: Sequence[RelativeMetrics],
    object_id_map: Mapping[str, str],
    relative_entity_ids: tuple[str, str],
    source: str = "sam-3.1",
) -> tuple[SpatialRelationship, ...]:
    """Combine mask geometry and centroid displacement into relationships.

    relative_entity_ids identifies the ordered SAM IDs used when calculating
    RelativeMetrics: delta = centroid(A) - centroid(B).
    """

    _validate_object_id_map(object_id_map)
    _validate_source(source)

    if (
        not isinstance(relative_entity_ids, tuple)
        or len(relative_entity_ids) != 2
    ):
        raise ValueError(
            "relative_entity_ids must be a tuple containing exactly two IDs."
        )

    for object_id in relative_entity_ids:
        if object_id not in object_id_map:
            raise ValueError(
                f"Unknown SAM object ID in relative_entity_ids: {object_id!r}."
            )

    relative_by_frame: dict[int, RelativeMetrics] = {}

    for metric in relative_metrics:
        if metric.frame in relative_by_frame:
            raise ValueError(
                f"Duplicate relative metrics for frame {metric.frame}."
            )

        relative_by_frame[metric.frame] = metric

    relationships: list[SpatialRelationship] = []
    seen_frames: set[int] = set()

    for metric in relationship_metrics:
        frame = metric.frame

        if frame in seen_frames:
            raise ValueError(
                f"Duplicate relationship metrics for frame {frame}."
            )

        seen_frames.add(frame)

        if metric.object_a_id not in object_id_map:
            raise ValueError(
                f"Unknown SAM object ID: {metric.object_a_id!r}."
            )

        if metric.object_b_id not in object_id_map:
            raise ValueError(
                f"Unknown SAM object ID: {metric.object_b_id!r}."
            )

        expected_pair = (
            metric.object_a_id,
            metric.object_b_id,
        )

        if expected_pair != relative_entity_ids:
            raise ValueError(
                "Relative metric orientation does not match the "
                "relationship object order."
            )

        relative = relative_by_frame.get(frame)

        if relative is None:
            raise ValueError(
                f"Missing relative metrics for frame {frame}."
            )

        entity_a_id = object_id_map[metric.object_a_id]
        entity_b_id = object_id_map[metric.object_b_id]

        relationships.append(
            SpatialRelationship(
                frame_index=frame,
                entity_a_id=entity_a_id,
                entity_b_id=entity_b_id,
                distance_pixels=metric.min_distance,
                delta_x_pixels=relative.delta_x,
                delta_y_pixels=relative.delta_y,
                intersection_area_pixels=metric.intersection_area,
                iou=metric.iou,
                overlap_ratio_a=metric.overlap_a,
                overlap_ratio_b=metric.overlap_b,
                attributes={
                    "centroid_distance_pixels": relative.distance,
                    "normalized_delta_x": metric.normalized_delta_x,
                    "normalized_delta_y": metric.normalized_delta_y,
                    "distance_metric": (
                        "minimum_euclidean_distance_between_pixel_centers"
                    ),
                    "physical_contact_established": False,
                    "source": source,
                },
            )
        )

    return tuple(relationships)


def build_region_observations(
    masks_by_frame: Mapping[int, Mapping[str, np.ndarray]],
    object_id_map: Mapping[str, str],
    region: RegionDefinition,
    sam_object_id: str,
) -> tuple[RegionObservation, ...]:
    """Evaluate how much of an entity mask lies inside a defined region.

    Each foreground pixel is tested using RegionDefinition.contains_point().
    Pixel coordinates follow the image-coordinate convention used by
    the mask analyzer.
    """

    _validate_object_id_map(object_id_map)

    if sam_object_id not in object_id_map:
        raise ValueError(
            f"SAM object ID {sam_object_id!r} has no semantic mapping."
        )

    entity_id = object_id_map[sam_object_id]
    observations: list[RegionObservation] = []

    for frame_index in sorted(masks_by_frame):
        frame_masks = masks_by_frame[frame_index]

        if sam_object_id not in frame_masks:
            raise ValueError(
                f"Missing mask for SAM object {sam_object_id!r} "
                f"in frame {frame_index}."
            )

        mask = frame_masks[sam_object_id]

        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise ValueError(
                f"Mask in frame {frame_index} must be a 2D numpy array."
            )

        expected_shape = (region.frame_height, region.frame_width)

        if mask.shape != expected_shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match "
                f"region dimensions {expected_shape}."
            )

        foreground_y, foreground_x = np.where(mask > 0)
        area = len(foreground_x)

        if area == 0:
            raise ValueError(
                f"Empty mask for SAM object {sam_object_id!r} "
                f"in frame {frame_index}."
            )

        inside_count = sum(
            region.contains_point((float(x), float(y)))
            for x, y in zip(foreground_x, foreground_y)
        )

        overlap_ratio = inside_count / area

        centroid_x = float(np.mean(foreground_x))
        centroid_y = float(np.mean(foreground_y))

        centroid_inside = region.contains_point(
            (centroid_x, centroid_y)
        )

        if inside_count == area:
            relation = RegionRelation.INSIDE
        elif inside_count == 0:
            relation = RegionRelation.OUTSIDE
        else:
            relation = RegionRelation.PARTIALLY_INTERSECTING

        observations.append(
            RegionObservation(
                entity_id=entity_id,
                region_id=region.region_id,
                frame_index=frame_index,
                relation=relation,
                centroid_inside=centroid_inside,
                intersection_area_pixels=inside_count,
                overlap_ratio=overlap_ratio,
            )
        )

    return tuple(observations)


def adapt_sam_observations(
    object_metrics: Sequence[MaskMetrics],
    relative_metrics: Sequence[RelativeMetrics],
    relationship_metrics: Sequence[MaskRelationshipMetrics],
    masks_by_frame: Mapping[int, Mapping[str, np.ndarray]],
    object_id_map: Mapping[str, str],
    confidences: Mapping[tuple[int, str], float | None],
    fps: float,
    relative_entity_ids: tuple[str, str],
    region: RegionDefinition | None = None,
    region_sam_object_id: str | None = None,
    time_offset_seconds: float = 0.0,
    source: str = "sam-3.1",
) -> SAMObservationBundle:
    """Build all Boro observation contracts from an analyzed SAM episode."""

    _validate_fps(fps)
    _validate_time_offset(time_offset_seconds)
    _validate_object_id_map(object_id_map)
    _validate_source(source)

    entities = build_entity_observations(
        object_metrics=object_metrics,
        object_id_map=object_id_map,
        confidences=confidences,
        fps=fps,
        time_offset_seconds=time_offset_seconds,
        source=source,
    )

    relationships = build_spatial_relationships(
        relationship_metrics=relationship_metrics,
        relative_metrics=relative_metrics,
        object_id_map=object_id_map,
        relative_entity_ids=relative_entity_ids,
        source=source,
    )

    regions: tuple[RegionObservation, ...] = ()

    if region is not None:
        if region_sam_object_id is None:
            raise ValueError(
                "region_sam_object_id is required when a region is supplied."
            )

        regions = build_region_observations(
            masks_by_frame=masks_by_frame,
            object_id_map=object_id_map,
            region=region,
            sam_object_id=region_sam_object_id,
        )
    elif region_sam_object_id is not None:
        raise ValueError(
            "region must be supplied when region_sam_object_id is provided."
        )

    return SAMObservationBundle(
        entity_observations=entities,
        spatial_relationships=relationships,
        region_observations=regions,
    )
