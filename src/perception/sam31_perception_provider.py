
"""SAM 3.1 implementation of the provider-independent perception contract."""

from __future__ import annotations

import logging
from collections import Counter
from itertools import combinations
from math import isclose, isfinite
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from meta_sam_parser import VideoSegmentationResult

from src.adapters.sam_observation_adapter import (
    build_entity_observations,
    build_region_observations,
    build_spatial_relationships,
)
from src.models.observation_context import ObservationContext
from src.models.perception_config import PerceptionConfig
from src.models.perception_result import PerceptionResult
from src.models.region import RegionDefinition
from src.models.video_metadata import VideoMetadata
from src.perception.mask_analyzer import (
    MaskMetrics,
    calculate_mask_metrics,
    calculate_mask_relationship_metrics,
    calculate_relative_metrics,
)
from src.perception.perception_provider import PerceptionProvider
from src.perception.sam_mask_reconstructor import reconstruct_frame_mask
from src.perception.video_processor import VideoProcessor


logger = logging.getLogger(__name__)


class SAM31PerceptionError(RuntimeError):
    """Raised when SAM 3.1 output cannot be normalized into Boro observations."""


@runtime_checkable
class SAM31Inference(Protocol):
    """Contract for an injected SAM 3.1 inference implementation."""

    def infer(
        self,
        video_path: Path,
        context: ObservationContext,
        config: PerceptionConfig,
        regions: Sequence[RegionDefinition],
    ) -> VideoSegmentationResult:
        """Run SAM 3.1 inference and return its parsed video result."""
        ...


class Sam31PerceptionProvider(PerceptionProvider):
    """Normalize injected SAM 3.1 results into Boro perception contracts.

    This class deliberately does not upload videos, create API clients, or
    perform network requests. Those responsibilities belong to the injected
    inference implementation.
    """

    def __init__(self, inference: SAM31Inference) -> None:
        if not isinstance(inference, SAM31Inference):
            raise TypeError("inference must implement the SAM31Inference protocol.")

        self._inference = inference

    def perceive(
        self,
        video_path: Path,
        context: ObservationContext,
        config: PerceptionConfig,
        regions: Sequence[RegionDefinition] = (),
    ) -> PerceptionResult:
        """Analyze a video and return normalized perception observations."""

        if not isinstance(video_path, (str, Path)):
            raise TypeError("video_path must be a string or Path.")

        source_path = Path(video_path)

        if not source_path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {source_path}")

        if not isinstance(context, ObservationContext):
            raise TypeError("context must be an ObservationContext.")

        if not isinstance(config, PerceptionConfig):
            raise TypeError("config must be a PerceptionConfig.")

        normalized_regions = self._normalize_regions(regions)

        with VideoProcessor(source_path) as processor:
            metadata = processor.metadata

        self._validate_video_context(metadata, context)
        self._validate_regions(normalized_regions, metadata)

        try:
            segmentation_result = self._inference.infer(
                video_path=source_path.resolve(),
                context=context,
                config=config,
                regions=normalized_regions,
            )
        except Exception as error:
            raise SAM31PerceptionError(
                "The injected SAM 3.1 inference implementation failed."
            ) from error

        self._validate_segmentation_result(segmentation_result)

        masks_by_frame, confidences = self._reconstruct_masks(
            records=segmentation_result.records,
            metadata=metadata,
            context=context,
            config=config,
        )

        if not masks_by_frame:
            raise SAM31PerceptionError(
                "SAM 3.1 returned no usable masks in the requested frame range."
            )

        object_metrics = self._calculate_object_metrics(
            masks_by_frame=masks_by_frame,
        )

        entity_observations = build_entity_observations(
            object_metrics=object_metrics,
            object_id_map=config.entity_id_map,
            confidences=confidences,
            fps=context.fps,
            time_offset_seconds=context.time_offset_seconds,
            source=config.model,
        )

        spatial_relationships = ()

        if config.include_spatial_relationships:
            spatial_relationships = self._calculate_spatial_relationships(
                masks_by_frame=masks_by_frame,
                object_metrics=object_metrics,
                object_id_map=config.entity_id_map,
                source=config.model,
            )

        region_observations = ()

        if config.include_region_observations and normalized_regions:
            region_observations = self._calculate_region_observations(
                masks_by_frame=masks_by_frame,
                object_id_map=config.entity_id_map,
                regions=normalized_regions,
            )

        self._log_perception_diagnostics(
            masks_by_frame=masks_by_frame,
            entity_id_map=config.entity_id_map,
            spatial_relationships=spatial_relationships,
            include_spatial_relationships=config.include_spatial_relationships,
        )

        try:
            return PerceptionResult(
                context=context,
                video_metadata=metadata,
                observations=entity_observations,
                spatial_relationships=spatial_relationships,
                region_observations=region_observations,
                provider=config.provider,
                model=config.model,
            )
        except (TypeError, ValueError) as error:
            raise SAM31PerceptionError(
                "Normalized SAM observations violate the Boro perception contract."
            ) from error

    @staticmethod
    def _normalize_regions(
        regions: Sequence[RegionDefinition],
    ) -> tuple[RegionDefinition, ...]:
        if isinstance(regions, (str, bytes)):
            raise TypeError("regions must be a sequence of RegionDefinition objects.")

        try:
            normalized = tuple(regions)
        except TypeError:
            raise TypeError(
                "regions must be an iterable of RegionDefinition objects."
            ) from None

        if any(not isinstance(region, RegionDefinition) for region in normalized):
            raise TypeError(
                "Every item in regions must be a RegionDefinition."
            )

        region_ids = [region.region_id for region in normalized]

        if len(region_ids) != len(set(region_ids)):
            raise ValueError("Region IDs must be unique.")

        return normalized

    @staticmethod
    def _validate_video_context(
        metadata: VideoMetadata,
        context: ObservationContext,
    ) -> None:
        if context.frame_count != metadata.frame_count:
            raise ValueError(
                "ObservationContext frame_count does not match video metadata."
            )

        if not isclose(
            context.fps,
            metadata.fps,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "ObservationContext fps does not match video metadata."
            )

        if context.frame_end >= metadata.frame_count:
            raise ValueError(
                "ObservationContext frame range exceeds the video."
            )

    @staticmethod
    def _validate_regions(
        regions: Sequence[RegionDefinition],
        metadata: VideoMetadata,
    ) -> None:
        for region in regions:
            if (
                region.frame_width != metadata.width
                or region.frame_height != metadata.height
            ):
                raise ValueError(
                    f"Region {region.region_id!r} dimensions do not match the video."
                )

            if region.reference_frame >= metadata.frame_count:
                raise ValueError(
                    f"Region {region.region_id!r} reference frame exceeds the video."
                )

    @staticmethod
    def _validate_segmentation_result(
        result: VideoSegmentationResult,
    ) -> None:
        if not isinstance(result, VideoSegmentationResult):
            raise TypeError(
                "Inference must return a VideoSegmentationResult."
            )

        outcome = getattr(result, "outcome", None)

        if outcome is None or type(outcome).__name__ != "CompletedOutcome":
            raise SAM31PerceptionError(
                "SAM 3.1 returned an incomplete segmentation result."
            )

        if not isinstance(result.records, tuple):
            raise SAM31PerceptionError(
                "SAM 3.1 result records must be a tuple."
            )

    @staticmethod
    def _reconstruct_masks(
        records: Sequence[object],
        metadata: VideoMetadata,
        context: ObservationContext,
        config: PerceptionConfig,
    ) -> tuple[
        dict[int, dict[str, NDArray[np.uint8]]],
        dict[tuple[int, str], float | None],
    ]:
        masks_by_frame: dict[int, dict[str, NDArray[np.uint8]]] = {}
        confidences: dict[tuple[int, str], float | None] = {}

        configured_provider_ids = set(config.entity_id_map)

        for record in records:
            if getattr(record, "kind", None) != "mask":
                continue

            identity = getattr(record, "identity", None)

            if identity is None:
                raise SAM31PerceptionError(
                    "A SAM mask record is missing its identity."
                )

            if getattr(identity, "media", None) != "video":
                raise SAM31PerceptionError(
                    "A SAM mask record does not identify video media."
                )

            object_id = getattr(record, "object_id", None)
            identity_object_id = getattr(identity, "object_id", None)

            if object_id != identity_object_id:
                raise SAM31PerceptionError(
                    "SAM mask object ID does not match its identity."
                )

            if object_id not in configured_provider_ids:
                raise SAM31PerceptionError(
                    f"SAM returned an unconfigured object ID: {object_id!r}."
                )

            frame_index = getattr(identity, "frame_index", None)

            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or frame_index < 0
            ):
                raise SAM31PerceptionError(
                    "SAM mask identity must contain a valid frame index."
                )

            frame_reference = getattr(record, "frame", None)

            if frame_reference is not None:
                reference_index = getattr(frame_reference, "frame_index", None)

                if reference_index != frame_index:
                    raise SAM31PerceptionError(
                        "SAM mask frame reference conflicts with its identity."
                    )

            if not context.frame_start <= frame_index <= context.frame_end:
                raise SAM31PerceptionError(
                    f"SAM returned frame {frame_index} outside the requested "
                    f"range {context.frame_start}-{context.frame_end}."
                )

            if frame_index >= metadata.frame_count:
                raise SAM31PerceptionError(
                    f"SAM returned frame {frame_index} outside the video."
                )

            frame_masks = masks_by_frame.setdefault(frame_index, {})

            if object_id in frame_masks:
                raise SAM31PerceptionError(
                    f"Duplicate SAM mask for object {object_id!r} "
                    f"in frame {frame_index}."
                )

            confidence = getattr(record, "confidence", None)

            if confidence is not None:
                if (
                    isinstance(confidence, bool)
                    or not isinstance(confidence, (int, float))
                    or not isfinite(confidence)
                    or not 0.0 <= confidence <= 1.0
                ):
                    raise SAM31PerceptionError(
                        "SAM mask confidence must be None or a finite "
                        "number between 0 and 1."
                    )

            try:
                raster = reconstruct_frame_mask(
                    record=record,
                    frame_width=metadata.width,
                    frame_height=metadata.height,
                )
            except Exception as error:
                raise SAM31PerceptionError(
                    f"Could not reconstruct SAM mask for object {object_id!r} "
                    f"in frame {frame_index}."
                ) from error

            if (
                not isinstance(raster, np.ndarray)
                or raster.ndim != 2
                or raster.shape != (metadata.height, metadata.width)
            ):
                raise SAM31PerceptionError(
                    "Reconstructed SAM mask does not match video dimensions."
                )

            if not np.isin(raster, (0, 1)).all():
                raise SAM31PerceptionError(
                    "Reconstructed SAM mask must contain only binary values."
                )

            if not np.any(raster):
                raise SAM31PerceptionError(
                    f"SAM returned an empty mask for object {object_id!r} "
                    f"in frame {frame_index}."
                )

            frame_masks[object_id] = raster
            confidences[(frame_index, object_id)] = (
                float(confidence) if confidence is not None else None
            )

        return masks_by_frame, confidences

    @staticmethod
    def _log_perception_diagnostics(
        masks_by_frame: dict[int, dict[str, NDArray[np.uint8]]],
        entity_id_map: dict[str, str] | object,
        spatial_relationships: Sequence[object],
        include_spatial_relationships: bool,
    ) -> None:
        """Log tracking coverage and spatial relationship diagnostics."""

        configured_ids = sorted(entity_id_map)

        mask_counts = Counter(
            object_id
            for frame_masks in masks_by_frame.values()
            for object_id in frame_masks
        )

        simultaneous_counts = {
            f"{object_a}:{object_b}": 0
            for object_a, object_b in combinations(configured_ids, 2)
        }

        for frame_masks in masks_by_frame.values():
            present_ids = set(frame_masks)

            for object_a, object_b in combinations(configured_ids, 2):
                if object_a in present_ids and object_b in present_ids:
                    simultaneous_counts[f"{object_a}:{object_b}"] += 1

        semantic_mapping = {
            provider_id: entity_id_map[provider_id]
            for provider_id in configured_ids
        }

        logger.warning(
            "SAM 3.1 perception diagnostics | "
            "semantic_mapping=%s | "
            "mask_counts=%s | "
            "frames_with_masks=%d | "
            "simultaneous_pair_frames=%s | "
            "spatial_relationships=%d | "
            "spatial_relationships_enabled=%s",
            semantic_mapping,
            dict(mask_counts),
            len(masks_by_frame),
            simultaneous_counts,
            len(spatial_relationships),
            include_spatial_relationships,
        )

        missing_ids = [
            provider_id
            for provider_id in configured_ids
            if mask_counts[provider_id] == 0
        ]

        if missing_ids:
            logger.warning(
                "SAM 3.1 returned no masks for configured object IDs: %s",
                missing_ids,
            )

        if include_spatial_relationships and not spatial_relationships:
            logger.warning(
                "SAM 3.1 produced no spatial relationships."
            )

    @staticmethod
    def _calculate_object_metrics(
        masks_by_frame: dict[int, dict[str, NDArray[np.uint8]]],
    ) -> tuple[MaskMetrics, ...]:
        metrics: list[MaskMetrics] = []
        previous_masks: dict[str, tuple[int, NDArray[np.uint8]]] = {}

        for frame_index in sorted(masks_by_frame):
            frame_masks = masks_by_frame[frame_index]

            for object_id in sorted(frame_masks):
                current_mask = frame_masks[object_id]
                previous_entry = previous_masks.get(object_id)

                previous_mask = None

                if previous_entry is not None:
                    previous_frame, previous_raster = previous_entry

                    if previous_frame == frame_index - 1:
                        previous_mask = previous_raster

                object_metric = calculate_mask_metrics(
                    mask=current_mask,
                    frame=frame_index,
                    object_id=object_id,
                    previous_mask=previous_mask,
                )

                metrics.append(object_metric)
                previous_masks[object_id] = (frame_index, current_mask)

        return tuple(metrics)

    @staticmethod
    def _calculate_spatial_relationships(
        masks_by_frame: dict[int, dict[str, NDArray[np.uint8]]],
        object_metrics: Sequence[MaskMetrics],
        object_id_map: dict[str, str] | object,
        source: str,
    ) -> tuple:
        metrics_by_frame: dict[int, dict[str, MaskMetrics]] = {}

        for metric in object_metrics:
            metrics_by_frame.setdefault(metric.frame, {})[
                metric.object_id
            ] = metric

        relationship_results = []

        for frame_index in sorted(masks_by_frame):
            frame_masks = masks_by_frame[frame_index]
            object_ids = sorted(frame_masks)

            for object_a_id, object_b_id in combinations(object_ids, 2):
                mask_a = frame_masks[object_a_id]
                mask_b = frame_masks[object_b_id]

                metric_a = metrics_by_frame[frame_index][object_a_id]
                metric_b = metrics_by_frame[frame_index][object_b_id]

                relative_metric = calculate_relative_metrics(
                    metric_a,
                    metric_b,
                )

                relationship_metric = calculate_mask_relationship_metrics(
                    mask_a=mask_a,
                    mask_b=mask_b,
                    frame=frame_index,
                    object_a_id=object_a_id,
                    object_b_id=object_b_id,
                )

                relationship_results.extend(
                    build_spatial_relationships(
                        relationship_metrics=(relationship_metric,),
                        relative_metrics=(relative_metric,),
                        object_id_map=object_id_map,
                        relative_entity_ids=(object_a_id, object_b_id),
                        source=source,
                    )
                )

        return tuple(relationship_results)

    @staticmethod
    def _calculate_region_observations(
        masks_by_frame: dict[int, dict[str, NDArray[np.uint8]]],
        object_id_map: dict[str, str] | object,
        regions: Sequence[RegionDefinition],
    ) -> tuple:
        observations = []

        for region in regions:
            for sam_object_id in sorted(object_id_map):
                entity_masks = {
                    frame_index: {
                        sam_object_id: frame_masks[sam_object_id]
                    }
                    for frame_index, frame_masks in masks_by_frame.items()
                    if sam_object_id in frame_masks
                }

                if not entity_masks:
                    continue

                observations.extend(
                    build_region_observations(
                        masks_by_frame=entity_masks,
                        object_id_map=object_id_map,
                        region=region,
                        sam_object_id=sam_object_id,
                    )
                )

        return tuple(observations)
