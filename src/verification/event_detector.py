"""Generic temporal event detection from normalized visual observations."""

import logging

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot
from statistics import median
from types import MappingProxyType
from typing import Mapping

from src.models.event import EventStatus, TemporalEvent
from src.models.event_detection_config import EventDetectionConfig
from src.models.evidence import (
    EntityObservation,
    Evidence,
    SpatialRelationship,
)
from src.models.observation_context import ObservationContext
from src.models.region import RegionObservation, RegionRelation
from src.models.task_roles import TaskRoles


logger = logging.getLogger(__name__)

EntityPair = tuple[str, str]
RegionKey = tuple[str, str]


@dataclass(frozen=True)
class _PreparedObservations:
    """Immutable, chronologically ordered observation groups."""

    entity_observations: Mapping[str, tuple[EntityObservation, ...]]
    spatial_relationships: Mapping[
        EntityPair, tuple[SpatialRelationship, ...]
    ]
    region_observations: Mapping[
        RegionKey, tuple[RegionObservation, ...]
    ]


class EventDetector:
    """Detect temporal events from normalized perception observations."""

    _MOTION_WINDOW_SECONDS = 0.50
    _MOTION_WINDOW_STRIDE_SECONDS = 0.25

    def __init__(
        self,
        config: EventDetectionConfig | None = None,
    ) -> None:
        if config is not None and not isinstance(
            config,
            EventDetectionConfig,
        ):
            raise TypeError("config must be an EventDetectionConfig.")

        self.config = config or EventDetectionConfig()

    def detect(
        self,
        entity_observations: Sequence[EntityObservation],
        spatial_relationships: Sequence[SpatialRelationship],
        region_observations: Sequence[RegionObservation],
        task_roles: TaskRoles,
        context: ObservationContext,
    ) -> tuple[TemporalEvent, ...]:
        """Prepare observations and detect supported temporal events."""

        prepared = self._prepare_observations(
            entity_observations,
            spatial_relationships,
            region_observations,
            task_roles,
            context,
        )

        events = self._detect_object_moving(prepared, task_roles)

        events.extend(
            self._detect_object_moving_with_effector(
                prepared,
                task_roles,
            )
        )
        events.extend(
            self._detect_relative_separation(
                prepared,
                task_roles,
            )
        )
        events.extend(
            self._detect_region_entry(prepared, task_roles)
        )
        events.extend(
            self._detect_stable_placement(prepared, task_roles)
        )

        return tuple(
            sorted(
                events,
                key=lambda event: (
                    event.frame_start,
                    event.frame_end,
                    event.event_type,
                    event.event_id,
                ),
            )
        )

    # ================================================================
    # Object movement
    # ================================================================

    def _detect_object_moving(
        self,
        prepared: _PreparedObservations,
        task_roles: TaskRoles,
    ) -> list[TemporalEvent]:
        if not task_roles.has_entity_role("object"):
            return []

        object_id = task_roles.get_entity_id("object")
        observations = prepared.entity_observations.get(object_id, ())

        events = []

        for segment in self._split_into_contiguous_segments(observations):
            event = self._build_object_moving_event(object_id, segment)
            if event is not None:
                events.append(event)

        return events

    def _build_object_moving_event(
        self,
        entity_id: str,
        observations: tuple[EntityObservation, ...],
    ) -> TemporalEvent | None:
        if len(observations) < self.config.min_observations:
            return None

        first = observations[0]
        last = observations[-1]
        duration = last.timestamp_seconds - first.timestamp_seconds

        if duration < self.config.min_event_duration_seconds:
            return None

        displacement = hypot(
            last.centroid[0] - first.centroid[0],
            last.centroid[1] - first.centroid[1],
        )

        if displacement < self.config.min_object_displacement_pixels:
            return None

        event_id = (
            f"OBJECT_MOVING:{entity_id}:"
            f"{first.frame_index}:{last.frame_index}"
        )

        evidence = Evidence(
            evidence_id=f"evidence:{event_id}",
            evidence_type="CENTROID_DISPLACEMENT",
            description=(
                "The entity centroid moved beyond the configured "
                "minimum net displacement."
            ),
            frame_start=first.frame_index,
            frame_end=last.frame_index,
            observation_ids=tuple(
                item.observation_id for item in observations
            ),
            entity_ids=(entity_id,),
            metrics={
                "displacement_pixels": displacement,
                "duration_seconds": duration,
                "observation_count": len(observations),
                "start_centroid_x": first.centroid[0],
                "start_centroid_y": first.centroid[1],
                "end_centroid_x": last.centroid[0],
                "end_centroid_y": last.centroid[1],
                "minimum_displacement_pixels": (
                    self.config.min_object_displacement_pixels
                ),
            },
            limitations=(
                "Net centroid displacement does not represent total path length.",
                "Pixel displacement is not a physical-world distance.",
                "The event does not establish the cause of the movement.",
            ),
        )

        confidence = min(
            1.0,
            displacement
            / (2.0 * self.config.min_object_displacement_pixels),
        )

        return TemporalEvent(
            event_id=event_id,
            event_type="OBJECT_MOVING",
            status=EventStatus.DETECTED,
            frame_start=first.frame_index,
            frame_end=last.frame_index,
            timestamp_start=first.timestamp_seconds,
            timestamp_end=last.timestamp_seconds,
            entity_ids=(entity_id,),
            confidence=confidence,
            evidence=(evidence,),
            attributes={
                "displacement_pixels": displacement,
                "duration_seconds": duration,
                "observation_count": len(observations),
                "measurement": "NET_CENTROID_DISPLACEMENT",
            },
        )

    # ================================================================
    # Object and effector coordinated movement
    # ================================================================

    def _detect_object_moving_with_effector(
        self,
        prepared: _PreparedObservations,
        task_roles: TaskRoles,
    ) -> list[TemporalEvent]:
        if not (
            task_roles.has_entity_role("object")
            and task_roles.has_entity_role("effector")
        ):
            return []

        object_id = task_roles.get_entity_id("object")
        effector_id = task_roles.get_entity_id("effector")

        if object_id == effector_id:
            logger.warning(
                "Coordinated movement skipped because object and "
                "effector resolve to the same entity ID."
            )
            return []

        object_by_frame = {
            item.frame_index: item
            for item in prepared.entity_observations.get(object_id, ())
        }
        effector_by_frame = {
            item.frame_index: item
            for item in prepared.entity_observations.get(effector_id, ())
        }

        pair = tuple(sorted((object_id, effector_id)))
        relationships = prepared.spatial_relationships.get(pair, ())
        relationship_by_frame = {
            item.frame_index: item
            for item in relationships
            if item.distance_pixels is not None
        }

        common_frames = sorted(
            set(object_by_frame)
            & set(effector_by_frame)
            & set(relationship_by_frame)
        )

        close_frames = [
            frame
            for frame in common_frames
            if relationship_by_frame[frame].distance_pixels
            <= self.config.max_relative_distance_pixels
        ]

        events = []

        for frames in self._split_frame_sequences(close_frames):
            if len(frames) < self.config.min_observations:
                continue

            object_segment = tuple(
                object_by_frame[frame] for frame in frames
            )
            effector_segment = tuple(
                effector_by_frame[frame] for frame in frames
            )
            relationship_segment = tuple(
                relationship_by_frame[frame] for frame in frames
            )

            event = self._build_object_moving_with_effector_event(
                object_id,
                effector_id,
                object_segment,
                effector_segment,
                relationship_segment,
            )

            if event is not None:
                events.append(event)

        logger.warning(
            "OBJECT_MOVING_WITH_EFFECTOR result | detected_events=%s",
            len(events),
        )

        return events

    def _calculate_windowed_motion_similarities(
        self,
        object_observations: tuple[EntityObservation, ...],
        effector_observations: tuple[EntityObservation, ...],
    ) -> tuple[float, ...]:
        """Compare accumulated movement directions over adaptive windows."""

        if len(object_observations) != len(effector_observations):
            return ()

        if len(object_observations) < self.config.min_observations:
            return ()

        total_duration = (
            object_observations[-1].timestamp_seconds
            - object_observations[0].timestamp_seconds
        )

        if total_duration <= 0:
            return ()

        window_duration = min(
            max(
                self._MOTION_WINDOW_SECONDS,
                self.config.min_event_duration_seconds,
            ),
            total_duration,
        )
        stride_duration = min(
            self._MOTION_WINDOW_STRIDE_SECONDS,
            window_duration / 2.0,
        )

        similarities: list[float] = []
        start_index = 0
        window_number = 0

        while start_index < len(object_observations) - 1:
            start_object = object_observations[start_index]
            start_effector = effector_observations[start_index]

            target_timestamp = (
                start_object.timestamp_seconds + window_duration
            )

            end_index = start_index + 1

            while end_index < len(object_observations):
                if (
                    object_observations[end_index].timestamp_seconds
                    >= target_timestamp
                ):
                    break
                end_index += 1

            if end_index >= len(object_observations):
                if (
                    object_observations[-1].timestamp_seconds
                    > start_object.timestamp_seconds
                ):
                    end_index = len(object_observations) - 1
                else:
                    break

            end_object = object_observations[end_index]
            end_effector = effector_observations[end_index]

            object_dx = (
                end_object.centroid[0] - start_object.centroid[0]
            )
            object_dy = (
                end_object.centroid[1] - start_object.centroid[1]
            )
            effector_dx = (
                end_effector.centroid[0] - start_effector.centroid[0]
            )
            effector_dy = (
                end_effector.centroid[1] - start_effector.centroid[1]
            )

            object_magnitude = hypot(object_dx, object_dy)
            effector_magnitude = hypot(effector_dx, effector_dy)

            minimum_displacement = max(
                1.0,
                self.config.min_object_displacement_pixels * 0.25,
            )

            window_number += 1

            if (
                object_magnitude >= minimum_displacement
                and effector_magnitude >= minimum_displacement
            ):
                similarity = (
                    object_dx * effector_dx
                    + object_dy * effector_dy
                ) / (object_magnitude * effector_magnitude)

                similarity = max(-1.0, min(1.0, similarity))
                similarities.append(similarity)

                logger.warning(
                    "MOTION WINDOW DETAIL | window=%s | frames=%s-%s | "
                    "object_delta=(%.2f, %.2f) | "
                    "effector_delta=(%.2f, %.2f) | "
                    "object_displacement=%.2f | "
                    "effector_displacement=%.2f | "
                    "similarity=%.4f | qualifies=%s",
                    window_number,
                    start_object.frame_index,
                    end_object.frame_index,
                    object_dx,
                    object_dy,
                    effector_dx,
                    effector_dy,
                    object_magnitude,
                    effector_magnitude,
                    similarity,
                    similarity >= self.config.min_motion_similarity,
                )
            else:
                logger.warning(
                    "MOTION WINDOW SKIPPED | window=%s | frames=%s-%s | "
                    "object_displacement=%.2f | "
                    "effector_displacement=%.2f | "
                    "minimum_required=%.2f",
                    window_number,
                    start_object.frame_index,
                    end_object.frame_index,
                    object_magnitude,
                    effector_magnitude,
                    minimum_displacement,
                )

            next_timestamp = (
                start_object.timestamp_seconds + stride_duration
            )

            next_start_index = start_index + 1

            while next_start_index < len(object_observations) - 1:
                if (
                    object_observations[
                        next_start_index
                    ].timestamp_seconds >= next_timestamp
                ):
                    break
                next_start_index += 1

            if next_start_index <= start_index:
                next_start_index = start_index + 1

            start_index = next_start_index

        return tuple(similarities)

    def _build_object_moving_with_effector_event(
        self,
        object_id: str,
        effector_id: str,
        object_observations: tuple[EntityObservation, ...],
        effector_observations: tuple[EntityObservation, ...],
        relationships: tuple[SpatialRelationship, ...],
    ) -> TemporalEvent | None:
        if not (
            len(object_observations)
            == len(effector_observations)
            == len(relationships)
        ):
            return None

        if len(object_observations) < self.config.min_observations:
            return None

        first_object = object_observations[0]
        last_object = object_observations[-1]
        first_effector = effector_observations[0]
        last_effector = effector_observations[-1]

        duration = (
            last_object.timestamp_seconds
            - first_object.timestamp_seconds
        )

        if duration < self.config.min_event_duration_seconds:
            return None

        object_displacement = hypot(
            last_object.centroid[0] - first_object.centroid[0],
            last_object.centroid[1] - first_object.centroid[1],
        )
        effector_displacement = hypot(
            last_effector.centroid[0] - first_effector.centroid[0],
            last_effector.centroid[1] - first_effector.centroid[1],
        )

        minimum_displacement = (
            self.config.min_object_displacement_pixels
        )

        if (
            object_displacement < minimum_displacement
            or effector_displacement < minimum_displacement
        ):
            return None

        similarities = self._calculate_windowed_motion_similarities(
            object_observations,
            effector_observations,
        )

        if not similarities:
            logger.warning(
                "OBJECT_MOVING_WITH_EFFECTOR rejected | "
                "Reason: no valid motion windows."
            )
            return None

        mean_similarity = sum(similarities) / len(similarities)
        median_similarity = median(similarities)
        qualifying_ratio = sum(
            value >= self.config.min_motion_similarity
            for value in similarities
        ) / len(similarities)

        logger.warning(
            "OBJECT_MOVING_WITH_EFFECTOR window diagnostics | "
            "frames=%s-%s | windows=%s | mean=%.4f | median=%.4f | "
            "qualifying_ratio=%.4f | threshold=%.4f",
            first_object.frame_index,
            last_object.frame_index,
            len(similarities),
            mean_similarity,
            median_similarity,
            qualifying_ratio,
            self.config.min_motion_similarity,
        )

        if (
            median_similarity < self.config.min_motion_similarity
            or qualifying_ratio < 0.50
        ):
            return None

        mask_distances = [
            item.distance_pixels
            for item in relationships
            if item.distance_pixels is not None
        ]

        if not mask_distances:
            return None

        mean_mask_distance = sum(mask_distances) / len(mask_distances)
        maximum_mask_distance = max(mask_distances)

        event_id = (
            f"OBJECT_MOVING_WITH_EFFECTOR:{object_id}:{effector_id}:"
            f"{first_object.frame_index}:{last_object.frame_index}"
        )

        observation_ids = tuple(
            observation.observation_id
            for pair in zip(object_observations, effector_observations)
            for observation in pair
        )

        evidence = Evidence(
            evidence_id=f"evidence:{event_id}",
            evidence_type="COORDINATED_ENTITY_MOTION",
            description=(
                "The object and effector moved in similar directions "
                "across overlapping temporal windows while their visual "
                "masks remained within the configured maximum distance."
            ),
            frame_start=first_object.frame_index,
            frame_end=last_object.frame_index,
            observation_ids=observation_ids,
            entity_ids=(object_id, effector_id),
            metrics={
                "object_displacement_pixels": object_displacement,
                "effector_displacement_pixels": effector_displacement,
                "mean_motion_similarity": mean_similarity,
                "median_motion_similarity": median_similarity,
                "qualifying_motion_window_ratio": qualifying_ratio,
                "minimum_motion_similarity": (
                    self.config.min_motion_similarity
                ),
                "motion_window_duration_seconds": min(
                    max(
                        self._MOTION_WINDOW_SECONDS,
                        self.config.min_event_duration_seconds,
                    ),
                    duration,
                ),
                "valid_motion_window_count": len(similarities),
                "mean_mask_distance_pixels": mean_mask_distance,
                "maximum_observed_mask_distance_pixels": (
                    maximum_mask_distance
                ),
                "maximum_allowed_mask_distance_pixels": (
                    self.config.max_relative_distance_pixels
                ),
                "duration_seconds": duration,
                "observation_count": len(object_observations),
            },
            limitations=(
                "Coordinated visual movement does not prove physical contact "
                "or a successful grasp.",
                "Mask distance is a two-dimensional image-space measurement.",
                "Centroid motion does not establish force, suction, or hidden "
                "gripper state.",
                "The motion similarity score is a geometric heuristic, "
                "not a calibrated probability.",
            ),
        )

        displacement_score = min(
            1.0,
            min(object_displacement, effector_displacement)
            / (2.0 * minimum_displacement),
        )
        proximity_score = max(
            0.0,
            1.0 - (
                mean_mask_distance
                / self.config.max_relative_distance_pixels
            ),
        )

        confidence = max(
            0.0,
            min(
                1.0,
                0.5 * median_similarity
                + 0.25 * displacement_score
                + 0.25 * proximity_score,
            ),
        )

        return TemporalEvent(
            event_id=event_id,
            event_type="OBJECT_MOVING_WITH_EFFECTOR",
            status=EventStatus.DETECTED,
            frame_start=first_object.frame_index,
            frame_end=last_object.frame_index,
            timestamp_start=first_object.timestamp_seconds,
            timestamp_end=last_object.timestamp_seconds,
            entity_ids=(object_id, effector_id),
            confidence=confidence,
            evidence=(evidence,),
            attributes={
                "object_displacement_pixels": object_displacement,
                "effector_displacement_pixels": effector_displacement,
                "mean_motion_similarity": mean_similarity,
                "median_motion_similarity": median_similarity,
                "qualifying_motion_window_ratio": qualifying_ratio,
                "mean_mask_distance_pixels": mean_mask_distance,
                "duration_seconds": duration,
                "observation_count": len(object_observations),
                "valid_motion_window_count": len(similarities),
                "measurement": "WINDOWED_COORDINATED_CENTROID_MOTION",
            },
        )

    # ================================================================
    # Object and effector relative separation
    # ================================================================

    def _detect_relative_separation(
        self,
        prepared: _PreparedObservations,
        task_roles: TaskRoles,
    ) -> list[TemporalEvent]:
        """Detect increasing object-effector separation."""

        if not (
            task_roles.has_entity_role("object")
            and task_roles.has_entity_role("effector")
        ):
            return []

        object_id = task_roles.get_entity_id("object")
        effector_id = task_roles.get_entity_id("effector")

        if object_id == effector_id:
            return []

        object_by_frame = {
            item.frame_index: item
            for item in prepared.entity_observations.get(object_id, ())
        }
        effector_by_frame = {
            item.frame_index: item
            for item in prepared.entity_observations.get(effector_id, ())
        }

        common_frames = sorted(
            set(object_by_frame) & set(effector_by_frame)
        )

        if len(common_frames) < self.config.min_observations:
            return []

        pair = tuple(sorted((object_id, effector_id)))
        relationships = prepared.spatial_relationships.get(pair, ())
        relationship_by_frame = {
            item.frame_index: item
            for item in relationships
            if item.distance_pixels is not None
        }

        has_complete_mask_distances = all(
            frame in relationship_by_frame
            for frame in common_frames
        )

        events = []

        for frames in self._split_frame_sequences(common_frames):
            if len(frames) < self.config.min_observations:
                continue

            object_segment = tuple(
                object_by_frame[frame] for frame in frames
            )
            effector_segment = tuple(
                effector_by_frame[frame] for frame in frames
            )

            if has_complete_mask_distances:
                relationship_segment = tuple(
                    relationship_by_frame[frame]
                    for frame in frames
                )
                event = self._build_relative_separation_event(
                    object_id,
                    effector_id,
                    object_segment,
                    effector_segment,
                    relationship_segment,
                    require_prior_proximity=True,
                )
            else:
                event = self._build_centroid_relative_separation_event(
                    object_id,
                    effector_id,
                    object_segment,
                    effector_segment,
                )

            if event is not None:
                events.append(event)

        logger.warning(
            "RELATIVE_SEPARATION result | detected_events=%s",
            len(events),
        )

        return events

    def _build_centroid_relative_separation_event(
        self,
        object_id: str,
        effector_id: str,
        object_observations: tuple[EntityObservation, ...],
        effector_observations: tuple[EntityObservation, ...],
    ) -> TemporalEvent | None:
        """Fallback separation detection using centroid-to-centroid distance."""

        if len(object_observations) != len(effector_observations):
            return None

        if len(object_observations) < self.config.min_observations:
            return None

        distances = tuple(
            hypot(
                object_item.centroid[0] - effector_item.centroid[0],
                object_item.centroid[1] - effector_item.centroid[1],
            )
            for object_item, effector_item in zip(
                object_observations,
                effector_observations,
            )
        )

        return self._build_separation_event_from_distances(
            object_id=object_id,
            effector_id=effector_id,
            object_observations=object_observations,
            effector_observations=effector_observations,
            distances=distances,
            evidence_type="RELATIVE_CENTROID_SEPARATION",
            require_prior_proximity=False,
        )

    def _build_relative_separation_event(
        self,
        object_id: str,
        effector_id: str,
        object_observations: tuple[EntityObservation, ...],
        effector_observations: tuple[EntityObservation, ...],
        relationships: tuple[SpatialRelationship, ...],
        require_prior_proximity: bool = True,
    ) -> TemporalEvent | None:
        """Build separation evidence from mask-to-mask distances."""

        if not (
            len(object_observations)
            == len(effector_observations)
            == len(relationships)
        ):
            return None

        if len(object_observations) < self.config.min_observations:
            return None

        if any(
            relationship.distance_pixels is None
            for relationship in relationships
        ):
            return None

        distances = tuple(
            float(relationship.distance_pixels)
            for relationship in relationships
        )

        return self._build_separation_event_from_distances(
            object_id=object_id,
            effector_id=effector_id,
            object_observations=object_observations,
            effector_observations=effector_observations,
            distances=distances,
            evidence_type="MASK_DISTANCE_SEPARATION",
            require_prior_proximity=require_prior_proximity,
        )

    def _build_separation_event_from_distances(
        self,
        object_id: str,
        effector_id: str,
        object_observations: tuple[EntityObservation, ...],
        effector_observations: tuple[EntityObservation, ...],
        distances: tuple[float, ...],
        evidence_type: str,
        require_prior_proximity: bool,
    ) -> TemporalEvent | None:
        """Apply common temporal separation criteria to a distance series."""

        if not (
            len(object_observations)
            == len(effector_observations)
            == len(distances)
        ):
            return None

        if len(distances) < self.config.min_observations:
            return None

        minimum_increase = (
            self.config.min_relative_separation_increase_pixels
        )
        proximity_threshold = self.config.max_relative_distance_pixels
        minimum_count = self.config.min_observations

        candidate: tuple[int, int] | None = None

        if require_prior_proximity:
            # A mask-distance release candidate must start near the
            # effector and then remain separated for confirmation frames.
            for separation_start in range(
                1,
                len(distances) - minimum_count + 1,
            ):
                confirmation_end = (
                    separation_start + minimum_count - 1
                )

                if not all(
                    distance > proximity_threshold
                    for distance in distances[
                        separation_start:confirmation_end + 1
                    ]
                ):
                    continue

                for approach_start in range(
                    separation_start - 1,
                    -1,
                    -1,
                ):
                    initial_distance = distances[approach_start]
                    final_distance = distances[confirmation_end]

                    if initial_distance > proximity_threshold:
                        continue

                    if (
                        final_distance - initial_distance
                        < minimum_increase
                    ):
                        continue

                    changes = [
                        current - previous
                        for previous, current in zip(
                            distances[approach_start:confirmation_end],
                            distances[
                                approach_start + 1:confirmation_end + 1
                            ],
                        )
                    ]

                    if not changes:
                        continue

                    positive_ratio = (
                        sum(change > 0 for change in changes)
                        / len(changes)
                    )

                    if positive_ratio <= 0.5:
                        continue

                    duration = (
                        object_observations[
                            confirmation_end
                        ].timestamp_seconds
                        - object_observations[
                            approach_start
                        ].timestamp_seconds
                    )

                    if duration < self.config.min_event_duration_seconds:
                        continue

                    candidate = (approach_start, confirmation_end)
                    break

                if candidate is not None:
                    break

        else:
            # Evaluate the complete contiguous centroid sequence.
            # Do not accept a shorter prefix when the complete sequence
            # fails the increasing-step ratio requirement.
            initial_distance = distances[0]
            final_distance = distances[-1]

            changes = [
                current - previous
                for previous, current in zip(
                    distances,
                    distances[1:],
                )
            ]

            if not changes:
                return None

            positive_ratio = (
                sum(change > 0 for change in changes)
                / len(changes)
            )

            duration = (
                object_observations[-1].timestamp_seconds
                - object_observations[0].timestamp_seconds
            )

            if (
                final_distance - initial_distance >= minimum_increase
                and positive_ratio > 0.5
                and duration >= self.config.min_event_duration_seconds
            ):
                candidate = (0, len(distances) - 1)

        if candidate is None:
            return None

        start_index, end_index = candidate

        selected_objects = object_observations[start_index:end_index + 1]
        selected_effectors = effector_observations[start_index:end_index + 1]
        selected_distances = distances[start_index:end_index + 1]

        first = selected_objects[0]
        last = selected_objects[-1]

        initial_distance = selected_distances[0]
        final_distance = selected_distances[-1]
        increase = final_distance - initial_distance

        changes = [
            current - previous
            for previous, current in zip(
                selected_distances,
                selected_distances[1:],
            )
        ]

        positive_steps = sum(change > 0 for change in changes)
        negative_steps = sum(change < 0 for change in changes)
        flat_steps = sum(change == 0 for change in changes)
        step_count = len(changes)

        positive_ratio = (
            positive_steps / step_count if step_count else 0.0
        )

        duration = last.timestamp_seconds - first.timestamp_seconds
        minimum_distance = min(selected_distances)
        maximum_distance = max(selected_distances)
        mean_distance = (
            sum(selected_distances) / len(selected_distances)
        )

        event_id = (
            f"RELATIVE_SEPARATION:{object_id}:{effector_id}:"
            f"{first.frame_index}:{last.frame_index}"
        )

        observation_ids = tuple(
            observation.observation_id
            for pair in zip(selected_objects, selected_effectors)
            for observation in pair
        )

        evidence = Evidence(
            evidence_id=f"evidence:{event_id}",
            evidence_type=evidence_type,
            description=(
                "The object and effector showed an increasing relative "
                "image-space distance over the detected interval."
            ),
            frame_start=first.frame_index,
            frame_end=last.frame_index,
            observation_ids=observation_ids,
            entity_ids=(object_id, effector_id),
            metrics={
                "initial_distance_pixels": initial_distance,
                "final_distance_pixels": final_distance,
                "initial_centroid_distance_pixels": initial_distance,
                "final_centroid_distance_pixels": final_distance,
                "separation_increase_pixels": increase,
                "minimum_distance_pixels": minimum_distance,
                "mean_distance_pixels": mean_distance,
                "maximum_distance_pixels": maximum_distance,
                "proximity_threshold_pixels": proximity_threshold,
                "minimum_separation_increase_pixels": minimum_increase,
                "positive_separation_steps": positive_steps,
                "negative_separation_steps": negative_steps,
                "flat_separation_steps": flat_steps,
                "total_separation_steps": step_count,
                "positive_step_ratio": positive_ratio,
                "duration_seconds": duration,
                "observation_count": len(selected_distances),
                "confirmation_observation_count": minimum_count,
                "prior_proximity_required": require_prior_proximity,
            },
            limitations=(
                "Increasing image-space distance does not prove that the "
                "object was physically released.",
                "Image-space distance is not a physical-world distance.",
                "Apparent separation can be affected by occlusion, camera "
                "movement, perspective, and segmentation error.",
                "The event does not establish the cause of the separation.",
                "The detector score is a geometric heuristic, not a "
                "calibrated probability.",
            ),
        )

        increase_score = min(
            1.0,
            increase / (2.0 * minimum_increase),
        )
        confidence = max(
            0.0,
            min(1.0, 0.5 * increase_score + 0.5 * positive_ratio),
        )

        return TemporalEvent(
            event_id=event_id,
            event_type="RELATIVE_SEPARATION",
            status=EventStatus.DETECTED,
            frame_start=first.frame_index,
            frame_end=last.frame_index,
            timestamp_start=first.timestamp_seconds,
            timestamp_end=last.timestamp_seconds,
            entity_ids=(object_id, effector_id),
            confidence=confidence,
            evidence=(evidence,),
            attributes={
                "initial_distance_pixels": initial_distance,
                "final_distance_pixels": final_distance,
                "separation_increase_pixels": increase,
                "positive_step_ratio": positive_ratio,
                "duration_seconds": duration,
                "observation_count": len(selected_distances),
                "measurement": (
                    "LOCAL_MASK_DISTANCE_SEPARATION"
                    if require_prior_proximity
                    else "CENTROID_RELATIVE_SEPARATION"
                ),
            },
        )

    # ================================================================
    # Region entry
    # ================================================================

    def _detect_region_entry(
        self,
        prepared: _PreparedObservations,
        task_roles: TaskRoles,
    ) -> list[TemporalEvent]:
        if not (
            task_roles.has_entity_role("object")
            and task_roles.has_region_role("target_region")
        ):
            return []

        object_id = task_roles.get_entity_id("object")
        region_id = task_roles.get_region_id("target_region")
        region_observations = prepared.region_observations.get(
            (object_id, region_id),
            (),
        )

        entity_by_frame = {
            item.frame_index: item
            for item in prepared.entity_observations.get(object_id, ())
        }

        records = []
        segments = []

        def flush() -> None:
            if records:
                segments.append(tuple(records))
                records.clear()

        for region_item in region_observations:
            entity_item = entity_by_frame.get(region_item.frame_index)

            if (
                entity_item is None
                or region_item.relation == RegionRelation.UNKNOWN
                or region_item.overlap_ratio is None
            ):
                flush()
                continue

            if records:
                missing = (
                    region_item.frame_index
                    - records[-1][0].frame_index
                    - 1
                )
                if missing > self.config.max_frame_gap:
                    flush()

            records.append((region_item, entity_item))

        flush()

        events = []
        threshold = self.config.min_region_overlap_ratio
        minimum_count = self.config.min_observations

        for segment in segments:
            if len(segment) < minimum_count + 1:
                continue

            index = 1

            while index < len(segment):
                previous_ratio = segment[index - 1][0].overlap_ratio
                current_ratio = segment[index][0].overlap_ratio

                if (
                    previous_ratio is None
                    or current_ratio is None
                    or previous_ratio >= threshold
                    or current_ratio < threshold
                ):
                    index += 1
                    continue

                confirmation_end = index

                while confirmation_end + 1 < len(segment):
                    next_ratio = segment[confirmation_end + 1][0].overlap_ratio
                    if next_ratio is None or next_ratio < threshold:
                        break
                    confirmation_end += 1

                if confirmation_end - index + 1 >= minimum_count:
                    event = self._build_region_entry_event(
                        object_id,
                        region_id,
                        segment[index - 1:confirmation_end + 1],
                        1,
                    )
                    if event is not None:
                        events.append(event)

                index = confirmation_end + 1

        return events

    def _build_region_entry_event(
        self,
        object_id: str,
        region_id: str,
        transition_records: tuple[
            tuple[RegionObservation, EntityObservation], ...
        ],
        entry_index: int,
    ) -> TemporalEvent | None:
        if len(transition_records) < 2:
            return None

        if not 0 < entry_index < len(transition_records):
            return None

        previous_region, previous_entity = transition_records[0]
        entry_region, entry_entity = transition_records[entry_index]
        final_region, final_entity = transition_records[-1]

        previous_ratio = previous_region.overlap_ratio
        entry_ratio = entry_region.overlap_ratio
        threshold = self.config.min_region_overlap_ratio

        if (
            previous_ratio is None
            or entry_ratio is None
            or previous_ratio >= threshold
            or entry_ratio < threshold
        ):
            return None

        confirmation = transition_records[entry_index:]
        if len(confirmation) < self.config.min_observations:
            return None

        ratios = [
            float(item.overlap_ratio)
            for item, _ in confirmation
            if item.overlap_ratio is not None
        ]

        if (
            len(ratios) != len(confirmation)
            or any(value < threshold for value in ratios)
        ):
            return None

        mean_overlap = sum(ratios) / len(ratios)
        duration = (
            final_entity.timestamp_seconds
            - previous_entity.timestamp_seconds
        )

        if duration < 0:
            return None

        event_id = (
            f"REGION_ENTRY:{object_id}:{region_id}:"
            f"{entry_entity.frame_index}:{final_entity.frame_index}"
        )

        evidence = Evidence(
            evidence_id=f"evidence:{event_id}",
            evidence_type="ENTITY_ENTERED_REGION",
            description=(
                "The object crossed the configured region-overlap "
                "threshold and remained above it for the required "
                "number of consecutive observations."
            ),
            frame_start=previous_entity.frame_index,
            frame_end=final_entity.frame_index,
            observation_ids=tuple(
                entity.observation_id
                for _, entity in transition_records
            ),
            entity_ids=(object_id,),
            metrics={
                "region_id": region_id,
                "overlap_before_entry": previous_ratio,
                "overlap_at_entry": entry_ratio,
                "mean_overlap_during_confirmation": mean_overlap,
                "minimum_overlap_during_confirmation": min(ratios),
                "maximum_overlap_during_confirmation": max(ratios),
                "required_overlap_ratio": threshold,
                "confirmation_observation_count": len(confirmation),
                "duration_seconds": duration,
            },
            limitations=(
                "Region membership is evaluated from a two-dimensional "
                "image-space segmentation mask.",
                "Image-space overlap does not prove physical placement "
                "inside the target in three-dimensional space.",
                "The event does not establish contact with the target "
                "surface or stable support.",
                "Perspective, camera movement, occlusion, and mask "
                "segmentation errors can affect the overlap measurement.",
                "The event does not establish complete task success.",
            ),
        )

        return TemporalEvent(
            event_id=event_id,
            event_type="REGION_ENTRY",
            status=EventStatus.DETECTED,
            frame_start=previous_entity.frame_index,
            frame_end=final_entity.frame_index,
            timestamp_start=previous_entity.timestamp_seconds,
            timestamp_end=final_entity.timestamp_seconds,
            entity_ids=(object_id,),
            confidence=max(0.0, min(1.0, mean_overlap)),
            evidence=(evidence,),
            attributes={
                "region_id": region_id,
                "entry_frame": entry_entity.frame_index,
                "confirmation_end_frame": final_entity.frame_index,
                "overlap_before_entry": previous_ratio,
                "overlap_at_entry": entry_ratio,
                "mean_overlap_ratio": mean_overlap,
                "minimum_overlap_ratio": min(ratios),
                "maximum_overlap_ratio": max(ratios),
                "required_overlap_ratio": threshold,
                "confirmation_observation_count": len(confirmation),
                "duration_seconds": duration,
                "measurement": "REGION_MASK_OVERLAP_RATIO",
                "confidence_basis": "MEAN_CONFIRMATION_OVERLAP_RATIO",
            },
        )

    # ================================================================
    # Stable placement
    # ================================================================

    def _detect_stable_placement(
        self,
        prepared: _PreparedObservations,
        task_roles: TaskRoles,
    ) -> list[TemporalEvent]:
        if not (
            task_roles.has_entity_role("object")
            and task_roles.has_region_role("target_region")
        ):
            return []

        object_id = task_roles.get_entity_id("object")
        region_id = task_roles.get_region_id("target_region")
        region_observations = prepared.region_observations.get(
            (object_id, region_id),
            (),
        )

        entity_by_frame = {
            item.frame_index: item
            for item in prepared.entity_observations.get(object_id, ())
        }

        threshold = self.config.min_region_overlap_ratio
        in_region_segments = []
        current = []

        def flush_region() -> None:
            if current:
                in_region_segments.append(tuple(current))
                current.clear()

        for region_item in region_observations:
            entity_item = entity_by_frame.get(region_item.frame_index)

            if (
                entity_item is None
                or region_item.relation == RegionRelation.UNKNOWN
                or region_item.overlap_ratio is None
                or region_item.overlap_ratio < threshold
            ):
                flush_region()
                continue

            if current:
                missing = (
                    region_item.frame_index - current[-1][0].frame_index - 1
                )
                if missing > self.config.max_frame_gap:
                    flush_region()

            current.append((region_item, entity_item))

        flush_region()

        events = []

        for segment in in_region_segments:
            stable_records = []
            step_speeds = []

            def flush_stable() -> None:
                if len(stable_records) >= self.config.min_observations:
                    event = self._build_stable_placement_event(
                        object_id,
                        region_id,
                        tuple(stable_records),
                        tuple(step_speeds),
                    )
                    if event is not None:
                        events.append(event)

                stable_records.clear()
                step_speeds.clear()

            for record in segment:
                if not stable_records:
                    stable_records.append(record)
                    continue

                previous_entity = stable_records[-1][1]
                current_entity = record[1]
                elapsed = (
                    current_entity.timestamp_seconds
                    - previous_entity.timestamp_seconds
                )

                if elapsed <= 0:
                    flush_stable()
                    stable_records.append(record)
                    continue

                displacement = hypot(
                    current_entity.centroid[0]
                    - previous_entity.centroid[0],
                    current_entity.centroid[1]
                    - previous_entity.centroid[1],
                )
                speed = displacement / elapsed

                if (
                    speed
                    <= self.config.max_stationary_speed_pixels_per_second
                ):
                    stable_records.append(record)
                    step_speeds.append(speed)
                else:
                    flush_stable()
                    stable_records.append(record)

            flush_stable()

        return events

    def _build_stable_placement_event(
        self,
        object_id: str,
        region_id: str,
        stable_records: tuple[
            tuple[RegionObservation, EntityObservation], ...
        ],
        step_speeds: tuple[float, ...],
    ) -> TemporalEvent | None:
        if len(stable_records) < self.config.min_observations:
            return None

        if len(step_speeds) != len(stable_records) - 1:
            return None

        first_region, first_entity = stable_records[0]
        final_region, final_entity = stable_records[-1]

        duration = (
            final_entity.timestamp_seconds
            - first_entity.timestamp_seconds
        )

        if (
            duration < self.config.min_stable_duration_seconds
            or duration <= 0
        ):
            return None

        overlap_ratios = [
            item.overlap_ratio
            for item, _ in stable_records
        ]

        if any(
            value is None or value < self.config.min_region_overlap_ratio
            for value in overlap_ratios
        ):
            return None

        ratios = [float(value) for value in overlap_ratios]
        maximum_speed = max(step_speeds, default=0.0)

        if (
            maximum_speed
            > self.config.max_stationary_speed_pixels_per_second
        ):
            return None

        mean_speed = sum(step_speeds) / len(step_speeds)
        mean_overlap = sum(ratios) / len(ratios)
        minimum_overlap = min(ratios)

        event_id = (
            f"STABLE_PLACEMENT:{object_id}:{region_id}:"
            f"{first_entity.frame_index}:{final_entity.frame_index}"
        )

        evidence = Evidence(
            evidence_id=f"evidence:{event_id}",
            evidence_type="STABLE_IN_REGION",
            description=(
                "The object remained above the configured region-overlap "
                "threshold while its centroid speed stayed within the "
                "configured stationary-speed limit for the required duration."
            ),
            frame_start=first_entity.frame_index,
            frame_end=final_entity.frame_index,
            observation_ids=tuple(
                entity.observation_id
                for _, entity in stable_records
            ),
            entity_ids=(object_id,),
            metrics={
                "region_id": region_id,
                "duration_seconds": duration,
                "observation_count": len(stable_records),
                "mean_overlap_ratio": mean_overlap,
                "minimum_overlap_ratio": minimum_overlap,
                "mean_step_speed_pixels_per_second": mean_speed,
                "maximum_step_speed_pixels_per_second": maximum_speed,
                "maximum_allowed_stationary_speed_pixels_per_second": (
                    self.config.max_stationary_speed_pixels_per_second
                ),
                "minimum_required_overlap_ratio": (
                    self.config.min_region_overlap_ratio
                ),
                "minimum_required_duration_seconds": (
                    self.config.min_stable_duration_seconds
                ),
            },
            limitations=(
                "Stability is estimated from two-dimensional image-space "
                "centroid motion and does not establish three-dimensional "
                "physical stability.",
                "A stable centroid does not prove physical contact with "
                "or support from the target surface.",
                "Centroid stability does not measure object rotation, "
                "tilting, force, or hidden physical state.",
                "Camera movement, perspective, occlusion, and segmentation "
                "jitter can affect the measurements.",
                "The event does not establish successful completion of "
                "the complete task.",
                "The stability score is a geometric heuristic, not a "
                "calibrated probability.",
            ),
        )

        speed_limit = (
            self.config.max_stationary_speed_pixels_per_second
        )
        speed_score = max(
            0.0,
            min(1.0, 1.0 - maximum_speed / speed_limit),
        )
        confidence = max(
            0.0,
            min(1.0, 0.5 * mean_overlap + 0.5 * speed_score),
        )

        return TemporalEvent(
            event_id=event_id,
            event_type="STABLE_PLACEMENT",
            status=EventStatus.DETECTED,
            frame_start=first_entity.frame_index,
            frame_end=final_entity.frame_index,
            timestamp_start=first_entity.timestamp_seconds,
            timestamp_end=final_entity.timestamp_seconds,
            entity_ids=(object_id,),
            confidence=confidence,
            evidence=(evidence,),
            attributes={
                "region_id": region_id,
                "duration_seconds": duration,
                "observation_count": len(stable_records),
                "mean_overlap_ratio": mean_overlap,
                "minimum_overlap_ratio": minimum_overlap,
                "mean_step_speed_pixels_per_second": mean_speed,
                "maximum_step_speed_pixels_per_second": maximum_speed,
                "maximum_allowed_stationary_speed_pixels_per_second": (
                    speed_limit
                ),
                "minimum_required_overlap_ratio": (
                    self.config.min_region_overlap_ratio
                ),
                "minimum_required_duration_seconds": (
                    self.config.min_stable_duration_seconds
                ),
                "measurement": (
                    "CENTROID_STABILITY_AND_REGION_OVERLAP"
                ),
                "confidence_basis": (
                    "HEURISTIC_OVERLAP_AND_STABILITY_SCORE"
                ),
            },
        )

    # ================================================================
    # Generic observation segmentation
    # ================================================================

    def _split_into_contiguous_segments(
        self,
        observations: Sequence[EntityObservation],
    ) -> tuple[tuple[EntityObservation, ...], ...]:
        if not observations:
            return ()

        segments = []
        current = [observations[0]]

        for observation in observations[1:]:
            missing = (
                observation.frame_index
                - current[-1].frame_index
                - 1
            )

            if missing > self.config.max_frame_gap:
                segments.append(tuple(current))
                current = []

            current.append(observation)

        segments.append(tuple(current))
        return tuple(segments)

    def _split_frame_sequences(
        self,
        frame_indices: Sequence[int],
    ) -> tuple[tuple[int, ...], ...]:
        if not frame_indices:
            return ()

        segments = []
        current = [frame_indices[0]]

        for frame_index in frame_indices[1:]:
            missing = frame_index - current[-1] - 1

            if missing > self.config.max_frame_gap:
                segments.append(tuple(current))
                current = []

            current.append(frame_index)

        segments.append(tuple(current))
        return tuple(segments)

    # ================================================================
    # Observation preparation and validation
    # ================================================================

    def _prepare_observations(
        self,
        entity_observations: Sequence[EntityObservation],
        spatial_relationships: Sequence[SpatialRelationship],
        region_observations: Sequence[RegionObservation],
        task_roles: TaskRoles,
        context: ObservationContext,
    ) -> _PreparedObservations:
        self._validate_inputs(
            entity_observations,
            spatial_relationships,
            region_observations,
            task_roles,
            context,
        )

        entities_by_id = defaultdict(list)
        relationships_by_pair = defaultdict(list)
        regions_by_key = defaultdict(list)

        for observation in entity_observations:
            entities_by_id[observation.entity_id].append(observation)

        for relationship in spatial_relationships:
            pair = tuple(
                sorted(
                    (
                        relationship.entity_a_id,
                        relationship.entity_b_id,
                    )
                )
            )
            relationships_by_pair[pair].append(relationship)

        for observation in region_observations:
            key = (observation.entity_id, observation.region_id)
            regions_by_key[key].append(observation)

        for values in entities_by_id.values():
            values.sort(key=lambda item: item.frame_index)

        for values in relationships_by_pair.values():
            values.sort(key=lambda item: item.frame_index)

        for values in regions_by_key.values():
            values.sort(key=lambda item: item.frame_index)

        return _PreparedObservations(
            entity_observations=MappingProxyType(
                {
                    key: tuple(values)
                    for key, values in entities_by_id.items()
                }
            ),
            spatial_relationships=MappingProxyType(
                {
                    key: tuple(values)
                    for key, values in relationships_by_pair.items()
                }
            ),
            region_observations=MappingProxyType(
                {
                    key: tuple(values)
                    for key, values in regions_by_key.items()
                }
            ),
        )

    def _validate_inputs(
        self,
        entity_observations: Sequence[EntityObservation],
        spatial_relationships: Sequence[SpatialRelationship],
        region_observations: Sequence[RegionObservation],
        task_roles: TaskRoles,
        context: ObservationContext,
    ) -> None:
        self._validate_sequence(
            entity_observations,
            "entity_observations",
            EntityObservation,
        )
        self._validate_sequence(
            spatial_relationships,
            "spatial_relationships",
            SpatialRelationship,
        )
        self._validate_sequence(
            region_observations,
            "region_observations",
            RegionObservation,
        )

        if not isinstance(task_roles, TaskRoles):
            raise TypeError("task_roles must be a TaskRoles instance.")

        if not isinstance(context, ObservationContext):
            raise TypeError(
                "context must be an ObservationContext instance."
            )

        entity_ids = set()
        observation_ids = set()
        entity_frame_keys = set()

        for observation in entity_observations:
            self._validate_frame(
                observation.frame_index,
                context,
                "EntityObservation",
            )

            if observation.observation_id in observation_ids:
                raise ValueError(
                    "Duplicate observation_id: "
                    f"{observation.observation_id!r}."
                )

            observation_ids.add(observation.observation_id)

            frame_key = (
                observation.entity_id,
                observation.frame_index,
            )

            if frame_key in entity_frame_keys:
                raise ValueError(
                    "Duplicate entity observation for entity "
                    f"{observation.entity_id!r} in frame "
                    f"{observation.frame_index}."
                )

            entity_frame_keys.add(frame_key)
            entity_ids.add(observation.entity_id)

        relationship_frame_keys = set()

        for relationship in spatial_relationships:
            self._validate_frame(
                relationship.frame_index,
                context,
                "SpatialRelationship",
            )

            if relationship.entity_a_id not in entity_ids:
                raise ValueError(
                    "SpatialRelationship references unknown entity: "
                    f"{relationship.entity_a_id!r}."
                )

            if relationship.entity_b_id not in entity_ids:
                raise ValueError(
                    "SpatialRelationship references unknown entity: "
                    f"{relationship.entity_b_id!r}."
                )

            if relationship.entity_a_id == relationship.entity_b_id:
                raise ValueError(
                    "SpatialRelationship must reference two distinct entities."
                )

            pair = tuple(
                sorted(
                    (
                        relationship.entity_a_id,
                        relationship.entity_b_id,
                    )
                )
            )
            frame_key = (relationship.frame_index, pair)

            if frame_key in relationship_frame_keys:
                raise ValueError(
                    "Duplicate spatial relationship for entities "
                    f"{pair!r} in frame {relationship.frame_index}."
                )

            relationship_frame_keys.add(frame_key)

        region_frame_keys = set()

        for observation in region_observations:
            self._validate_frame(
                observation.frame_index,
                context,
                "RegionObservation",
            )

            if observation.entity_id not in entity_ids:
                raise ValueError(
                    "RegionObservation references unknown entity: "
                    f"{observation.entity_id!r}."
                )

            frame_key = (
                observation.entity_id,
                observation.region_id,
                observation.frame_index,
            )

            if frame_key in region_frame_keys:
                raise ValueError(
                    "Duplicate region observation for entity "
                    f"{observation.entity_id!r}, region "
                    f"{observation.region_id!r}, frame "
                    f"{observation.frame_index}."
                )

            region_frame_keys.add(frame_key)

    @staticmethod
    def _validate_sequence(
        values: Sequence,
        field_name: str,
        expected_type: type,
    ) -> None:
        if isinstance(values, (str, bytes)) or not isinstance(
            values,
            Sequence,
        ):
            raise TypeError(f"{field_name} must be a sequence.")

        for index, value in enumerate(values):
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"{field_name}[{index}] must be an instance of "
                    f"{expected_type.__name__}."
                )

    @staticmethod
    def _validate_frame(
        frame_index: int,
        context: ObservationContext,
        source_name: str,
    ) -> None:
        if not context.frame_start <= frame_index <= context.frame_end:
            raise ValueError(
                f"{source_name} frame_index {frame_index} is outside "
                f"the observed interval "
                f"[{context.frame_start}, {context.frame_end}]."
            )