
"""Generic temporal event detection from normalized visual observations."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot
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
    """Detect temporal events from generic perception observations.

    The detector is independent of any specific perception provider.
    It receives normalized domain models rather than SAM-specific data.
    """

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
            entity_observations=entity_observations,
            spatial_relationships=spatial_relationships,
            region_observations=region_observations,
            task_roles=task_roles,
            context=context,
        )

        events = self._detect_object_moving(
            prepared=prepared,
            task_roles=task_roles,
        )

        events.extend(
            self._detect_object_moving_with_effector(
                prepared=prepared,
                task_roles=task_roles,
            )
        )

        events.extend(
            self._detect_relative_separation(
                prepared=prepared,
                task_roles=task_roles,
            )
        )

        events.extend(
            self._detect_region_entry(
                prepared=prepared,
                task_roles=task_roles,
            )
        )

        events.extend(
            self._detect_stable_placement(
                prepared=prepared,
                task_roles=task_roles,
            )
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

    def _detect_object_moving(
        self,
        prepared: _PreparedObservations,
        task_roles: TaskRoles,
    ) -> list[TemporalEvent]:
        """Detect significant net displacement of the task object."""

        if not task_roles.has_entity_role("object"):
            return []

        object_id = task_roles.get_entity_id("object")
        observations = prepared.entity_observations.get(object_id, ())

        if len(observations) < self.config.min_observations:
            return []

        segments = self._split_into_contiguous_segments(observations)

        events: list[TemporalEvent] = []

        for segment in segments:
            event = self._build_object_moving_event(
                entity_id=object_id,
                observations=segment,
            )

            if event is not None:
                events.append(event)

        return events

    def _detect_object_moving_with_effector(
        self,
        prepared: _PreparedObservations,
        task_roles: TaskRoles,
    ) -> list[TemporalEvent]:
        """Detect coordinated movement between an object and an effector."""

        if not task_roles.has_entity_role("object"):
            return []

        if not task_roles.has_entity_role("effector"):
            return []

        object_id = task_roles.get_entity_id("object")
        effector_id = task_roles.get_entity_id("effector")

        if object_id == effector_id:
            return []

        object_observations = prepared.entity_observations.get(
            object_id,
            (),
        )
        effector_observations = prepared.entity_observations.get(
            effector_id,
            (),
        )

        if (
            len(object_observations) < self.config.min_observations
            or len(effector_observations) < self.config.min_observations
        ):
            return []

        pair = tuple(sorted((object_id, effector_id)))
        relationships = prepared.spatial_relationships.get(pair, ())

        if len(relationships) < self.config.min_observations:
            return []

        object_by_frame = {
            observation.frame_index: observation
            for observation in object_observations
        }
        effector_by_frame = {
            observation.frame_index: observation
            for observation in effector_observations
        }
        relationship_by_frame = {
            relationship.frame_index: relationship
            for relationship in relationships
        }

        common_frames = sorted(
            set(object_by_frame)
            & set(effector_by_frame)
            & set(relationship_by_frame)
        )

        if not common_frames:
            return []

        close_frames = [
            frame
            for frame in common_frames
            if (
                relationship_by_frame[frame].distance_pixels
                <= self.config.max_relative_distance_pixels
            )
        ]

        if len(close_frames) < self.config.min_observations:
            return []

        segments = self._split_frame_sequences(close_frames)

        events: list[TemporalEvent] = []

        for segment_frames in segments:
            if len(segment_frames) < self.config.min_observations:
                continue

            object_segment = tuple(
                object_by_frame[frame]
                for frame in segment_frames
            )
            effector_segment = tuple(
                effector_by_frame[frame]
                for frame in segment_frames
            )
            relationship_segment = tuple(
                relationship_by_frame[frame]
                for frame in segment_frames
            )

            event = self._build_object_moving_with_effector_event(
                object_id=object_id,
                effector_id=effector_id,
                object_observations=object_segment,
                effector_observations=effector_segment,
                relationships=relationship_segment,
            )

            if event is not None:
                events.append(event)

        return events

    def _build_object_moving_with_effector_event(
        self,
        object_id: str,
        effector_id: str,
        object_observations: tuple[EntityObservation, ...],
        effector_observations: tuple[EntityObservation, ...],
        relationships: tuple[SpatialRelationship, ...],
    ) -> TemporalEvent | None:
        """Build an event when both entities move in a coordinated way."""

        if (
            len(object_observations) < self.config.min_observations
            or len(effector_observations) < self.config.min_observations
            or len(relationships) < self.config.min_observations
        ):
            return None

        if not (
            len(object_observations)
            == len(effector_observations)
            == len(relationships)
        ):
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

        similarities: list[float] = []

        for index in range(1, len(object_observations)):
            previous_object = object_observations[index - 1]
            current_object = object_observations[index]

            previous_effector = effector_observations[index - 1]
            current_effector = effector_observations[index]

            object_dx = (
                current_object.centroid[0]
                - previous_object.centroid[0]
            )
            object_dy = (
                current_object.centroid[1]
                - previous_object.centroid[1]
            )

            effector_dx = (
                current_effector.centroid[0]
                - previous_effector.centroid[0]
            )
            effector_dy = (
                current_effector.centroid[1]
                - previous_effector.centroid[1]
            )

            object_magnitude = hypot(object_dx, object_dy)
            effector_magnitude = hypot(effector_dx, effector_dy)

            if object_magnitude == 0.0 or effector_magnitude == 0.0:
                return None

            cosine_similarity = (
                object_dx * effector_dx
                + object_dy * effector_dy
            ) / (object_magnitude * effector_magnitude)

            cosine_similarity = max(
                -1.0,
                min(1.0, cosine_similarity),
            )

            similarities.append(cosine_similarity)

        if not similarities:
            return None

        mean_similarity = sum(similarities) / len(similarities)

        if mean_similarity < self.config.min_motion_similarity:
            return None

        mean_mask_distance = sum(
            relationship.distance_pixels
            for relationship in relationships
        ) / len(relationships)

        maximum_mask_distance = max(
            relationship.distance_pixels
            for relationship in relationships
        )

        event_id = (
            f"OBJECT_MOVING_WITH_EFFECTOR:{object_id}:{effector_id}:"
            f"{first_object.frame_index}:{last_object.frame_index}"
        )

        observation_ids = tuple(
            observation.observation_id
            for pair in zip(
                object_observations,
                effector_observations,
            )
            for observation in pair
        )

        evidence = Evidence(
            evidence_id=f"evidence:{event_id}",
            evidence_type="COORDINATED_ENTITY_MOTION",
            description=(
                "The object and effector moved in similar directions "
                "while their visual masks remained within the configured "
                "maximum distance."
            ),
            frame_start=first_object.frame_index,
            frame_end=last_object.frame_index,
            observation_ids=observation_ids,
            entity_ids=(object_id, effector_id),
            metrics={
                "object_displacement_pixels": object_displacement,
                "effector_displacement_pixels": effector_displacement,
                "mean_motion_similarity": mean_similarity,
                "minimum_motion_similarity": (
                    self.config.min_motion_similarity
                ),
                "mean_mask_distance_pixels": mean_mask_distance,
                "maximum_observed_mask_distance_pixels": (
                    maximum_mask_distance
                ),
                "maximum_allowed_mask_distance_pixels": (
                    self.config.max_relative_distance_pixels
                ),
                "duration_seconds": duration,
                "observation_count": len(object_observations),
                "coordinated_motion_step_count": len(similarities),
            },
            limitations=(
                "Coordinated visual movement does not prove physical "
                "contact or a successful grasp.",
                "Mask distance is a two-dimensional image-space "
                "measurement, not a physical-world distance.",
                "Centroid motion does not establish force, suction, "
                "or hidden gripper state.",
                "The event does not establish the cause of the movement.",
                "The motion similarity score is a geometric heuristic, "
                "not a calibrated probability.",
            ),
        )

        displacement_score = min(
            1.0,
            min(
                object_displacement,
                effector_displacement,
            ) / (2.0 * minimum_displacement),
        )

        proximity_score = max(
            0.0,
            1.0
            - (
                mean_mask_distance
                / self.config.max_relative_distance_pixels
            ),
        )

        confidence = (
            0.5 * mean_similarity
            + 0.25 * displacement_score
            + 0.25 * proximity_score
        )
        confidence = max(0.0, min(1.0, confidence))

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
                "mean_mask_distance_pixels": mean_mask_distance,
                "duration_seconds": duration,
                "observation_count": len(object_observations),
                "measurement": "COORDINATED_CENTROID_MOTION",
            },
        )

    def _detect_relative_separation(
        self,
        prepared: _PreparedObservations,
        task_roles: TaskRoles,
    ) -> list[TemporalEvent]:
        """Detect an increasing centroid distance between object and effector."""

        if not task_roles.has_entity_role("object"):
            return []

        if not task_roles.has_entity_role("effector"):
            return []

        object_id = task_roles.get_entity_id("object")
        effector_id = task_roles.get_entity_id("effector")

        if object_id == effector_id:
            return []

        object_observations = prepared.entity_observations.get(
            object_id,
            (),
        )
        effector_observations = prepared.entity_observations.get(
            effector_id,
            (),
        )

        if (
            len(object_observations) < self.config.min_observations
            or len(effector_observations) < self.config.min_observations
        ):
            return []

        object_by_frame = {
            observation.frame_index: observation
            for observation in object_observations
        }
        effector_by_frame = {
            observation.frame_index: observation
            for observation in effector_observations
        }

        common_frames = sorted(
            set(object_by_frame) & set(effector_by_frame)
        )

        if len(common_frames) < self.config.min_observations:
            return []

        segments = self._split_frame_sequences(common_frames)

        events: list[TemporalEvent] = []

        for segment_frames in segments:
            if len(segment_frames) < self.config.min_observations:
                continue

            object_segment = tuple(
                object_by_frame[frame]
                for frame in segment_frames
            )
            effector_segment = tuple(
                effector_by_frame[frame]
                for frame in segment_frames
            )

            event = self._build_relative_separation_event(
                object_id=object_id,
                effector_id=effector_id,
                object_observations=object_segment,
                effector_observations=effector_segment,
            )

            if event is not None:
                events.append(event)

        return events

    def _build_relative_separation_event(
        self,
        object_id: str,
        effector_id: str,
        object_observations: tuple[EntityObservation, ...],
        effector_observations: tuple[EntityObservation, ...],
    ) -> TemporalEvent | None:
        """Build a relative-separation event from paired centroid tracks."""

        if (
            len(object_observations) < self.config.min_observations
            or len(effector_observations) < self.config.min_observations
        ):
            return None

        if len(object_observations) != len(effector_observations):
            return None

        first_object = object_observations[0]
        last_object = object_observations[-1]

        duration = (
            last_object.timestamp_seconds
            - first_object.timestamp_seconds
        )

        if duration < self.config.min_event_duration_seconds:
            return None

        distances: list[float] = []

        for object_observation, effector_observation in zip(
            object_observations,
            effector_observations,
        ):
            distance = hypot(
                object_observation.centroid[0]
                - effector_observation.centroid[0],
                object_observation.centroid[1]
                - effector_observation.centroid[1],
            )
            distances.append(distance)

        if len(distances) < 2:
            return None

        initial_distance = distances[0]
        final_distance = distances[-1]
        separation_increase = final_distance - initial_distance

        minimum_increase = (
            self.config.min_relative_separation_increase_pixels
        )

        if separation_increase < minimum_increase:
            return None

        distance_changes = [
            current - previous
            for previous, current in zip(
                distances,
                distances[1:],
            )
        ]

        positive_steps = sum(
            change > 0.0
            for change in distance_changes
        )
        negative_steps = sum(
            change < 0.0
            for change in distance_changes
        )

        step_count = len(distance_changes)

        if step_count == 0:
            return None

        positive_step_ratio = positive_steps / step_count

        # More than half of all temporal steps must show increasing
        # separation. Flat steps are included in the denominator so
        # a single jump followed by a plateau cannot qualify.
        if positive_step_ratio <= 0.5:
            return None

        mean_distance = sum(distances) / len(distances)
        maximum_distance = max(distances)
        minimum_distance = min(distances)

        event_id = (
            f"RELATIVE_SEPARATION:{object_id}:{effector_id}:"
            f"{first_object.frame_index}:{last_object.frame_index}"
        )

        observation_ids = tuple(
            observation.observation_id
            for pair in zip(
                object_observations,
                effector_observations,
            )
            for observation in pair
        )

        evidence = Evidence(
            evidence_id=f"evidence:{event_id}",
            evidence_type="RELATIVE_CENTROID_SEPARATION",
            description=(
                "The centroid distance between the task object and "
                "the effector increased beyond the configured threshold "
                "with a predominantly increasing temporal trend."
            ),
            frame_start=first_object.frame_index,
            frame_end=last_object.frame_index,
            observation_ids=observation_ids,
            entity_ids=(object_id, effector_id),
            metrics={
                "initial_centroid_distance_pixels": initial_distance,
                "final_centroid_distance_pixels": final_distance,
                "separation_increase_pixels": separation_increase,
                "minimum_separation_increase_pixels": minimum_increase,
                "mean_centroid_distance_pixels": mean_distance,
                "minimum_centroid_distance_pixels": minimum_distance,
                "maximum_centroid_distance_pixels": maximum_distance,
                "positive_separation_steps": positive_steps,
                "negative_separation_steps": negative_steps,
                "total_separation_steps": step_count,
                "positive_step_ratio": positive_step_ratio,
                "duration_seconds": duration,
                "observation_count": len(distances),
            },
            limitations=(
                "Increasing centroid distance does not prove that the "
                "effector physically released the object.",
                "Relative separation is measured in two-dimensional "
                "image coordinates, not physical-world distance.",
                "Apparent separation can be affected by occlusion, "
                "camera movement, perspective, and segmentation error.",
                "The event does not establish the cause of the separation.",
                "The detector score is a geometric heuristic, "
                "not a calibrated probability.",
            ),
        )

        increase_score = min(
            1.0,
            separation_increase / (2.0 * minimum_increase),
        )

        confidence = (
            0.5 * increase_score
            + 0.5 * positive_step_ratio
        )
        confidence = max(0.0, min(1.0, confidence))

        return TemporalEvent(
            event_id=event_id,
            event_type="RELATIVE_SEPARATION",
            status=EventStatus.DETECTED,
            frame_start=first_object.frame_index,
            frame_end=last_object.frame_index,
            timestamp_start=first_object.timestamp_seconds,
            timestamp_end=last_object.timestamp_seconds,
            entity_ids=(object_id, effector_id),
            confidence=confidence,
            evidence=(evidence,),
            attributes={
                "initial_centroid_distance_pixels": initial_distance,
                "final_centroid_distance_pixels": final_distance,
                "separation_increase_pixels": separation_increase,
                "positive_step_ratio": positive_step_ratio,
                "duration_seconds": duration,
                "observation_count": len(distances),
                "measurement": "RELATIVE_CENTROID_DISTANCE",
            },
        )

    def _detect_region_entry(
        self,
        prepared: _PreparedObservations,
        task_roles: TaskRoles,
    ) -> list[TemporalEvent]:
        """Detect when the task object enters its assigned target region.

        An entry requires:
        - A valid observation below the configured overlap threshold.
        - A subsequent observation at or above the threshold.
        - At least min_observations consecutive qualifying observations.

        Region observations are paired with entity observations from the
        same entity and frame to obtain timestamps and traceable evidence.
        """

        if not task_roles.has_entity_role("object"):
            return []

        if not task_roles.has_region_role("target_region"):
            return []

        object_id = task_roles.get_entity_id("object")
        region_id = task_roles.get_region_id("target_region")

        region_observations = prepared.region_observations.get(
            (object_id, region_id),
            (),
        )

        if len(region_observations) < 2:
            return []

        entity_by_frame = {
            observation.frame_index: observation
            for observation in prepared.entity_observations.get(
                object_id,
                (),
            )
        }

        if not entity_by_frame:
            return []

        valid_records: list[
            tuple[RegionObservation, EntityObservation]
        ] = []

        record_segments: list[
            tuple[tuple[RegionObservation, EntityObservation], ...]
        ] = []

        def flush_records() -> None:
            if valid_records:
                record_segments.append(tuple(valid_records))
                valid_records.clear()

        for region_observation in region_observations:
            entity_observation = entity_by_frame.get(
                region_observation.frame_index
            )

            if (
                entity_observation is None
                or region_observation.relation == RegionRelation.UNKNOWN
                or region_observation.overlap_ratio is None
            ):
                flush_records()
                continue

            if valid_records:
                previous_region_observation = valid_records[-1][0]

                missing_frames = (
                    region_observation.frame_index
                    - previous_region_observation.frame_index
                    - 1
                )

                if missing_frames > self.config.max_frame_gap:
                    flush_records()

            valid_records.append(
                (region_observation, entity_observation)
            )

        flush_records()

        events: list[TemporalEvent] = []

        threshold = self.config.min_region_overlap_ratio
        minimum_observations = self.config.min_observations

        for record_segment in record_segments:
            if len(record_segment) < minimum_observations + 1:
                continue

            index = 1

            while index < len(record_segment):
                previous_region, _ = record_segment[index - 1]
                current_region, _ = record_segment[index]

                previous_ratio = previous_region.overlap_ratio
                current_ratio = current_region.overlap_ratio

                if previous_ratio is None or current_ratio is None:
                    index += 1
                    continue

                crossed_threshold = (
                    previous_ratio < threshold
                    and current_ratio >= threshold
                )

                if not crossed_threshold:
                    index += 1
                    continue

                confirmation_end = index

                while confirmation_end + 1 < len(record_segment):
                    next_region = record_segment[
                        confirmation_end + 1
                    ][0]

                    next_ratio = next_region.overlap_ratio

                    if next_ratio is None or next_ratio < threshold:
                        break

                    confirmation_end += 1

                confirmation_count = confirmation_end - index + 1

                if confirmation_count >= minimum_observations:
                    transition_records = record_segment[
                        index - 1:confirmation_end + 1
                    ]

                    event = self._build_region_entry_event(
                        object_id=object_id,
                        region_id=region_id,
                        transition_records=transition_records,
                        entry_index=1,
                    )

                    if event is not None:
                        events.append(event)

                    index = confirmation_end + 1
                else:
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
        """Build a REGION_ENTRY event from a confirmed threshold crossing."""

        if len(transition_records) < 2:
            return None

        if entry_index <= 0 or entry_index >= len(transition_records):
            return None

        previous_region, previous_entity = transition_records[0]
        entry_region, entry_entity = transition_records[entry_index]
        final_region, final_entity = transition_records[-1]

        previous_ratio = previous_region.overlap_ratio
        entry_ratio = entry_region.overlap_ratio

        if previous_ratio is None or entry_ratio is None:
            return None

        threshold = self.config.min_region_overlap_ratio

        if previous_ratio >= threshold:
            return None

        if entry_ratio < threshold:
            return None

        confirmation_records = transition_records[entry_index:]

        if len(confirmation_records) < self.config.min_observations:
            return None

        confirmation_ratios: list[float] = []

        for region_observation, _ in confirmation_records:
            ratio = region_observation.overlap_ratio

            if ratio is None or ratio < threshold:
                return None

            confirmation_ratios.append(float(ratio))

        mean_overlap_ratio = (
            sum(confirmation_ratios) / len(confirmation_ratios)
        )
        minimum_overlap_ratio = min(confirmation_ratios)
        maximum_overlap_ratio = max(confirmation_ratios)

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

        observation_ids = tuple(
            entity_observation.observation_id
            for _, entity_observation in transition_records
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
            observation_ids=observation_ids,
            entity_ids=(object_id,),
            metrics={
                "region_id": region_id,
                "overlap_before_entry": previous_ratio,
                "overlap_at_entry": entry_ratio,
                "mean_overlap_during_confirmation": mean_overlap_ratio,
                "minimum_overlap_during_confirmation": (
                    minimum_overlap_ratio
                ),
                "maximum_overlap_during_confirmation": (
                    maximum_overlap_ratio
                ),
                "required_overlap_ratio": threshold,
                "confirmation_observation_count": (
                    len(confirmation_records)
                ),
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
                "The event does not establish that the complete task "
                "was successful.",
            ),
        )

        confidence = max(
            0.0,
            min(1.0, mean_overlap_ratio),
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
            confidence=confidence,
            evidence=(evidence,),
            attributes={
                "region_id": region_id,
                "entry_frame": entry_entity.frame_index,
                "confirmation_end_frame": final_entity.frame_index,
                "overlap_before_entry": previous_ratio,
                "overlap_at_entry": entry_ratio,
                "mean_overlap_ratio": mean_overlap_ratio,
                "minimum_overlap_ratio": minimum_overlap_ratio,
                "maximum_overlap_ratio": maximum_overlap_ratio,
                "required_overlap_ratio": threshold,
                "confirmation_observation_count": (
                    len(confirmation_records)
                ),
                "duration_seconds": duration,
                "measurement": "REGION_MASK_OVERLAP_RATIO",
                "confidence_basis": "MEAN_CONFIRMATION_OVERLAP_RATIO",
            },
        )

    def _detect_stable_placement(
        self,
        prepared: _PreparedObservations,
        task_roles: TaskRoles,
    ) -> list[TemporalEvent]:
        """Detect an object that remains visually stable inside a region.

        Stability is evaluated from consecutive image-space centroid
        measurements. This event does not establish physical support,
        surface contact, or successful completion of the task.
        """

        if not task_roles.has_entity_role("object"):
            return []

        if not task_roles.has_region_role("target_region"):
            return []

        object_id = task_roles.get_entity_id("object")
        region_id = task_roles.get_region_id("target_region")

        region_observations = prepared.region_observations.get(
            (object_id, region_id),
            (),
        )

        if len(region_observations) < self.config.min_observations:
            return []

        entity_by_frame = {
            observation.frame_index: observation
            for observation in prepared.entity_observations.get(
                object_id,
                (),
            )
        }

        if not entity_by_frame:
            return []

        # First, build contiguous segments where region membership and
        # the matching entity observation are both available.
        in_region_segments: list[
            tuple[tuple[RegionObservation, EntityObservation], ...]
        ] = []

        current_segment: list[
            tuple[RegionObservation, EntityObservation]
        ] = []

        def flush_in_region_segment() -> None:
            if current_segment:
                in_region_segments.append(tuple(current_segment))
                current_segment.clear()

        threshold = self.config.min_region_overlap_ratio

        for region_observation in region_observations:
            entity_observation = entity_by_frame.get(
                region_observation.frame_index
            )

            if (
                entity_observation is None
                or region_observation.relation == RegionRelation.UNKNOWN
                or region_observation.overlap_ratio is None
                or region_observation.overlap_ratio < threshold
            ):
                flush_in_region_segment()
                continue

            if current_segment:
                previous_region = current_segment[-1][0]

                missing_frames = (
                    region_observation.frame_index
                    - previous_region.frame_index
                    - 1
                )

                if missing_frames > self.config.max_frame_gap:
                    flush_in_region_segment()

            current_segment.append(
                (region_observation, entity_observation)
            )

        flush_in_region_segment()

        events: list[TemporalEvent] = []

        # Within each in-region segment, split observations into stable
        # runs. Every adjacent centroid speed in a stable run must meet
        # the configured stationary-speed threshold.
        for in_region_segment in in_region_segments:
            stable_records: list[
                tuple[RegionObservation, EntityObservation]
            ] = []
            stable_step_speeds: list[float] = []

            def flush_stable_run() -> None:
                if (
                    len(stable_records)
                    >= self.config.min_observations
                ):
                    event = self._build_stable_placement_event(
                        object_id=object_id,
                        region_id=region_id,
                        stable_records=tuple(stable_records),
                        step_speeds=tuple(stable_step_speeds),
                    )

                    if event is not None:
                        events.append(event)

                stable_records.clear()
                stable_step_speeds.clear()

            for current_record in in_region_segment:
                if not stable_records:
                    stable_records.append(current_record)
                    continue

                previous_entity = stable_records[-1][1]
                current_entity = current_record[1]

                elapsed_seconds = (
                    current_entity.timestamp_seconds
                    - previous_entity.timestamp_seconds
                )

                # Non-increasing timestamps cannot establish a valid
                # temporal speed measurement. Start a fresh candidate.
                if elapsed_seconds <= 0.0:
                    flush_stable_run()
                    stable_records.append(current_record)
                    continue

                displacement = hypot(
                    current_entity.centroid[0]
                    - previous_entity.centroid[0],
                    current_entity.centroid[1]
                    - previous_entity.centroid[1],
                )

                speed = displacement / elapsed_seconds

                if (
                    speed
                    <= self.config.max_stationary_speed_pixels_per_second
                ):
                    stable_records.append(current_record)
                    stable_step_speeds.append(speed)
                else:
                    # The current observation may be the first frame
                    # after the object has settled. Start a new run here.
                    flush_stable_run()
                    stable_records.append(current_record)

            flush_stable_run()

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
        """Build a STABLE_PLACEMENT event from a stable in-region run."""

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

        if duration < self.config.min_stable_duration_seconds:
            return None

        if duration <= 0.0:
            return None

        overlap_ratios: list[float] = []

        for region_observation, _ in stable_records:
            ratio = region_observation.overlap_ratio

            if (
                ratio is None
                or ratio < self.config.min_region_overlap_ratio
            ):
                return None

            overlap_ratios.append(float(ratio))

        maximum_speed = max(step_speeds, default=0.0)

        if (
            maximum_speed
            > self.config.max_stationary_speed_pixels_per_second
        ):
            return None

        mean_speed = (
            sum(step_speeds) / len(step_speeds)
            if step_speeds
            else 0.0
        )

        mean_overlap = sum(overlap_ratios) / len(overlap_ratios)
        minimum_overlap = min(overlap_ratios)

        event_id = (
            f"STABLE_PLACEMENT:{object_id}:{region_id}:"
            f"{first_entity.frame_index}:{final_entity.frame_index}"
        )

        observation_ids = tuple(
            entity_observation.observation_id
            for _, entity_observation in stable_records
        )

        evidence = Evidence(
            evidence_id=f"evidence:{event_id}",
            evidence_type="STABLE_IN_REGION",
            description=(
                "The object remained above the configured region-overlap "
                "threshold while its centroid speed stayed within the "
                "configured stationary-speed limit for the required "
                "duration."
            ),
            frame_start=first_entity.frame_index,
            frame_end=final_entity.frame_index,
            observation_ids=observation_ids,
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

        # This heuristic combines region overlap and the maximum measured
        # speed. It is not a calibrated probability of successful placement.
        speed_limit = (
            self.config.max_stationary_speed_pixels_per_second
        )

        speed_score = max(
            0.0,
            min(1.0, 1.0 - maximum_speed / speed_limit),
        )

        confidence = (
            0.5 * mean_overlap
            + 0.5 * speed_score
        )
        confidence = max(0.0, min(1.0, confidence))

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
                "measurement": "CENTROID_STABILITY_AND_REGION_OVERLAP",
                "confidence_basis": "HEURISTIC_OVERLAP_AND_STABILITY_SCORE",
            },
        )

    def _split_into_contiguous_segments(
        self,
        observations: Sequence[EntityObservation],
    ) -> tuple[tuple[EntityObservation, ...], ...]:
        """Split observations when missing frames exceed the configured gap."""

        if not observations:
            return ()

        segments: list[list[EntityObservation]] = []
        current_segment = [observations[0]]

        for observation in observations[1:]:
            previous = current_segment[-1]

            missing_frames = (
                observation.frame_index - previous.frame_index - 1
            )

            if missing_frames > self.config.max_frame_gap:
                segments.append(current_segment)
                current_segment = []

            current_segment.append(observation)

        segments.append(current_segment)

        return tuple(
            tuple(segment)
            for segment in segments
        )

    def _split_frame_sequences(
        self,
        frame_indices: Sequence[int],
    ) -> tuple[tuple[int, ...], ...]:
        """Split frame indices when their gap exceeds the configured limit."""

        if not frame_indices:
            return ()

        segments: list[list[int]] = []
        current_segment = [frame_indices[0]]

        for frame_index in frame_indices[1:]:
            previous_frame = current_segment[-1]
            missing_frames = frame_index - previous_frame - 1

            if missing_frames > self.config.max_frame_gap:
                segments.append(current_segment)
                current_segment = []

            current_segment.append(frame_index)

        segments.append(current_segment)

        return tuple(
            tuple(segment)
            for segment in segments
        )

    def _build_object_moving_event(
        self,
        entity_id: str,
        observations: tuple[EntityObservation, ...],
    ) -> TemporalEvent | None:
        """Build an OBJECT_MOVING event when all thresholds are satisfied."""

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
                observation.observation_id
                for observation in observations
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
                "Net centroid displacement does not represent total "
                "path length.",
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

    def _prepare_observations(
        self,
        entity_observations: Sequence[EntityObservation],
        spatial_relationships: Sequence[SpatialRelationship],
        region_observations: Sequence[RegionObservation],
        task_roles: TaskRoles,
        context: ObservationContext,
    ) -> _PreparedObservations:
        """Validate, sort, and group normalized observations."""

        self._validate_inputs(
            entity_observations=entity_observations,
            spatial_relationships=spatial_relationships,
            region_observations=region_observations,
            task_roles=task_roles,
            context=context,
        )

        entities_by_id: dict[str, list[EntityObservation]] = defaultdict(list)

        relationships_by_pair: dict[
            EntityPair,
            list[SpatialRelationship],
        ] = defaultdict(list)

        regions_by_key: dict[
            RegionKey,
            list[RegionObservation],
        ] = defaultdict(list)

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

        for observations in entities_by_id.values():
            observations.sort(key=lambda item: item.frame_index)

        for relationships in relationships_by_pair.values():
            relationships.sort(key=lambda item: item.frame_index)

        for observations in regions_by_key.values():
            observations.sort(key=lambda item: item.frame_index)

        return _PreparedObservations(
            entity_observations=MappingProxyType(
                {
                    entity_id: tuple(observations)
                    for entity_id, observations in entities_by_id.items()
                }
            ),
            spatial_relationships=MappingProxyType(
                {
                    pair: tuple(relationships)
                    for pair, relationships in relationships_by_pair.items()
                }
            ),
            region_observations=MappingProxyType(
                {
                    key: tuple(observations)
                    for key, observations in regions_by_key.items()
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
        """Validate input types, frame ranges, and observation references."""

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

        entity_ids: set[str] = set()
        observation_ids: set[str] = set()
        entity_frame_keys: set[tuple[str, int]] = set()

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

        relationship_frame_keys: set[
            tuple[int, EntityPair]
        ] = set()

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
                    "SpatialRelationship must reference two distinct "
                    "entities."
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

        region_frame_keys: set[tuple[str, str, int]] = set()

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
        """Validate a sequence and all its elements."""

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
        """Ensure an observation belongs to the observed frame interval."""

        if not context.frame_start <= frame_index <= context.frame_end:
            raise ValueError(
                f"{source_name} frame_index {frame_index} is outside "
                f"the observed interval "
                f"[{context.frame_start}, {context.frame_end}]."
            )


