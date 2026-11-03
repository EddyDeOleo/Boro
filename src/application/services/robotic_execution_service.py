
"""Orchestrate task interpretation and robotic execution verification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.application.services.execution_verification_service import (
    ExecutionVerificationService,
)
from src.application.services.task_interpretation_service import (
    TaskInterpretationService,
)
from src.models.observation_context import CoverageScope
from src.models.perception_config import PerceptionConfig
from src.models.region import RegionDefinition
from src.perception.meta_sam31_client import SAM31_MODEL


class RoboticExecutionServiceError(ValueError):
    """Raised when robotic execution inputs cannot be orchestrated."""


@dataclass(frozen=True)
class _PreparedPerceptionInputs:
    """Internal perception configuration prepared for SAM 3.1."""

    perception_config: PerceptionConfig
    tracking_boxes: Mapping[str, Mapping[str, int]]
    regions: tuple[RegionDefinition, ...]


class RoboticExecutionService:
    """Coordinate Muse Spark interpretation and SAM-based verification.

    The service translates role-based user inputs into Boro semantic IDs
    and numeric SAM provider IDs. It does not perform perception or
    verification itself.
    """

    def __init__(
        self,
        task_interpretation_service: TaskInterpretationService,
        execution_verification_service: ExecutionVerificationService,
    ) -> None:
        if task_interpretation_service is None:
            raise TypeError(
                "task_interpretation_service cannot be None."
            )

        if execution_verification_service is None:
            raise TypeError(
                "execution_verification_service cannot be None."
            )

        self._task_interpretation_service = task_interpretation_service
        self._execution_verification_service = (
            execution_verification_service
        )

    def execute(
        self,
        video_path: str | Path,
        task_instruction: str,
        tracking_boxes_by_role: Mapping[
            str,
            Mapping[str, int],
        ],
        regions_by_role: Mapping[str, RegionDefinition] | None = None,
        coverage_scope: CoverageScope = CoverageScope.UNKNOWN,
        frame_start: int = 0,
        frame_end: int | None = None,
        time_offset_seconds: float = 0.0,
    ) -> Any:
        """Interpret a task and execute its visual verification.

        Args:
            video_path: Path to the uploaded execution video.
            task_instruction: Natural-language description of the task.
            tracking_boxes_by_role: Initial SAM boxes keyed by semantic role.
                Each box uses x1, y1, x2, and y2 integer coordinates.
            regions_by_role: User-defined image regions keyed by semantic
                role, such as target_region.
            coverage_scope: Declares how much of the task the video covers.
            frame_start: First frame to analyze.
            frame_end: Last frame to analyze, inclusive.
            time_offset_seconds: Offset applied to video timestamps.

        Returns:
            The report returned by ExecutionVerificationService.
        """

        self._validate_request(
            video_path=video_path,
            task_instruction=task_instruction,
            tracking_boxes_by_role=tracking_boxes_by_role,
            regions_by_role=regions_by_role,
        )

        interpretation = (
            self._task_interpretation_service.interpret(
                task_instruction=task_instruction,
            )
        )

        prepared_inputs = self._prepare_perception_inputs(
            interpretation=interpretation,
            tracking_boxes_by_role=tracking_boxes_by_role,
            regions_by_role=regions_by_role or {},
        )

        adapted_task = interpretation.adapted_task_definition

        return self._execution_verification_service.execute(
            video_path=video_path,
            perception_config=prepared_inputs.perception_config,
            tracking_boxes=prepared_inputs.tracking_boxes,
            task_roles=adapted_task.task_roles,
            criteria=adapted_task.criteria,
            regions=prepared_inputs.regions,
            coverage_scope=coverage_scope,
            frame_start=frame_start,
            frame_end=frame_end,
            time_offset_seconds=time_offset_seconds,
        )

    @staticmethod
    def _validate_request(
        video_path: str | Path,
        task_instruction: str,
        tracking_boxes_by_role: Mapping[str, Mapping[str, int]],
        regions_by_role: Mapping[str, RegionDefinition] | None,
    ) -> None:
        """Validate role-based inputs before calling external services."""

        if not isinstance(video_path, (str, Path)):
            raise TypeError("video_path must be a string or Path.")

        if not isinstance(task_instruction, str) or not task_instruction.strip():
            raise ValueError("task_instruction cannot be empty.")

        if not isinstance(tracking_boxes_by_role, Mapping):
            raise TypeError(
                "tracking_boxes_by_role must be a mapping."
            )

        if not tracking_boxes_by_role:
            raise ValueError(
                "tracking_boxes_by_role cannot be empty."
            )

        for role, box in tracking_boxes_by_role.items():
            if not isinstance(role, str) or not role.strip():
                raise ValueError(
                    "Tracking box roles must be non-empty strings."
                )

            if not isinstance(box, Mapping):
                raise TypeError(
                    f"Tracking box for role {role!r} must be a mapping."
                )

            if set(box) != {"x1", "y1", "x2", "y2"}:
                raise ValueError(
                    f"Tracking box for role {role!r} must contain "
                    "exactly x1, y1, x2, and y2."
                )

            coordinates = tuple(box[key] for key in ("x1", "y1", "x2", "y2"))

            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in coordinates
            ):
                raise ValueError(
                    f"Tracking box coordinates for role {role!r} "
                    "must be integers."
                )

            x1, y1, x2, y2 = coordinates

            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"Tracking box for role {role!r} must have "
                    "non-negative coordinates and positive area."
                )

        if regions_by_role is not None:
            if not isinstance(regions_by_role, Mapping):
                raise TypeError("regions_by_role must be a mapping.")

            for role, region in regions_by_role.items():
                if not isinstance(role, str) or not role.strip():
                    raise ValueError(
                        "Region roles must be non-empty strings."
                    )

                if not isinstance(region, RegionDefinition):
                    raise TypeError(
                        f"Region for role {role!r} must be a "
                        "RegionDefinition."
                    )

    @staticmethod
    def _prepare_perception_inputs(
        interpretation: Any,
        tracking_boxes_by_role: Mapping[
            str,
            Mapping[str, int],
        ],
        regions_by_role: Mapping[str, RegionDefinition],
    ) -> _PreparedPerceptionInputs:
        """Translate semantic roles into SAM IDs and Boro region IDs."""

        entity_id_map = interpretation.entity_id_map
        region_id_map = interpretation.region_id_map

        if not isinstance(entity_id_map, Mapping) or not entity_id_map:
            raise RoboticExecutionServiceError(
                "Task interpretation did not produce entity mappings."
            )

        if not isinstance(region_id_map, Mapping):
            raise RoboticExecutionServiceError(
                "Task interpretation produced invalid region mappings."
            )

        expected_entity_roles = set(entity_id_map)
        supplied_entity_roles = set(tracking_boxes_by_role)

        missing_entity_roles = expected_entity_roles - supplied_entity_roles
        unexpected_entity_roles = supplied_entity_roles - expected_entity_roles

        if missing_entity_roles or unexpected_entity_roles:
            raise RoboticExecutionServiceError(
                "Tracking box roles do not match the interpreted task. "
                f"Missing: {sorted(missing_entity_roles)}. "
                f"Unexpected: {sorted(unexpected_entity_roles)}."
            )

        expected_region_roles = set(region_id_map)
        supplied_region_roles = set(regions_by_role)

        missing_region_roles = expected_region_roles - supplied_region_roles
        unexpected_region_roles = supplied_region_roles - expected_region_roles

        if missing_region_roles or unexpected_region_roles:
            raise RoboticExecutionServiceError(
                "Region roles do not match the interpreted task. "
                f"Missing: {sorted(missing_region_roles)}. "
                f"Unexpected: {sorted(unexpected_region_roles)}."
            )

        provider_to_semantic: dict[str, str] = {}
        provider_tracking_boxes: dict[str, Mapping[str, int]] = {}

        for provider_index, role in enumerate(
            sorted(entity_id_map),
            start=1,
        ):
            provider_id = str(provider_index)
            semantic_entity_id = entity_id_map[role]

            if (
                not isinstance(semantic_entity_id, str)
                or not semantic_entity_id.strip()
            ):
                raise RoboticExecutionServiceError(
                    f"Invalid semantic entity ID for role {role!r}."
                )

            provider_to_semantic[provider_id] = semantic_entity_id
            provider_tracking_boxes[provider_id] = dict(
                tracking_boxes_by_role[role]
            )

        if len(set(provider_to_semantic.values())) != len(
            provider_to_semantic
        ):
            raise RoboticExecutionServiceError(
                "Semantic entity IDs must be unique."
            )

        normalized_regions: list[RegionDefinition] = []

        for role in sorted(region_id_map):
            region_id = region_id_map[role]
            supplied_region = regions_by_role[role]

            if not isinstance(region_id, str) or not region_id.strip():
                raise RoboticExecutionServiceError(
                    f"Invalid region ID for role {role!r}."
                )

            normalized_regions.append(
                replace(
                    supplied_region,
                    region_id=region_id,
                )
            )

        perception_config = PerceptionConfig(
            provider="meta",
            model=SAM31_MODEL,
            entity_id_map=provider_to_semantic,
        )

        return _PreparedPerceptionInputs(
            perception_config=perception_config,
            tracking_boxes=provider_tracking_boxes,
            regions=tuple(normalized_regions),
        )
