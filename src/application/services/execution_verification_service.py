
"""Orchestrate end-to-end robot execution verification."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.application.services.verification_service import (
    VerificationService,
)
from src.application.services.video_preparation_service import (
    VideoPreparationService,
)
from src.models.observation_context import CoverageScope
from src.models.perception_config import PerceptionConfig
from src.models.region import RegionDefinition
from src.models.task_roles import TaskRoles
from src.models.verification_criteria import VerificationCriteria
from src.perception.perception_provider_factory import (
    PerceptionProviderFactory,
    TrackingBoxes,
)


class ExecutionVerificationService:
    """Coordinate video preparation, perception, and verification."""

    def __init__(
        self,
        video_preparation_service: VideoPreparationService,
        perception_provider_factory: PerceptionProviderFactory,
        verification_service: VerificationService,
    ) -> None:
        """Initialize the execution verification workflow."""

        if not isinstance(
            video_preparation_service,
            VideoPreparationService,
        ):
            raise TypeError(
                "video_preparation_service must be a "
                "VideoPreparationService."
            )

        if not callable(
            getattr(perception_provider_factory, "create", None)
        ):
            raise TypeError(
                "perception_provider_factory must implement "
                "the create method."
            )

        if not isinstance(
            verification_service,
            VerificationService,
        ):
            raise TypeError(
                "verification_service must be a VerificationService."
            )

        self.video_preparation_service = video_preparation_service
        self.perception_provider_factory = perception_provider_factory
        self.verification_service = verification_service

    def execute(
        self,
        video_path: str | Path,
        perception_config: PerceptionConfig,
        tracking_boxes: TrackingBoxes,
        task_roles: TaskRoles,
        criteria: tuple[VerificationCriteria, ...],
        regions: tuple[RegionDefinition, ...] = (),
        coverage_scope: CoverageScope = CoverageScope.UNKNOWN,
        frame_start: int = 0,
        frame_end: int | None = None,
        time_offset_seconds: float = 0.0,
    ) -> dict[str, Any]:
        """Prepare a video, analyze its contents, and verify execution."""

        if not isinstance(tracking_boxes, Mapping):
            raise TypeError("tracking_boxes must be a mapping.")

        prepared_video = self.video_preparation_service.prepare(
            video_path=video_path,
            coverage_scope=coverage_scope,
            frame_start=frame_start,
            frame_end=frame_end,
            time_offset_seconds=time_offset_seconds,
        )

        try:
            source_path = prepared_video.source_path
            context = prepared_video.context
        finally:
            prepared_video.close()

        perception_provider = self.perception_provider_factory.create(
            config=perception_config,
            tracking_boxes=tracking_boxes,
        )

        perception_result = perception_provider.perceive(
            video_path=source_path,
            context=context,
            config=perception_config,
            regions=regions,
        )

        return self.verification_service.execute(
            perception_result=perception_result,
            task_roles=task_roles,
            criteria=criteria,
        )
