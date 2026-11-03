
"""FastAPI application entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from src.api.routes.robotic_execution import (
    router as robotic_execution_router,
)
from src.api.routes.task_interpretation import (
    router as task_interpretation_router,
)
from src.api.routes.verification import (
    router as verification_router,
)
from src.application.services.execution_verification_service import (
    ExecutionVerificationService,
)
from src.application.services.robotic_execution_service import (
    RoboticExecutionService,
)
from src.application.services.task_interpretation_service import (
    TaskInterpretationService,
)
from src.application.services.verification_service import (
    VerificationService,
)
from src.application.services.video_preparation_service import (
    VideoPreparationService,
)
from src.interpretation.muse_spark_client import MuseSparkClient
from src.interpretation.muse_spark_task_interpreter import (
    MuseSparkTaskInterpreter,
)
from src.interpretation.task_definition_adapter import (
    TaskDefinitionAdapter,
)
from src.perception.meta_sam31_client import MetaSAM31Client
from src.perception.sam31_perception_provider_factory import (
    SAM31PerceptionProviderFactory,
)


def build_execution_verification_service() -> ExecutionVerificationService:
    """Build the production verification service and its dependencies."""

    client = MetaSAM31Client.from_environment()

    perception_provider_factory = SAM31PerceptionProviderFactory(
        client=client,
    )

    return ExecutionVerificationService(
        video_preparation_service=VideoPreparationService(),
        perception_provider_factory=perception_provider_factory,
        verification_service=VerificationService(),
    )


def build_task_interpretation_service() -> TaskInterpretationService:
    """Build the production task interpretation service."""

    client = MuseSparkClient.from_environment()

    interpreter = MuseSparkTaskInterpreter(
        client=client,
    )

    adapter = TaskDefinitionAdapter()

    return TaskInterpretationService(
        interpreter=interpreter,
        adapter=adapter,
    )


def create_app(
    execution_verification_service: Any | None = None,
    *,
    task_interpretation_service: Any | None = None,
    robotic_execution_service: Any | None = None,
    configure_meta: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Initialize production dependencies when the application starts."""

        if configure_meta:
            if (
                application.state.execution_verification_service is None
            ):
                application.state.execution_verification_service = (
                    build_execution_verification_service()
                )

            if (
                application.state.task_interpretation_service is None
            ):
                application.state.task_interpretation_service = (
                    build_task_interpretation_service()
                )

            if application.state.robotic_execution_service is None:
                application.state.robotic_execution_service = (
                    RoboticExecutionService(
                        task_interpretation_service=(
                            application.state.task_interpretation_service
                        ),
                        execution_verification_service=(
                            application.state.execution_verification_service
                        ),
                    )
                )

        yield

    application = FastAPI(
        title="Boro API",
        description=(
            "Robotic Execution Intelligence API. "
            "Verify robotic task execution through visual and "
            "temporal evidence."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    application.state.execution_verification_service = (
        execution_verification_service
    )

    application.state.task_interpretation_service = (
        task_interpretation_service
    )

    application.state.robotic_execution_service = (
        robotic_execution_service
    )

    application.include_router(verification_router)
    application.include_router(task_interpretation_router)
    application.include_router(robotic_execution_router)

    @application.get(
        "/health",
        tags=["health"],
        summary="Check API health",
    )
    async def health_check() -> dict[str, str]:
        """Return the API health status."""

        return {
            "status": "healthy",
            "service": "boro-api",
        }

    return application


app = create_app(configure_meta=True)
