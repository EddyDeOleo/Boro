"""REST endpoint for robotic task interpretation."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from starlette.concurrency import run_in_threadpool

from src.api.schemas.task_interpretation_request import (
    TaskInterpretationRequest,
)
from src.application.services.task_interpretation_service import (
    TaskInterpretationResult,
    TaskInterpretationService,
)
from src.interpretation.muse_spark_client import MuseSparkClientError
from src.interpretation.muse_spark_task_interpreter import (
    TaskInterpretationError,
)
from src.interpretation.task_definition_adapter import (
    TaskDefinitionAdapterError,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["task-interpretations"],
)


def get_task_interpretation_service(
    request: Request,
) -> TaskInterpretationService:
    """Retrieve the task interpretation service from application state."""

    service = getattr(
        request.app.state,
        "task_interpretation_service",
        None,
    )

    if service is None or not callable(
        getattr(service, "interpret", None)
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The task interpretation service is not available.",
        )

    return service


def _serialize_task_definition(
    result: TaskInterpretationResult,
) -> dict[str, Any]:
    """Serialize the interpreted task and verification configuration."""

    task_definition = result.task_definition
    adapted = result.adapted_task_definition

    return {
        "task_summary": task_definition.task_summary,
        "entities": [
            {
                "role": entity.role,
                "description": entity.description,
                "visual_attributes": dict(entity.visual_attributes),
                "required": entity.required,
            }
            for entity in task_definition.entities
        ],
        "regions": [
            {
                "role": region.role,
                "description": region.description,
                "region_type": region.region_type,
                "required": region.required,
            }
            for region in task_definition.regions
        ],
        "criteria": [
            {
                "criterion_id": criterion.criterion_id,
                "description": criterion.description,
                "event_type": criterion.event_type,
                "entity_roles": list(criterion.entity_roles),
                "region_role": criterion.region_role,
                "is_mandatory": criterion.is_mandatory,
            }
            for criterion in task_definition.criteria
        ],
        "unresolved_items": list(task_definition.unresolved_items),
        "entity_id_map": dict(result.entity_id_map),
        "region_id_map": dict(result.region_id_map),
        "verification_configuration": {
            "task_roles": {
                "entity_roles": dict(
                    adapted.task_roles.entity_roles
                ),
                "region_roles": dict(
                    adapted.task_roles.region_roles
                ),
            },
            "criteria": [
                {
                    "criterion_id": criterion.criterion_id,
                    "description": criterion.description,
                    "event_type": criterion.event_type,
                    "requirement": criterion.requirement.value,
                    "entity_roles": list(criterion.entity_roles),
                    "region_role": criterion.region_role,
                    "required_evidence_types": list(
                        criterion.required_evidence_types
                    ),
                    "is_mandatory": criterion.is_mandatory,
                }
                for criterion in adapted.criteria
            ],
        },
    }


@router.post(
    "/task-interpretations",
    status_code=status.HTTP_200_OK,
    summary="Interpret a robotic task instruction",
    description=(
        "Use Muse Spark to convert a natural-language robotic task "
        "instruction into a structured task definition and verification "
        "criteria."
    ),
)
async def create_task_interpretation(
    request_data: TaskInterpretationRequest,
    service: TaskInterpretationService = Depends(
        get_task_interpretation_service
    ),
) -> dict[str, Any]:
    """Interpret a task instruction using the configured Muse Spark client."""

    try:
        result = await run_in_threadpool(
            service.interpret,
            task_instruction=request_data.task_instruction,
        )

        return _serialize_task_definition(result)

    except (TaskInterpretationError, TaskDefinitionAdapterError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    except MuseSparkClientError as error:
        logger.exception(
            "Muse Spark request failed.",
            exc_info=error,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Muse Spark could not process the task instruction.",
        ) from error

    except Exception as error:
        logger.exception(
            "Task interpretation failed.",
            exc_info=error,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The task interpretation could not be completed.",
        ) from error