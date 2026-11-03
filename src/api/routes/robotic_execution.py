
"""REST endpoint for integrated robotic execution."""

from __future__ import annotations

import json
import logging

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from src.api.schemas.robotic_execution_request import (
    RoboticExecutionRequest,
)
from src.application.services.robotic_execution_service import (
    RoboticExecutionService,
    RoboticExecutionServiceError,
)
from src.interpretation.muse_spark_client import (
    MuseSparkClientError,
)
from src.interpretation.muse_spark_task_interpreter import (
    TaskInterpretationError,
)
from src.interpretation.task_definition_adapter import (
    TaskDefinitionAdapterError,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["robotic-executions"],
)

MAX_VIDEO_SIZE_BYTES = 500 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}


def get_robotic_execution_service(
    request: Request,
) -> RoboticExecutionService:
    """Retrieve the orchestration service from application state."""

    service = getattr(
        request.app.state,
        "robotic_execution_service",
        None,
    )

    if service is None or not callable(
        getattr(service, "execute", None)
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The robotic execution service is not available.",
        )

    return service


async def _save_upload_to_temporary_file(
    upload: UploadFile,
) -> Path:
    """Save an uploaded video with extension and size validation."""

    if not upload.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A video filename is required.",
        )

    extension = Path(upload.filename).suffix.lower()

    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="The uploaded file has an unsupported video extension.",
        )

    temporary_path: Path | None = None
    total_size = 0

    try:
        with NamedTemporaryFile(
            mode="wb",
            suffix=extension,
            prefix="boro-robotic-execution-",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
                total_size += len(chunk)

                if total_size > MAX_VIDEO_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="The uploaded video exceeds the maximum size.",
                    )

                temporary_file.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded video is empty.",
            )

        return temporary_path

    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

        raise


def _parse_request_data(
    task_instruction: str,
    tracking_boxes: str,
    regions: str,
    observation: str,
) -> RoboticExecutionRequest:
    """Parse and validate multipart JSON fields."""

    try:
        payload = {
            "task_instruction": task_instruction,
            "tracking_boxes": json.loads(tracking_boxes),
            "regions": json.loads(regions),
            "observation": json.loads(observation),
        }

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON in multipart form data: {error.msg}.",
        ) from error

    try:
        return RoboticExecutionRequest.model_validate(payload)

    except ValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=error.errors(
                include_context=False,
                include_url=False,
            ),
        ) from error

@router.post(
    "/robotic-executions",
    status_code=status.HTTP_200_OK,
    summary="Execute integrated robotic task verification",
    description=(
        "Interpret a natural-language robotic task with Muse Spark, "
        "analyze the uploaded video with SAM 3.1, and return an "
        "evidence-based execution verification report."
    ),
)
async def create_robotic_execution(
    video: UploadFile = File(...),
    task_instruction: str = Form(...),
    tracking_boxes: str = Form(...),
    regions: str = Form("{}"),
    observation: str = Form("{}"),
    service: RoboticExecutionService = Depends(
        get_robotic_execution_service
    ),
) -> dict[str, Any]:
    """Execute the complete robotic verification workflow."""

    temporary_video_path: Path | None = None

    try:
        request_data = _parse_request_data(
            task_instruction=task_instruction,
            tracking_boxes=tracking_boxes,
            regions=regions,
            observation=observation,
        )

        domain_config = request_data.to_domain()

        temporary_video_path = await _save_upload_to_temporary_file(
            video
        )

        report = await run_in_threadpool(
            service.execute,
            video_path=temporary_video_path,
            **domain_config,
        )

        return report

    except HTTPException:
        raise

    except (
        TaskInterpretationError,
        TaskDefinitionAdapterError,
        RoboticExecutionServiceError,
    ) as error:
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
            "Integrated robotic execution failed.",
            exc_info=error,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The robotic execution could not be completed.",
        ) from error

    finally:
        if temporary_video_path is not None:
            temporary_video_path.unlink(missing_ok=True)

        await video.close()
