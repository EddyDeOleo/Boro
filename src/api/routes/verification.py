
"""REST endpoint for robotic execution verification."""

from __future__ import annotations

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

from src.api.schemas.verification_request import VerificationRequest
from src.application.services.execution_verification_service import (
    ExecutionVerificationService,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["verifications"],
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


def get_execution_verification_service(
    request: Request,
) -> ExecutionVerificationService:
    """Retrieve the verification service from application state."""

    service = getattr(
        request.app.state,
        "execution_verification_service",
        None,
    )

    if service is None or not callable(getattr(service, "execute", None)):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The verification service is not available.",
        )

    return service


async def _save_upload_to_temporary_file(
    upload: UploadFile,
) -> Path:
    """Save an uploaded video to a temporary file with size validation."""

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
            prefix="boro-upload-",
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


@router.post(
    "/verifications",
    status_code=status.HTTP_200_OK,
    summary="Verify a robotic execution video",
    description=(
        "Analyze a robotic execution video using the configured perception "
        "provider and return an evidence-based verification report."
    ),
)
async def create_verification(
    video: UploadFile = File(...),
    verification: str = Form(...),
    service: ExecutionVerificationService = Depends(
        get_execution_verification_service
    ),
) -> dict[str, Any]:
    """Execute a robotic task verification from an uploaded video."""

    temporary_video_path: Path | None = None

    try:
        try:
            request_data = VerificationRequest.model_validate_json(
                verification
            )
        except ValidationError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error.errors(),
            ) from error

        try:
            domain_config = request_data.to_domain()
        except (TypeError, ValueError, KeyError) as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error

        temporary_video_path = await _save_upload_to_temporary_file(
            video
        )

        report = await run_in_threadpool(
            service.execute,
            video_path=temporary_video_path,
            perception_config=domain_config["perception_config"],
            tracking_boxes=domain_config["tracking_boxes"],
            task_roles=domain_config["task_roles"],
            criteria=domain_config["criteria"],
            regions=domain_config["regions"],
            coverage_scope=domain_config["coverage_scope"],
            frame_start=domain_config["frame_start"],
            frame_end=domain_config["frame_end"],
            time_offset_seconds=domain_config["time_offset_seconds"],
        )

        return report

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "Verification execution failed.",
            exc_info=error,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The verification could not be completed.",
        ) from error

    finally:
        if temporary_video_path is not None:
            temporary_video_path.unlink(missing_ok=True)

        await video.close()
