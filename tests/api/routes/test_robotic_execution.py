
"""Tests for the integrated robotic execution endpoint."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.robotic_execution import (
    router as robotic_execution_router,
)
from src.models.observation_context import CoverageScope
from src.models.region import RegionType


class FakeRoboticExecutionService:
    """Fake orchestration service for API tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)

        return {
            "status": "completed",
            "verdict": "VERIFIED",
            "evidence": [],
        }


@pytest.fixture
def robotic_execution_service() -> FakeRoboticExecutionService:
    return FakeRoboticExecutionService()


@pytest.fixture
def client(
    robotic_execution_service: FakeRoboticExecutionService,
) -> TestClient:
    application = FastAPI()

    application.state.robotic_execution_service = (
        robotic_execution_service
    )

    application.include_router(robotic_execution_router)

    return TestClient(application)


def _valid_form_data() -> dict[str, str]:
    """Build valid multipart form data."""

    return {
        "task_instruction": (
            "Pick up the red cube and place it inside the target zone."
        ),
        "tracking_boxes": json.dumps(
            {
                "object": {
                    "x1": 100,
                    "y1": 120,
                    "x2": 180,
                    "y2": 200,
                },
                "effector": {
                    "x1": 250,
                    "y1": 100,
                    "x2": 320,
                    "y2": 170,
                },
            }
        ),
        "regions": json.dumps(
            {
                "target_region": {
                    "name": "Target zone",
                    "region_type": RegionType.TARGET_ZONE.value,
                    "polygon": [
                        {"x": 300, "y": 250},
                        {"x": 450, "y": 250},
                        {"x": 450, "y": 400},
                        {"x": 300, "y": 400},
                    ],
                    "frame_width": 640,
                    "frame_height": 480,
                    "reference_frame": 0,
                    "coordinate_system": "pixel",
                }
            }
        ),
        "observation": json.dumps(
            {
                "coverage_scope": CoverageScope.COMPLETE_TASK.value,
                "frame_start": 0,
                "frame_end": 24,
                "time_offset_seconds": 0.0,
            }
        ),
    }


def _video_file() -> dict[str, tuple[str, bytes, str]]:
    return {
        "video": (
            "execution.mp4",
            b"fake-video-content",
            "video/mp4",
        )
    }


def test_create_robotic_execution_returns_report(
    client: TestClient,
    robotic_execution_service: FakeRoboticExecutionService,
) -> None:
    response = client.post(
        "/api/v1/robotic-executions",
        data=_valid_form_data(),
        files=_video_file(),
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "completed",
        "verdict": "VERIFIED",
        "evidence": [],
    }

    assert len(robotic_execution_service.calls) == 1

    call = robotic_execution_service.calls[0]

    assert call["task_instruction"] == (
        "Pick up the red cube and place it inside the target zone."
    )

    assert call["tracking_boxes_by_role"] == {
        "object": {
            "x1": 100,
            "y1": 120,
            "x2": 180,
            "y2": 200,
        },
        "effector": {
            "x1": 250,
            "y1": 100,
            "x2": 320,
            "y2": 170,
        },
    }

    assert set(call["regions_by_role"]) == {"target_region"}

    target_region = call["regions_by_role"]["target_region"]

    assert target_region.region_id == "pending-target_region"
    assert target_region.region_type is RegionType.TARGET_ZONE

    assert call["coverage_scope"] is CoverageScope.COMPLETE_TASK
    assert call["frame_start"] == 0
    assert call["frame_end"] == 24
    assert call["time_offset_seconds"] == 0.0

    assert isinstance(call["video_path"], Path)
    assert not call["video_path"].exists()


def test_create_robotic_execution_uses_default_optional_fields(
    client: TestClient,
    robotic_execution_service: FakeRoboticExecutionService,
) -> None:
    form_data = _valid_form_data()

    form_data.pop("regions")
    form_data.pop("observation")

    response = client.post(
        "/api/v1/robotic-executions",
        data=form_data,
        files=_video_file(),
    )

    assert response.status_code == 200

    call = robotic_execution_service.calls[0]

    assert call["regions_by_role"] == {}
    assert call["coverage_scope"] is CoverageScope.UNKNOWN
    assert call["frame_start"] == 0
    assert call["frame_end"] is None
    assert call["time_offset_seconds"] == 0.0


def test_create_robotic_execution_rejects_invalid_json(
    client: TestClient,
    robotic_execution_service: FakeRoboticExecutionService,
) -> None:
    form_data = _valid_form_data()
    form_data["tracking_boxes"] = "{invalid-json"

    response = client.post(
        "/api/v1/robotic-executions",
        data=form_data,
        files=_video_file(),
    )

    assert response.status_code == 422
    assert "Invalid JSON" in response.json()["detail"]
    assert robotic_execution_service.calls == []


def test_create_robotic_execution_rejects_invalid_bounding_box(
    client: TestClient,
    robotic_execution_service: FakeRoboticExecutionService,
) -> None:
    form_data = _valid_form_data()

    form_data["tracking_boxes"] = json.dumps(
        {
            "object": {
                "x1": 100,
                "y1": 120,
                "x2": 100,
                "y2": 200,
            }
        }
    )

    response = client.post(
        "/api/v1/robotic-executions",
        data=form_data,
        files=_video_file(),
    )

    assert response.status_code == 422
    assert robotic_execution_service.calls == []


def test_create_robotic_execution_rejects_empty_video(
    client: TestClient,
    robotic_execution_service: FakeRoboticExecutionService,
) -> None:
    response = client.post(
        "/api/v1/robotic-executions",
        data=_valid_form_data(),
        files={
            "video": (
                "execution.mp4",
                b"",
                "video/mp4",
            )
        },
    )

    assert response.status_code == 400
    assert robotic_execution_service.calls == []


def test_create_robotic_execution_rejects_unsupported_extension(
    client: TestClient,
    robotic_execution_service: FakeRoboticExecutionService,
) -> None:
    response = client.post(
        "/api/v1/robotic-executions",
        data=_valid_form_data(),
        files={
            "video": (
                "execution.txt",
                b"fake-video-content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415
    assert robotic_execution_service.calls == []


def test_create_robotic_execution_returns_503_when_service_is_unavailable(
    client: TestClient,
) -> None:
    client.app.state.robotic_execution_service = None

    response = client.post(
        "/api/v1/robotic-executions",
        data=_valid_form_data(),
        files=_video_file(),
    )

    assert response.status_code == 503
