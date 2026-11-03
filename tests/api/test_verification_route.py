
"""Tests for the verification REST endpoint."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import src.api.routes.verification as verification_route
from src.api.main import create_app
from src.models.observation_context import CoverageScope
from src.models.perception_config import PerceptionConfig
from src.models.region import RegionDefinition, RegionType
from src.models.task_roles import TaskRoles
from src.models.verification_criteria import (
    CriterionRequirement,
    VerificationCriteria,
)


# ============================================================
# VALID REQUEST FIXTURE
# ============================================================

VALID_VERIFICATION_DATA = {
    "perception": {
        "provider": "meta",
        "model": "sam-3.1",
        "entity_id_map": {
            "1": "red_cube",
            "2": "gripper",
        },
        "include_spatial_relationships": True,
        "include_region_observations": True,
    },
    "task_roles": {
        "entity_roles": {
            "object": "red_cube",
            "effector": "gripper",
        },
        "region_roles": {
            "target_region": "target_zone_01",
        },
    },
    "tracking_boxes": {
        "1": {
            "x1": 775,
            "y1": 744,
            "x2": 817,
            "y2": 788,
        },
        "2": {
            "x1": 700,
            "y1": 660,
            "x2": 910,
            "y2": 745,
        },
    },
    "criteria": [
        {
            "criterion_id": "stable_target_placement",
            "description": "The red cube must remain stable inside the target zone.",
            "event_type": "STABLE_PLACEMENT",
            "requirement": "REQUIRED",
            "entity_roles": ["object"],
            "region_role": "target_region",
            "required_evidence_types": [],
            "is_mandatory": True,
        }
    ],
    "regions": [
        {
            "region_id": "target_zone_01",
            "name": "Target Zone",
            "region_type": "TARGET_ZONE",
            "polygon": [
                {"x": 320, "y": 721},
                {"x": 488, "y": 744},
                {"x": 434, "y": 847},
                {"x": 241, "y": 819},
            ],
            "frame_width": 1280,
            "frame_height": 960,
            "reference_frame": 0,
            "coordinate_system": "pixel",
        }
    ],
    "observation": {
        "coverage_scope": "PARTIAL_SEGMENT",
        "frame_start": 0,
        "frame_end": 50,
        "time_offset_seconds": 5.8,
    },
}

VALID_VERIFICATION_JSON = json.dumps(VALID_VERIFICATION_DATA)

FAKE_VIDEO = (
    "episode.mp4",
    b"fake-video-content",
    "video/mp4",
)


def create_test_client(service=None):
    """Create a test client with an injected verification service."""

    app = create_app(
        execution_verification_service=service
    )

    return TestClient(app)


# ============================================================
# VALID REQUEST
# ============================================================

def test_verification_endpoint_accepts_valid_request():
    service = Mock()

    expected_report = {
        "schema_version": "1.0",
        "report": {},
    }

    service.execute.return_value = expected_report

    client = create_test_client(service)

    response = client.post(
        "/api/v1/verifications",
        files={"video": FAKE_VIDEO},
        data={
            "verification": VALID_VERIFICATION_JSON,
        },
    )

    assert response.status_code == 200
    assert response.json() == expected_report

    service.execute.assert_called_once()


# ============================================================
# DOMAIN CONVERSION
# ============================================================

def test_verification_request_converts_to_domain_models():
    from src.api.schemas.verification_request import (
        VerificationRequest,
    )

    request = VerificationRequest.model_validate(
        VALID_VERIFICATION_DATA
    )

    domain_data = request.to_domain()

    assert isinstance(
        domain_data["perception_config"],
        PerceptionConfig,
    )

    assert isinstance(
        domain_data["task_roles"],
        TaskRoles,
    )

    assert isinstance(
        domain_data["criteria"],
        tuple,
    )

    assert isinstance(
        domain_data["criteria"][0],
        VerificationCriteria,
    )

    assert isinstance(
        domain_data["regions"],
        tuple,
    )

    assert isinstance(
        domain_data["regions"][0],
        RegionDefinition,
    )

    assert domain_data["perception_config"].provider == "meta"
    assert domain_data["perception_config"].model == "sam-3.1"

    assert (
        domain_data["perception_config"].entity_id_map["1"]
        == "red_cube"
    )

    assert (
        domain_data["task_roles"].entity_roles["object"]
        == "red_cube"
    )

    assert domain_data["tracking_boxes"] == {
        "1": {
            "x1": 775,
            "y1": 744,
            "x2": 817,
            "y2": 788,
        },
        "2": {
            "x1": 700,
            "y1": 660,
            "x2": 910,
            "y2": 745,
        },
    }

    assert (
        domain_data["criteria"][0].requirement
        == CriterionRequirement.REQUIRED
    )

    assert (
        domain_data["regions"][0].region_type
        == RegionType.TARGET_ZONE
    )

    assert (
        domain_data["coverage_scope"]
        == CoverageScope.PARTIAL_SEGMENT
    )

    assert domain_data["frame_start"] == 0
    assert domain_data["frame_end"] == 50
    assert domain_data["time_offset_seconds"] == 5.8


# ============================================================
# SERVICE ARGUMENTS
# ============================================================

def test_endpoint_passes_expected_arguments_to_service():
    service = Mock()

    expected_report = {
        "schema_version": "1.0",
        "report": {},
    }

    execution_state = {}

    def execute_side_effect(**kwargs):
        video_path = kwargs["video_path"]

        execution_state["video_path"] = video_path
        execution_state["video_exists_during_execution"] = (
            video_path.exists()
        )

        return expected_report

    service.execute.side_effect = execute_side_effect

    client = create_test_client(service)

    response = client.post(
        "/api/v1/verifications",
        files={"video": FAKE_VIDEO},
        data={
            "verification": VALID_VERIFICATION_JSON,
        },
    )

    assert response.status_code == 200
    assert response.json() == expected_report

    service.execute.assert_called_once()

    arguments = service.execute.call_args.kwargs

    assert isinstance(arguments["video_path"], Path)

    # The file must exist while the service is executing.
    assert execution_state["video_exists_during_execution"] is True

    # The file must be deleted after the request finishes.
    assert not execution_state["video_path"].exists()

    assert isinstance(
        arguments["perception_config"],
        PerceptionConfig,
    )

    assert isinstance(
        arguments["task_roles"],
        TaskRoles,
    )

    assert arguments["tracking_boxes"] == {
        "1": {
            "x1": 775,
            "y1": 744,
            "x2": 817,
            "y2": 788,
        },
        "2": {
            "x1": 700,
            "y1": 660,
            "x2": 910,
            "y2": 745,
        },
    }

    assert len(arguments["criteria"]) == 1
    assert len(arguments["regions"]) == 1

    assert (
        arguments["coverage_scope"]
        == CoverageScope.PARTIAL_SEGMENT
    )

    assert arguments["frame_start"] == 0
    assert arguments["frame_end"] == 50
    assert arguments["time_offset_seconds"] == 5.8


# ============================================================
# TRACKING BOX VALIDATION
# ============================================================

def test_verification_request_rejects_missing_tracking_box():
    from src.api.schemas.verification_request import (
        VerificationRequest,
    )
    from pydantic import ValidationError

    invalid_data = {
        **VALID_VERIFICATION_DATA,
        "tracking_boxes": {
            "1": {
                "x1": 775,
                "y1": 744,
                "x2": 817,
                "y2": 788,
            }
        },
    }

    with pytest.raises(ValidationError):
        VerificationRequest.model_validate(invalid_data)


def test_verification_request_rejects_inverted_tracking_box():
    from src.api.schemas.verification_request import (
        VerificationRequest,
    )
    from pydantic import ValidationError

    invalid_data = {
        **VALID_VERIFICATION_DATA,
        "tracking_boxes": {
            **VALID_VERIFICATION_DATA["tracking_boxes"],
            "1": {
                "x1": 817,
                "y1": 744,
                "x2": 775,
                "y2": 788,
            },
        },
    }

    with pytest.raises(ValidationError):
        VerificationRequest.model_validate(invalid_data)


def test_verification_request_rejects_negative_tracking_coordinates():
    from src.api.schemas.verification_request import (
        VerificationRequest,
    )
    from pydantic import ValidationError

    invalid_data = {
        **VALID_VERIFICATION_DATA,
        "tracking_boxes": {
            **VALID_VERIFICATION_DATA["tracking_boxes"],
            "1": {
                "x1": -1,
                "y1": 744,
                "x2": 817,
                "y2": 788,
            },
        },
    }

    with pytest.raises(ValidationError):
        VerificationRequest.model_validate(invalid_data)


# ============================================================
# TEMPORARY FILE CLEANUP
# ============================================================

@pytest.mark.parametrize(
    "should_fail",
    [False, True],
    ids=["successful-execution", "failed-execution"],
)
def test_temporary_video_is_deleted_after_execution(
    monkeypatch,
    should_fail,
):
    service = Mock()

    if should_fail:
        service.execute.side_effect = RuntimeError(
            "Simulated execution failure"
        )
    else:
        service.execute.return_value = {
            "schema_version": "1.0",
            "report": {},
        }

    client = create_test_client(service)

    saved_paths = []

    original_save = (
        verification_route._save_upload_to_temporary_file
    )

    async def tracked_save(upload):
        path = await original_save(upload)
        saved_paths.append(path)
        return path

    monkeypatch.setattr(
        verification_route,
        "_save_upload_to_temporary_file",
        tracked_save,
    )

    response = client.post(
        "/api/v1/verifications",
        files={"video": FAKE_VIDEO},
        data={
            "verification": VALID_VERIFICATION_JSON,
        },
    )

    assert len(saved_paths) == 1
    assert not saved_paths[0].exists()

    if should_fail:
        assert response.status_code == 500
    else:
        assert response.status_code == 200


# ============================================================
# INVALID REQUESTS
# ============================================================

def test_endpoint_rejects_unsupported_video_extension():
    service = Mock()
    client = create_test_client(service)

    response = client.post(
        "/api/v1/verifications",
        files={
            "video": (
                "document.txt",
                b"invalid-video",
                "text/plain",
            )
        },
        data={
            "verification": VALID_VERIFICATION_JSON,
        },
    )

    assert response.status_code == 415
    service.execute.assert_not_called()


def test_endpoint_rejects_empty_video():
    service = Mock()
    client = create_test_client(service)

    response = client.post(
        "/api/v1/verifications",
        files={
            "video": (
                "episode.mp4",
                b"",
                "video/mp4",
            )
        },
        data={
            "verification": VALID_VERIFICATION_JSON,
        },
    )

    assert response.status_code == 400
    service.execute.assert_not_called()


def test_endpoint_rejects_malformed_json():
    service = Mock()
    client = create_test_client(service)

    response = client.post(
        "/api/v1/verifications",
        files={"video": FAKE_VIDEO},
        data={
            "verification": "{invalid-json",
        },
    )

    assert response.status_code == 422
    service.execute.assert_not_called()


def test_endpoint_rejects_invalid_configuration():
    service = Mock()
    client = create_test_client(service)

    invalid_data = {
        **VALID_VERIFICATION_DATA,
        "criteria": [],
    }

    response = client.post(
        "/api/v1/verifications",
        files={"video": FAKE_VIDEO},
        data={
            "verification": json.dumps(invalid_data),
        },
    )

    assert response.status_code == 422
    service.execute.assert_not_called()


def test_endpoint_rejects_missing_verification_field():
    service = Mock()
    client = create_test_client(service)

    response = client.post(
        "/api/v1/verifications",
        files={"video": FAKE_VIDEO},
    )

    assert response.status_code == 422
    service.execute.assert_not_called()


# ============================================================
# SERVICE AVAILABILITY
# ============================================================

def test_endpoint_returns_503_when_service_is_unavailable():
    client = create_test_client()

    response = client.post(
        "/api/v1/verifications",
        files={"video": FAKE_VIDEO},
        data={
            "verification": VALID_VERIFICATION_JSON,
        },
    )

    assert response.status_code == 503
