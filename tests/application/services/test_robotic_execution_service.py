
"""Tests for robotic execution orchestration."""

from types import SimpleNamespace

import pytest

from src.application.services.robotic_execution_service import (
    RoboticExecutionService,
    RoboticExecutionServiceError,
)
from src.models.observation_context import CoverageScope
from src.models.region import RegionDefinition, RegionType
from src.models.task_roles import TaskRoles


class FakeTaskInterpretationService:
    """Return a predefined task interpretation without calling Muse."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def interpret(self, task_instruction):
        self.calls.append(task_instruction)
        return self.result


class FakeExecutionVerificationService:
    """Capture verification inputs without running perception."""

    def __init__(self):
        self.calls = []
        self.result = object()

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


@pytest.fixture
def region():
    return RegionDefinition(
        region_id="temporary-region-id",
        name="Target zone",
        region_type=RegionType.TARGET_ZONE,
        polygon=(
            (100, 100),
            (300, 100),
            (300, 250),
            (100, 250),
        ),
        frame_width=640,
        frame_height=480,
        reference_frame=0,
    )


@pytest.fixture
def interpretation(region):
    task_roles = TaskRoles(
        entity_roles={
            "object": "entity-object",
            "effector": "entity-effector",
        },
        region_roles={
            "target_region": "region-generated-by-boro",
        },
    )

    adapted_task = SimpleNamespace(
        task_roles=task_roles,
        criteria=("criterion-object",),
    )

    return SimpleNamespace(
        task_definition=object(),
        adapted_task_definition=adapted_task,
        entity_id_map={
            "object": "entity-object",
            "effector": "entity-effector",
        },
        region_id_map={
            "target_region": "region-generated-by-boro",
        },
    )


@pytest.fixture
def service(interpretation):
    interpretation_service = FakeTaskInterpretationService(
        interpretation
    )
    execution_service = FakeExecutionVerificationService()

    robotic_service = RoboticExecutionService(
        task_interpretation_service=interpretation_service,
        execution_verification_service=execution_service,
    )

    return (
        robotic_service,
        interpretation_service,
        execution_service,
    )


@pytest.fixture
def tracking_boxes():
    return {
        "object": {
            "x1": 120,
            "y1": 140,
            "x2": 180,
            "y2": 200,
        },
        "effector": {
            "x1": 250,
            "y1": 80,
            "x2": 310,
            "y2": 140,
        },
    }


@pytest.fixture
def regions(region):
    return {
        "target_region": region,
    }


def test_execute_interprets_task_and_calls_verification(
    service,
    tracking_boxes,
    regions,
):
    robotic_service, interpretation_service, execution_service = service

    result = robotic_service.execute(
        video_path="episode.mp4",
        task_instruction="Pick up the red cube and place it in the target zone.",
        tracking_boxes_by_role=tracking_boxes,
        regions_by_role=regions,
        coverage_scope=CoverageScope.COMPLETE_TASK,
        frame_start=0,
        frame_end=99,
        time_offset_seconds=2.5,
    )

    assert result is execution_service.result

    assert interpretation_service.calls == [
        "Pick up the red cube and place it in the target zone."
    ]

    assert len(execution_service.calls) == 1

    call = execution_service.calls[0]

    assert call["video_path"] == "episode.mp4"
    assert call["perception_config"].provider == "meta"
    assert call["perception_config"].model == "sam-3.1"

    # Provider IDs are numeric strings, as required by SAM 3.1.
    # The mapping is provider ID -> Boro semantic entity ID.
    assert dict(call["perception_config"].entity_id_map) == {
        "1": "entity-effector",
        "2": "entity-object",
    }

    assert call["tracking_boxes"] == {
        "1": tracking_boxes["effector"],
        "2": tracking_boxes["object"],
    }

    assert call["task_roles"].get_entity_id("object") == "entity-object"
    assert call["task_roles"].get_entity_id("effector") == "entity-effector"
    assert (
        call["task_roles"].get_region_id("target_region")
        == "region-generated-by-boro"
    )

    assert call["criteria"] == ("criterion-object",)

    assert len(call["regions"]) == 1
    assert (
        call["regions"][0].region_id
        == "region-generated-by-boro"
    )

    # The original user-provided region remains unchanged.
    assert regions["target_region"].region_id == "temporary-region-id"

    assert call["coverage_scope"] == CoverageScope.COMPLETE_TASK
    assert call["frame_start"] == 0
    assert call["frame_end"] == 99
    assert call["time_offset_seconds"] == 2.5


def test_execute_rejects_missing_tracking_box_role(
    service,
    tracking_boxes,
    regions,
):
    robotic_service, _, execution_service = service

    incomplete_boxes = {
        "object": tracking_boxes["object"],
    }

    with pytest.raises(
        RoboticExecutionServiceError,
        match="Tracking box roles do not match",
    ):
        robotic_service.execute(
            video_path="episode.mp4",
            task_instruction="Pick up the red cube.",
            tracking_boxes_by_role=incomplete_boxes,
            regions_by_role=regions,
        )

    assert execution_service.calls == []


def test_execute_rejects_unexpected_tracking_box_role(
    service,
    tracking_boxes,
    regions,
):
    robotic_service, _, execution_service = service

    unexpected_boxes = {
        **tracking_boxes,
        "camera": {
            "x1": 10,
            "y1": 10,
            "x2": 30,
            "y2": 30,
        },
    }

    with pytest.raises(
        RoboticExecutionServiceError,
        match="Tracking box roles do not match",
    ):
        robotic_service.execute(
            video_path="episode.mp4",
            task_instruction="Pick up the red cube.",
            tracking_boxes_by_role=unexpected_boxes,
            regions_by_role=regions,
        )

    assert execution_service.calls == []


def test_execute_rejects_missing_region_role(
    service,
    tracking_boxes,
):
    robotic_service, _, execution_service = service

    with pytest.raises(
        RoboticExecutionServiceError,
        match="Region roles do not match",
    ):
        robotic_service.execute(
            video_path="episode.mp4",
            task_instruction="Pick up the red cube.",
            tracking_boxes_by_role=tracking_boxes,
            regions_by_role={},
        )

    assert execution_service.calls == []


@pytest.mark.parametrize(
    "invalid_box",
    [
        {"x1": 10, "y1": 10, "x2": 10, "y2": 20},
        {"x1": 10, "y1": 10, "x2": 20, "y2": 10},
        {"x1": -1, "y1": 10, "x2": 20, "y2": 20},
        {"x1": 10, "y1": 10, "x2": 20.5, "y2": 20},
        {"x1": 10, "y1": 10, "x2": True, "y2": 20},
        {"x1": 10, "y1": 10, "x2": 20, "y2": 20, "score": 0.9},
    ],
)
def test_execute_rejects_invalid_tracking_box_before_interpretation(
    service,
    tracking_boxes,
    regions,
    invalid_box,
):
    robotic_service, interpretation_service, execution_service = service

    invalid_boxes = {
        **tracking_boxes,
        "object": invalid_box,
    }

    with pytest.raises(ValueError, match="Tracking box"):
        robotic_service.execute(
            video_path="episode.mp4",
            task_instruction="Pick up the red cube.",
            tracking_boxes_by_role=invalid_boxes,
            regions_by_role=regions,
        )

    assert interpretation_service.calls == []
    assert execution_service.calls == []


@pytest.mark.parametrize(
    "task_instruction",
    [
        "",
        "   ",
        None,
    ],
)
def test_execute_rejects_empty_task_instruction(
    service,
    tracking_boxes,
    regions,
    task_instruction,
):
    robotic_service, interpretation_service, execution_service = service

    with pytest.raises(ValueError, match="task_instruction"):
        robotic_service.execute(
            video_path="episode.mp4",
            task_instruction=task_instruction,
            tracking_boxes_by_role=tracking_boxes,
            regions_by_role=regions,
        )

    assert interpretation_service.calls == []
    assert execution_service.calls == []


def test_execute_rejects_non_region_definition(
    service,
    tracking_boxes,
):
    robotic_service, _, execution_service = service

    with pytest.raises(TypeError, match="RegionDefinition"):
        robotic_service.execute(
            video_path="episode.mp4",
            task_instruction="Pick up the red cube.",
            tracking_boxes_by_role=tracking_boxes,
            regions_by_role={
                "target_region": {"polygon": []},
            },
        )

    assert execution_service.calls == []
