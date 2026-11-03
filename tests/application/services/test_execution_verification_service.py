
"""Tests for the end-to-end execution verification service."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.application.services.execution_verification_service import (
    ExecutionVerificationService,
)
from src.application.services.verification_service import (
    VerificationService,
)
from src.application.services.video_preparation_service import (
    PreparedVideo,
    VideoPreparationService,
)
from src.models.observation_context import CoverageScope
from src.models.perception_config import PerceptionConfig
from src.models.region import RegionDefinition
from src.models.task_roles import TaskRoles
from src.models.verification_criteria import VerificationCriteria
from src.perception.perception_provider import PerceptionProvider
from src.perception.perception_provider_factory import (
    PerceptionProviderFactory,
)


@pytest.fixture
def video_path(tmp_path):
    path = tmp_path / "execution.mp4"
    path.touch()
    return path


@pytest.fixture
def context():
    return object()


@pytest.fixture
def prepared_video(video_path, context):
    prepared = Mock(spec=PreparedVideo)
    prepared.source_path = video_path
    prepared.context = context
    prepared.close = Mock()
    return prepared


@pytest.fixture
def video_preparation_service(prepared_video):
    service = Mock(spec=VideoPreparationService)
    service.prepare.return_value = prepared_video
    return service


@pytest.fixture
def perception_provider():
    return Mock(spec=PerceptionProvider)


@pytest.fixture
def perception_provider_factory(perception_provider):
    factory = Mock(spec=PerceptionProviderFactory)
    factory.create.return_value = perception_provider
    return factory


@pytest.fixture
def verification_service():
    return Mock(spec=VerificationService)


@pytest.fixture
def service(
    video_preparation_service,
    perception_provider_factory,
    verification_service,
):
    return ExecutionVerificationService(
        video_preparation_service=video_preparation_service,
        perception_provider_factory=perception_provider_factory,
        verification_service=verification_service,
    )


@pytest.fixture
def execution_data():
    config = PerceptionConfig(
        provider="meta",
        model="sam-3.1",
        entity_id_map={
            "1": "red_cube",
            "2": "gripper",
        },
    )

    tracking_boxes = {
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

    task_roles = TaskRoles(
        entity_roles={
            "object": "red_cube",
            "effector": "gripper",
        },
        region_roles={
            "target": "target_zone_01",
        },
    )

    criteria = (Mock(spec=VerificationCriteria),)
    regions = (Mock(spec=RegionDefinition),)

    return config, tracking_boxes, task_roles, criteria, regions


def test_execute_coordinates_complete_workflow(
    service,
    video_preparation_service,
    perception_provider_factory,
    perception_provider,
    verification_service,
    prepared_video,
    video_path,
    context,
    execution_data,
):
    (
        config,
        tracking_boxes,
        task_roles,
        criteria,
        regions,
    ) = execution_data

    perception_result = object()
    expected_report = {
        "schema_version": "1.0",
        "report": {},
    }

    perception_provider.perceive.return_value = perception_result
    verification_service.execute.return_value = expected_report

    result = service.execute(
        video_path=video_path,
        perception_config=config,
        tracking_boxes=tracking_boxes,
        task_roles=task_roles,
        criteria=criteria,
        regions=regions,
        coverage_scope=CoverageScope.COMPLETE_TASK,
        frame_start=0,
        frame_end=49,
        time_offset_seconds=2.5,
    )

    video_preparation_service.prepare.assert_called_once_with(
        video_path=video_path,
        coverage_scope=CoverageScope.COMPLETE_TASK,
        frame_start=0,
        frame_end=49,
        time_offset_seconds=2.5,
    )

    prepared_video.close.assert_called_once()

    perception_provider_factory.create.assert_called_once_with(
        config=config,
        tracking_boxes=tracking_boxes,
    )

    perception_provider.perceive.assert_called_once_with(
        video_path=video_path,
        context=context,
        config=config,
        regions=regions,
    )

    verification_service.execute.assert_called_once_with(
        perception_result=perception_result,
        task_roles=task_roles,
        criteria=criteria,
    )

    assert result is expected_report


def test_execute_uses_unknown_coverage_by_default(
    service,
    video_preparation_service,
    execution_data,
    video_path,
):
    (
        config,
        tracking_boxes,
        task_roles,
        criteria,
        regions,
    ) = execution_data

    service.execute(
        video_path=video_path,
        perception_config=config,
        tracking_boxes=tracking_boxes,
        task_roles=task_roles,
        criteria=criteria,
        regions=regions,
    )

    video_preparation_service.prepare.assert_called_once_with(
        video_path=video_path,
        coverage_scope=CoverageScope.UNKNOWN,
        frame_start=0,
        frame_end=None,
        time_offset_seconds=0.0,
    )


def test_execute_closes_prepared_video_before_perception(
    service,
    prepared_video,
    perception_provider_factory,
    perception_provider,
    execution_data,
    video_path,
):
    (
        config,
        tracking_boxes,
        task_roles,
        criteria,
        regions,
    ) = execution_data

    def assert_video_is_closed(**kwargs):
        prepared_video.close.assert_called_once()
        return object()

    perception_provider.perceive.side_effect = (
        assert_video_is_closed
    )

    service.execute(
        video_path=video_path,
        perception_config=config,
        tracking_boxes=tracking_boxes,
        task_roles=task_roles,
        criteria=criteria,
        regions=regions,
    )

    perception_provider_factory.create.assert_called_once()
    perception_provider.perceive.assert_called_once()


def test_execute_closes_prepared_video_when_accessing_context_fails(
    service,
    video_preparation_service,
    perception_provider_factory,
    execution_data,
    video_path,
):
    (
        config,
        tracking_boxes,
        task_roles,
        criteria,
        regions,
    ) = execution_data

    class BrokenPreparedVideo:
        source_path = video_path

        def __init__(self):
            self.closed = False

        @property
        def context(self):
            raise RuntimeError("Context access failed.")

        def close(self):
            self.closed = True

    broken_video = BrokenPreparedVideo()
    video_preparation_service.prepare.return_value = broken_video

    with pytest.raises(
        RuntimeError,
        match="Context access failed.",
    ):
        service.execute(
            video_path=video_path,
            perception_config=config,
            tracking_boxes=tracking_boxes,
            task_roles=task_roles,
            criteria=criteria,
            regions=regions,
        )

    assert broken_video.closed is True
    perception_provider_factory.create.assert_not_called()


def test_execute_does_not_run_perception_when_preparation_fails(
    service,
    video_preparation_service,
    perception_provider_factory,
    verification_service,
    execution_data,
    video_path,
):
    (
        config,
        tracking_boxes,
        task_roles,
        criteria,
        regions,
    ) = execution_data

    video_preparation_service.prepare.side_effect = RuntimeError(
        "Video preparation failed."
    )

    with pytest.raises(
        RuntimeError,
        match="Video preparation failed.",
    ):
        service.execute(
            video_path=video_path,
            perception_config=config,
            tracking_boxes=tracking_boxes,
            task_roles=task_roles,
            criteria=criteria,
            regions=regions,
        )

    perception_provider_factory.create.assert_not_called()
    verification_service.execute.assert_not_called()


def test_execute_does_not_verify_when_perception_fails(
    service,
    perception_provider,
    verification_service,
    prepared_video,
    execution_data,
    video_path,
):
    (
        config,
        tracking_boxes,
        task_roles,
        criteria,
        regions,
    ) = execution_data

    perception_provider.perceive.side_effect = RuntimeError(
        "Perception failed."
    )

    with pytest.raises(
        RuntimeError,
        match="Perception failed.",
    ):
        service.execute(
            video_path=video_path,
            perception_config=config,
            tracking_boxes=tracking_boxes,
            task_roles=task_roles,
            criteria=criteria,
            regions=regions,
        )

    prepared_video.close.assert_called_once()
    verification_service.execute.assert_not_called()


def test_execute_propagates_verification_failure(
    service,
    verification_service,
    execution_data,
    video_path,
):
    (
        config,
        tracking_boxes,
        task_roles,
        criteria,
        regions,
    ) = execution_data

    verification_service.execute.side_effect = RuntimeError(
        "Verification failed."
    )

    with pytest.raises(
        RuntimeError,
        match="Verification failed.",
    ):
        service.execute(
            video_path=video_path,
            perception_config=config,
            tracking_boxes=tracking_boxes,
            task_roles=task_roles,
            criteria=criteria,
            regions=regions,
        )


def test_service_rejects_invalid_video_preparation_service():
    with pytest.raises(
        TypeError,
        match="video_preparation_service",
    ):
        ExecutionVerificationService(
            video_preparation_service=object(),
            perception_provider_factory=Mock(
                spec=PerceptionProviderFactory
            ),
            verification_service=Mock(spec=VerificationService),
        )


def test_service_rejects_invalid_perception_provider_factory():
    with pytest.raises(
        TypeError,
        match="perception_provider_factory",
    ):
        ExecutionVerificationService(
            video_preparation_service=Mock(
                spec=VideoPreparationService
            ),
            perception_provider_factory=object(),
            verification_service=Mock(spec=VerificationService),
        )


def test_service_rejects_invalid_verification_service():
    with pytest.raises(
        TypeError,
        match="verification_service",
    ):
        ExecutionVerificationService(
            video_preparation_service=Mock(
                spec=VideoPreparationService
            ),
            perception_provider_factory=Mock(
                spec=PerceptionProviderFactory
            ),
            verification_service=object(),
        )


def test_execute_creates_provider_for_each_execution(
    service,
    perception_provider_factory,
    video_preparation_service,
    verification_service,
    video_path,
    execution_data,
):
    (
        config,
        tracking_boxes,
        task_roles,
        criteria,
        regions,
    ) = execution_data

    first_provider = Mock(spec=PerceptionProvider)
    second_provider = Mock(spec=PerceptionProvider)

    first_provider.perceive.return_value = object()
    second_provider.perceive.return_value = object()

    perception_provider_factory.create.side_effect = [
        first_provider,
        second_provider,
    ]

    service.execute(
        video_path=video_path,
        perception_config=config,
        tracking_boxes=tracking_boxes,
        task_roles=task_roles,
        criteria=criteria,
        regions=regions,
    )

    service.execute(
        video_path=video_path,
        perception_config=config,
        tracking_boxes=tracking_boxes,
        task_roles=task_roles,
        criteria=criteria,
        regions=regions,
    )

    assert perception_provider_factory.create.call_count == 2

    first_call, second_call = (
        perception_provider_factory.create.call_args_list
    )

    assert first_call.kwargs["tracking_boxes"] == tracking_boxes
    assert second_call.kwargs["tracking_boxes"] == tracking_boxes

    assert first_provider.perceive.call_count == 1
    assert second_provider.perceive.call_count == 1

    assert first_provider is not second_provider
    assert verification_service.execute.call_count == 2

