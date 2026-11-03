
"""Tests for the verification application service."""

from unittest.mock import Mock, call

import pytest

from src.application.services.verification_service import (
    VerificationService,
)
from src.presentation.report_presenter import ReportPresenter
from src.presentation.report_serializer import ReportSerializer
from src.verification.execution_report_builder import (
    ExecutionReportBuilder,
)
from src.verification.verification_pipeline import VerificationPipeline


@pytest.fixture
def pipeline():
    return Mock(spec=VerificationPipeline)


@pytest.fixture
def report_builder():
    return Mock(spec=ExecutionReportBuilder)


@pytest.fixture
def presenter():
    return Mock(spec=ReportPresenter)


@pytest.fixture
def serializer():
    return Mock(spec=ReportSerializer)


@pytest.fixture
def service(pipeline, report_builder, presenter, serializer):
    return VerificationService(
        pipeline=pipeline,
        report_builder=report_builder,
        presenter=presenter,
        serializer=serializer,
    )


@pytest.fixture
def execution_data():
    perception_result = object()
    task_roles = object()
    criteria = (object(),)

    return perception_result, task_roles, criteria


def test_service_initializes_default_dependencies():
    service = VerificationService()

    assert isinstance(service.pipeline, VerificationPipeline)
    assert isinstance(service.report_builder, ExecutionReportBuilder)
    assert isinstance(service.presenter, ReportPresenter)
    assert isinstance(service.serializer, ReportSerializer)


def test_service_preserves_injected_dependencies(
    service,
    pipeline,
    report_builder,
    presenter,
    serializer,
):
    assert service.pipeline is pipeline
    assert service.report_builder is report_builder
    assert service.presenter is presenter
    assert service.serializer is serializer


def test_execute_runs_complete_report_workflow(
    service,
    pipeline,
    report_builder,
    presenter,
    serializer,
    execution_data,
):
    perception_result, task_roles, criteria = execution_data

    pipeline_result = object()
    execution_report = object()
    presentation_report = object()

    expected_output = {
        "schema_version": "1.0",
        "report": {},
    }

    pipeline.run.return_value = pipeline_result
    report_builder.build.return_value = execution_report
    presenter.present.return_value = presentation_report
    serializer.to_dict.return_value = expected_output

    result = service.execute(
        perception_result=perception_result,
        task_roles=task_roles,
        criteria=criteria,
    )

    pipeline.run.assert_called_once_with(
        perception_result=perception_result,
        task_roles=task_roles,
        criteria=criteria,
    )

    report_builder.build.assert_called_once_with(
        pipeline_result
    )

    presenter.present.assert_called_once_with(
        execution_report
    )

    serializer.to_dict.assert_called_once_with(
        presentation_report
    )

    assert result is expected_output


def test_execute_returns_serialized_report(
    service,
    pipeline,
    report_builder,
    presenter,
    serializer,
    execution_data,
):
    perception_result, task_roles, criteria = execution_data

    expected_output = {
        "schema_version": "1.0",
        "report": {
            "verdict": "INSUFFICIENT_EVIDENCE",
        },
    }

    serializer.to_dict.return_value = expected_output

    result = service.execute(
        perception_result=perception_result,
        task_roles=task_roles,
        criteria=criteria,
    )

    assert result == expected_output


def test_run_pipeline_returns_pipeline_result(
    service,
    pipeline,
    execution_data,
):
    perception_result, task_roles, criteria = execution_data
    expected_result = object()

    pipeline.run.return_value = expected_result

    result = service._run_pipeline(
        perception_result=perception_result,
        task_roles=task_roles,
        criteria=criteria,
    )

    assert result is expected_result

    pipeline.run.assert_called_once_with(
        perception_result=perception_result,
        task_roles=task_roles,
        criteria=criteria,
    )


def test_build_presentation_report_returns_presenter_result(
    service,
    report_builder,
    presenter,
):
    pipeline_result = object()
    execution_report = object()
    presentation_report = object()

    report_builder.build.return_value = execution_report
    presenter.present.return_value = presentation_report

    result = service._build_presentation_report(
        pipeline_result
    )

    report_builder.build.assert_called_once_with(
        pipeline_result
    )

    presenter.present.assert_called_once_with(
        execution_report
    )

    assert result is presentation_report


def test_execute_propagates_pipeline_exception(
    service,
    pipeline,
    report_builder,
    presenter,
    serializer,
    execution_data,
):
    perception_result, task_roles, criteria = execution_data

    pipeline.run.side_effect = RuntimeError(
        "Pipeline execution failed."
    )

    with pytest.raises(
        RuntimeError,
        match="Pipeline execution failed.",
    ):
        service.execute(
            perception_result=perception_result,
            task_roles=task_roles,
            criteria=criteria,
        )

    report_builder.build.assert_not_called()
    presenter.present.assert_not_called()
    serializer.to_dict.assert_not_called()


def test_execute_propagates_report_builder_exception(
    service,
    pipeline,
    report_builder,
    presenter,
    serializer,
    execution_data,
):
    perception_result, task_roles, criteria = execution_data

    pipeline.run.return_value = object()

    report_builder.build.side_effect = RuntimeError(
        "Report generation failed."
    )

    with pytest.raises(
        RuntimeError,
        match="Report generation failed.",
    ):
        service.execute(
            perception_result=perception_result,
            task_roles=task_roles,
            criteria=criteria,
        )

    presenter.present.assert_not_called()
    serializer.to_dict.assert_not_called()


def test_execute_propagates_presenter_exception(
    service,
    pipeline,
    report_builder,
    presenter,
    serializer,
    execution_data,
):
    perception_result, task_roles, criteria = execution_data

    pipeline.run.return_value = object()
    report_builder.build.return_value = object()

    presenter.present.side_effect = RuntimeError(
        "Report presentation failed."
    )

    with pytest.raises(
        RuntimeError,
        match="Report presentation failed.",
    ):
        service.execute(
            perception_result=perception_result,
            task_roles=task_roles,
            criteria=criteria,
        )

    serializer.to_dict.assert_not_called()


def test_execute_propagates_serializer_exception(
    service,
    pipeline,
    report_builder,
    presenter,
    serializer,
    execution_data,
):
    perception_result, task_roles, criteria = execution_data

    pipeline.run.return_value = object()
    report_builder.build.return_value = object()
    presenter.present.return_value = object()

    serializer.to_dict.side_effect = RuntimeError(
        "Report serialization failed."
    )

    with pytest.raises(
        RuntimeError,
        match="Report serialization failed.",
    ):
        service.execute(
            perception_result=perception_result,
            task_roles=task_roles,
            criteria=criteria,
        )


def test_execute_does_not_call_pipeline_more_than_once(
    service,
    pipeline,
    execution_data,
):
    perception_result, task_roles, criteria = execution_data

    pipeline.run.return_value = object()

    service.execute(
        perception_result=perception_result,
        task_roles=task_roles,
        criteria=criteria,
    )

    assert pipeline.run.call_count == 1
