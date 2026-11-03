
"""Coordinate verification and report generation."""

from typing import Any

from src.models.perception_result import PerceptionResult
from src.models.pipeline_result import PipelineResult
from src.models.task_roles import TaskRoles
from src.models.verification_criteria import VerificationCriteria
from src.presentation.report_presenter import ReportPresenter
from src.presentation.report_serializer import ReportSerializer
from src.verification.execution_report_builder import (
    ExecutionReportBuilder,
)
from src.verification.verification_pipeline import VerificationPipeline


class VerificationService:
    """Orchestrate verification and presentation of robot executions."""

    def __init__(
        self,
        pipeline: VerificationPipeline | None = None,
        report_builder: ExecutionReportBuilder | None = None,
        presenter: ReportPresenter | None = None,
        serializer: ReportSerializer | None = None,
    ) -> None:
        """Initialize the service and its dependencies."""

        self.pipeline = (
            pipeline
            if pipeline is not None
            else VerificationPipeline()
        )

        self.report_builder = (
            report_builder
            if report_builder is not None
            else ExecutionReportBuilder()
        )

        self.presenter = (
            presenter
            if presenter is not None
            else ReportPresenter()
        )

        self.serializer = (
            serializer
            if serializer is not None
            else ReportSerializer()
        )

    def execute(
        self,
        perception_result: PerceptionResult,
        task_roles: TaskRoles,
        criteria: tuple[VerificationCriteria, ...],
    ) -> dict[str, Any]:
        """Execute verification and return a serialized report."""

        pipeline_result = self._run_pipeline(
            perception_result=perception_result,
            task_roles=task_roles,
            criteria=criteria,
        )

        presentation_report = self._build_presentation_report(
            pipeline_result
        )

        return self.serializer.to_dict(presentation_report)

    def _run_pipeline(
        self,
        perception_result: PerceptionResult,
        task_roles: TaskRoles,
        criteria: tuple[VerificationCriteria, ...],
    ) -> PipelineResult:
        """Execute the domain verification pipeline."""

        return self.pipeline.run(
            perception_result=perception_result,
            task_roles=task_roles,
            criteria=criteria,
        )

    def _build_presentation_report(
        self,
        pipeline_result: PipelineResult,
    ):
        """Build and present the execution report."""

        execution_report = self.report_builder.build(
            pipeline_result
        )

        return self.presenter.present(execution_report)

