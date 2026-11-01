
"""Builder for explainable robotic execution reports."""

from src.models.execution_report import ExecutionReport
from src.models.pipeline_result import PipelineResult


class ExecutionReportBuilder:
    """Build an execution report from a verification pipeline result."""

    def build(self, pipeline_result: PipelineResult) -> ExecutionReport:
        """Create an execution report from a completed pipeline result."""

        if not isinstance(pipeline_result, PipelineResult):
            raise ValueError(
                "pipeline_result must be a PipelineResult."
            )

        return ExecutionReport(
            execution_timeline=pipeline_result.execution_timeline,
            verification=pipeline_result.verification,
        )
