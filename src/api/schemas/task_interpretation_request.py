"""HTTP request schema for robotic task interpretation."""

from pydantic import BaseModel, ConfigDict, Field


class TaskInterpretationRequest(BaseModel):
    """Task instruction submitted for interpretation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    task_instruction: str = Field(min_length=1)