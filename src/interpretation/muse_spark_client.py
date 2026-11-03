
"""Client for task interpretation through Meta Muse Spark."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


META_API_BASE_URL = "https://api.meta.ai/v1"
MUSE_SPARK_MODEL = "muse-spark-1.3"


TASK_INTERPRETATION_INSTRUCTIONS = """
You are the task interpretation component of Boro, a robotic execution
verification system.

Your only responsibility is to translate a natural-language robotic task
instruction into a structured task definition.

Return exactly one valid JSON object. Do not use Markdown or code fences.
Do not include explanations before or after the JSON.

The JSON object must follow this structure:

{
  "task_summary": "A concise description of the task.",
  "entities": [
    {
      "role": "object",
      "description": "The object manipulated by the robot.",
      "visual_attributes": {
        "color": "red",
        "shape": "cube"
      },
      "required": true
    },
    {
      "role": "effector",
      "description": "The robot gripper or end effector.",
      "visual_attributes": {},
      "required": true
    }
  ],
  "regions": [
    {
      "role": "target_region",
      "description": "The intended destination area.",
      "region_type": "TARGET_ZONE",
      "required": true
    }
  ],
  "criteria": [
    {
      "criterion_id": "object_reaches_target",
      "description": "The object enters the target region.",
      "event_type": "REGION_ENTRY",
      "entity_roles": ["object"],
      "region_role": "target_region",
      "is_mandatory": true
    }
  ],
  "unresolved_items": []
}

Supported entity roles:
- object
- effector

Supported region roles:
- target_region

Supported region types:
- TARGET_ZONE

Supported event types:
- OBJECT_MOVING
- OBJECT_MOVING_WITH_EFFECTOR
- RELATIVE_SEPARATION
- REGION_ENTRY
- STABLE_PLACEMENT

Required entity roles for each event:
- OBJECT_MOVING: object
- OBJECT_MOVING_WITH_EFFECTOR: object, effector
- RELATIVE_SEPARATION: object, effector
- REGION_ENTRY: object
- STABLE_PLACEMENT: object

Events requiring a region_role:
- REGION_ENTRY
- STABLE_PLACEMENT

Do not invent unsupported event types, roles, region types, physical properties,
or visual observations.

Do not generate pixel coordinates, polygons, bounding boxes, tracking IDs,
confidence scores, detector thresholds, evidence types, or verification
verdicts. Those are handled by Boro.

Do not claim that coordinated movement proves physical grasping.
Do not claim that visual separation proves physical release.
Do not claim that image-space overlap proves physical placement or support.

Use unresolved_items to identify information that cannot be determined from
the instruction but is necessary to define the task accurately.

If a required entity, destination, or success condition is ambiguous, do not
silently invent it. Add a concise explanation to unresolved_items.

Criteria must describe observable events supported by the listed event types.
Only create criteria that are relevant to the user's instruction.

Only create positive REQUIRED criteria. Do not create forbidden criteria,
negative criteria, or claims that an event did not occur.

Do not issue a final verification verdict. Boro's verification engine is
responsible for evaluating the evidence and determining the outcome.
"""


class MuseSparkClientError(RuntimeError):
    """Raised when a Muse Spark API request fails or returns unusable output."""


class MuseSparkClient:
    """Small client wrapper for Muse Spark task interpretation."""

    def __init__(
        self,
        client: Any,
        model: str = MUSE_SPARK_MODEL,
        max_output_tokens: int = 2500,
    ) -> None:
        if not hasattr(client, "responses"):
            raise TypeError("client must expose a responses interface.")

        if not callable(getattr(client.responses, "create", None)):
            raise TypeError(
                "client.responses must expose a callable create method."
            )

        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string.")

        if (
            not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer.")

        self._client = client
        self.model = model.strip()
        self.max_output_tokens = max_output_tokens

    @classmethod
    def from_environment(
        cls,
        model: str = MUSE_SPARK_MODEL,
        timeout_seconds: float = 60.0,
        max_output_tokens: int = 2500,
    ) -> "MuseSparkClient":
        """Build the client using the Meta Model API environment key."""

        api_key = (
            os.getenv("MODEL_API_KEY")
            or os.getenv("META_API_KEY")
        )

        if not api_key or not api_key.strip():
            raise MuseSparkClientError(
                "MODEL_API_KEY or META_API_KEY must be configured."
            )

        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive number.")

        try:
            sdk_client = OpenAI(
                base_url=META_API_BASE_URL,
                api_key=api_key.strip(),
                timeout=float(timeout_seconds),
            )
        except Exception as error:
            raise MuseSparkClientError(
                "Failed to initialize the Meta Model API client."
            ) from error

        return cls(
            client=sdk_client,
            model=model,
            max_output_tokens=max_output_tokens,
        )

    def generate_task_definition(self, task_instruction: str) -> str:
        """Request a structured task definition from Muse Spark."""

        if (
            not isinstance(task_instruction, str)
            or not task_instruction.strip()
        ):
            raise ValueError("task_instruction must be a non-empty string.")

        try:
            response = self._client.responses.create(
                model=self.model,
                instructions=TASK_INTERPRETATION_INSTRUCTIONS,
                input=task_instruction.strip(),
                max_output_tokens=self.max_output_tokens,
                store=False,
            )
        except Exception as error:
            raise MuseSparkClientError(
                "Muse Spark API request failed."
            ) from error

        output_text = getattr(response, "output_text", None)

        if not isinstance(output_text, str) or not output_text.strip():
            raise MuseSparkClientError(
                "Muse Spark returned an empty or invalid text response."
            )

        return output_text.strip()
