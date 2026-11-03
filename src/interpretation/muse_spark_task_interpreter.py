
"""Interpret natural-language task instructions into TaskDefinition objects."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from src.models.task_definition import (
    TaskCriterion,
    TaskDefinition,
    TaskEntity,
    TaskRegion,
)


logger = logging.getLogger(__name__)


class TaskDefinitionClient(Protocol):
    """Protocol required by the task interpreter."""

    def generate_task_definition(self, task_instruction: str) -> str:
        """Generate a structured task definition as JSON text."""
        ...


class TaskInterpretationError(ValueError):
    """Raised when a model response cannot become a valid task definition."""


_TOP_LEVEL_FIELDS = {
    "task_summary",
    "entities",
    "regions",
    "criteria",
    "unresolved_items",
}

_ENTITY_FIELDS = {
    "role",
    "description",
    "visual_attributes",
    "required",
}

_REGION_FIELDS = {
    "role",
    "description",
    "region_type",
    "required",
}

_CRITERION_FIELDS = {
    "criterion_id",
    "description",
    "event_type",
    "entity_roles",
    "region_role",
    "is_mandatory",
}


def _require_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskInterpretationError(
            f"{field_name} must be a JSON object."
        )

    return value


def _reject_unknown_fields(
    value: dict[str, Any],
    allowed_fields: set[str],
    field_name: str,
) -> None:
    unknown_fields = set(value) - allowed_fields

    if unknown_fields:
        raise TaskInterpretationError(
            f"{field_name} contains unsupported fields: "
            f"{sorted(unknown_fields)}."
        )


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskInterpretationError(
            f"{field_name} must be a non-empty string."
        )

    return value.strip()


def _require_boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TaskInterpretationError(
            f"{field_name} must be a boolean."
        )

    return value


def _require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TaskInterpretationError(
            f"{field_name} must be a JSON array."
        )

    return value


def _extract_json_payload(raw_response: str) -> Any:
    """Extract and validate the complete root JSON object."""

    response = raw_response.strip()

    if not response:
        raise TaskInterpretationError(
            "Muse Spark returned an empty response."
        )

    fenced_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced_match:
        response = fenced_match.group(1).strip()

    # Locate the beginning of the root JSON object.
    start_index = response.find("{")

    if start_index == -1:
        raise TaskInterpretationError(
            "Muse Spark response does not contain a JSON object."
        )

    json_text = response[start_index:]

    try:
        payload, end_index = json.JSONDecoder().raw_decode(json_text)

    except json.JSONDecodeError as error:
        preview = response[:1000].replace("\n", "\\n")

        logger.error(
            "Muse Spark returned incomplete or invalid JSON. "
            "Line: %s, column: %s, position: %s. "
            "Response preview: %s",
            error.lineno,
            error.colno,
            error.pos,
            preview,
        )

        raise TaskInterpretationError(
            "Muse Spark returned incomplete or invalid JSON. "
            f"Parser error at line {error.lineno}, "
            f"column {error.colno}."
        ) from error

    if not isinstance(payload, dict):
        raise TaskInterpretationError(
            "Muse Spark response root must be a JSON object."
        )

    trailing_content = json_text[end_index:].strip()

    if trailing_content:
        raise TaskInterpretationError(
            "Muse Spark returned unexpected content after the JSON object."
        )

    if not _TOP_LEVEL_FIELDS.issuperset(payload.keys()):
        unknown_fields = set(payload.keys()) - _TOP_LEVEL_FIELDS

        raise TaskInterpretationError(
            "Muse Spark returned unsupported root fields: "
            f"{sorted(unknown_fields)}."
        )

    return payload


class MuseSparkTaskInterpreter:
    """Convert Muse Spark JSON responses into validated task definitions."""

    def __init__(self, client: TaskDefinitionClient) -> None:
        if not callable(
            getattr(client, "generate_task_definition", None)
        ):
            raise TypeError(
                "client must implement generate_task_definition."
            )

        self._client = client

    def interpret(self, task_instruction: str) -> TaskDefinition:
        """Interpret a task instruction and validate the resulting model."""

        if (
            not isinstance(task_instruction, str)
            or not task_instruction.strip()
        ):
            raise ValueError(
                "task_instruction must be a non-empty string."
            )

        raw_response = self._client.generate_task_definition(
            task_instruction.strip()
        )

        if not isinstance(raw_response, str) or not raw_response.strip():
            raise TaskInterpretationError(
                "The task interpretation response must be non-empty text."
            )

        payload = _extract_json_payload(raw_response)


        try:
            return self._parse_task_definition(payload)

        except TaskInterpretationError:
            raise

        except (TypeError, ValueError, KeyError) as exc:
            raise TaskInterpretationError(
                f"Invalid task definition: {exc}"
            ) from exc

    def _parse_task_definition(self, payload: Any) -> TaskDefinition:
        data = _require_object(payload, "task_definition")

        _reject_unknown_fields(
            data,
            _TOP_LEVEL_FIELDS,
            "task_definition",
        )

        required_fields = {
            "task_summary",
            "entities",
            "regions",
            "criteria",
        }

        missing_fields = required_fields - set(data)

        if missing_fields:
            raise TaskInterpretationError(
                "task_definition is missing fields: "
                f"{sorted(missing_fields)}."
            )

        entities = self._parse_entities(data["entities"])
        regions = self._parse_regions(data["regions"])
        criteria = self._parse_criteria(data["criteria"])

        unresolved_items = data.get("unresolved_items", [])

        if not isinstance(unresolved_items, list):
            raise TaskInterpretationError(
                "unresolved_items must be a JSON array."
            )

        normalized_unresolved_items = [
            _require_string(item, "unresolved_items item")
            for item in unresolved_items
        ]

        return TaskDefinition(
            task_summary=_require_string(
                data["task_summary"],
                "task_summary",
            ),
            entities=entities,
            regions=regions,
            criteria=criteria,
            unresolved_items=tuple(normalized_unresolved_items),
        )

    def _parse_entities(self, value: Any) -> tuple[TaskEntity, ...]:
        raw_entities = _require_list(value, "entities")
        entities: list[TaskEntity] = []

        for index, raw_entity in enumerate(raw_entities):
            field_name = f"entities[{index}]"
            entity = _require_object(raw_entity, field_name)

            _reject_unknown_fields(
                entity,
                _ENTITY_FIELDS,
                field_name,
            )

            required_fields = {"role", "description"}
            missing_fields = required_fields - set(entity)

            if missing_fields:
                raise TaskInterpretationError(
                    f"{field_name} is missing fields: "
                    f"{sorted(missing_fields)}."
                )

            visual_attributes = entity.get("visual_attributes", {})

            if not isinstance(visual_attributes, dict):
                raise TaskInterpretationError(
                    f"{field_name}.visual_attributes must be an object."
                )

            required = _require_boolean(
                entity.get("required", True),
                f"{field_name}.required",
            )

            entities.append(
                TaskEntity(
                    role=_require_string(
                        entity["role"],
                        f"{field_name}.role",
                    ),
                    description=_require_string(
                        entity["description"],
                        f"{field_name}.description",
                    ),
                    visual_attributes=visual_attributes,
                    required=required,
                )
            )

        return tuple(entities)

    def _parse_regions(self, value: Any) -> tuple[TaskRegion, ...]:
        raw_regions = _require_list(value, "regions")
        regions: list[TaskRegion] = []

        for index, raw_region in enumerate(raw_regions):
            field_name = f"regions[{index}]"
            region = _require_object(raw_region, field_name)

            _reject_unknown_fields(
                region,
                _REGION_FIELDS,
                field_name,
            )

            required_fields = {
                "role",
                "description",
                "region_type",
            }

            missing_fields = required_fields - set(region)

            if missing_fields:
                raise TaskInterpretationError(
                    f"{field_name} is missing fields: "
                    f"{sorted(missing_fields)}."
                )

            regions.append(
                TaskRegion(
                    role=_require_string(
                        region["role"],
                        f"{field_name}.role",
                    ),
                    description=_require_string(
                        region["description"],
                        f"{field_name}.description",
                    ),
                    region_type=_require_string(
                        region["region_type"],
                        f"{field_name}.region_type",
                    ),
                    required=_require_boolean(
                        region.get("required", True),
                        f"{field_name}.required",
                    ),
                )
            )

        return tuple(regions)

    def _parse_criteria(self, value: Any) -> tuple[TaskCriterion, ...]:
        raw_criteria = _require_list(value, "criteria")
        criteria: list[TaskCriterion] = []

        for index, raw_criterion in enumerate(raw_criteria):
            field_name = f"criteria[{index}]"
            criterion = _require_object(raw_criterion, field_name)

            _reject_unknown_fields(
                criterion,
                _CRITERION_FIELDS,
                field_name,
            )

            required_fields = {
                "criterion_id",
                "description",
                "event_type",
                "entity_roles",
            }

            missing_fields = required_fields - set(criterion)

            if missing_fields:
                raise TaskInterpretationError(
                    f"{field_name} is missing fields: "
                    f"{sorted(missing_fields)}."
                )

            entity_roles = _require_list(
                criterion["entity_roles"],
                f"{field_name}.entity_roles",
            )

            region_role = criterion.get("region_role")

            if region_role is not None:
                region_role = _require_string(
                    region_role,
                    f"{field_name}.region_role",
                )

            criteria.append(
                TaskCriterion(
                    criterion_id=_require_string(
                        criterion["criterion_id"],
                        f"{field_name}.criterion_id",
                    ),
                    description=_require_string(
                        criterion["description"],
                        f"{field_name}.description",
                    ),
                    event_type=_require_string(
                        criterion["event_type"],
                        f"{field_name}.event_type",
                    ),
                    entity_roles=tuple(
                        _require_string(
                            role,
                            f"{field_name}.entity_roles item",
                        )
                        for role in entity_roles
                    ),
                    region_role=region_role,
                    is_mandatory=_require_boolean(
                        criterion.get("is_mandatory", True),
                        f"{field_name}.is_mandatory",
                    ),
                )
            )

        return tuple(criteria)

