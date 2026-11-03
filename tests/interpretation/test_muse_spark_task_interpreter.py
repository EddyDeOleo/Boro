
"""Tests for Muse Spark task interpretation and response validation."""

import json

import pytest

from src.interpretation.muse_spark_task_interpreter import (
    MuseSparkTaskInterpreter,
    TaskInterpretationError,
)
from src.models.task_definition import TaskDefinition


def make_valid_payload():
    return {
        "task_summary": "Pick up the red cube and place it in the target area.",
        "entities": [
            {
                "role": "object",
                "description": "The red cube manipulated by the robot.",
                "visual_attributes": {
                    "color": "red",
                    "shape": "cube",
                },
                "required": True,
            },
            {
                "role": "effector",
                "description": "The robot gripper.",
                "visual_attributes": {},
                "required": True,
            },
        ],
        "regions": [
            {
                "role": "target_region",
                "description": "The intended destination area.",
                "region_type": "TARGET_ZONE",
                "required": True,
            }
        ],
        "criteria": [
            {
                "criterion_id": "object_moves_with_gripper",
                "description": "The object moves in coordination with the gripper.",
                "event_type": "OBJECT_MOVING_WITH_EFFECTOR",
                "entity_roles": ["object", "effector"],
                "region_role": None,
                "is_mandatory": True,
            },
            {
                "criterion_id": "object_stays_in_target",
                "description": "The object remains visually stable in the target area.",
                "event_type": "STABLE_PLACEMENT",
                "entity_roles": ["object"],
                "region_role": "target_region",
                "is_mandatory": True,
            },
        ],
        "unresolved_items": [],
    }


class FakeTaskDefinitionClient:
    def __init__(self, response):
        self.response = response
        self.instructions = []

    def generate_task_definition(self, task_instruction):
        self.instructions.append(task_instruction)
        return self.response


def make_interpreter(payload):
    client = FakeTaskDefinitionClient(json.dumps(payload))
    return MuseSparkTaskInterpreter(client), client


def test_interpret_returns_task_definition():
    payload = make_valid_payload()
    interpreter, client = make_interpreter(payload)

    result = interpreter.interpret("Pick up the red cube.")

    assert isinstance(result, TaskDefinition)
    assert result.task_summary == payload["task_summary"]
    assert len(result.entities) == 2
    assert len(result.regions) == 1
    assert len(result.criteria) == 2
    assert result.criteria[0].event_type == "OBJECT_MOVING_WITH_EFFECTOR"
    assert result.criteria[1].region_role == "target_region"
    assert client.instructions == ["Pick up the red cube."]


def test_interpret_strips_instruction():
    interpreter, client = make_interpreter(make_valid_payload())

    interpreter.interpret("  Pick up the red cube.  ")

    assert client.instructions == ["Pick up the red cube."]


@pytest.mark.parametrize("instruction", ["", " ", None, 123])
def test_interpret_rejects_invalid_instruction(instruction):
    client = FakeTaskDefinitionClient("{}")
    interpreter = MuseSparkTaskInterpreter(client)

    with pytest.raises(ValueError, match="task_instruction"):
        interpreter.interpret(instruction)

    assert client.instructions == []


@pytest.mark.parametrize("response", ["", " ", None, 123])
def test_interpret_rejects_empty_or_invalid_response(response):
    interpreter = MuseSparkTaskInterpreter(
        FakeTaskDefinitionClient(response)
    )

    with pytest.raises(TaskInterpretationError, match="non-empty text"):
        interpreter.interpret("Move the cube.")


def test_interpret_rejects_invalid_json():
    interpreter = MuseSparkTaskInterpreter(
        FakeTaskDefinitionClient("{invalid json")
    )

    with pytest.raises(TaskInterpretationError, match="invalid JSON"):
        interpreter.interpret("Move the cube.")


@pytest.mark.parametrize("payload", [None, [], "text", 123])
def test_interpret_rejects_non_object_json(payload):
    interpreter = MuseSparkTaskInterpreter(
        FakeTaskDefinitionClient(json.dumps(payload))
    )

    with pytest.raises(TaskInterpretationError, match="JSON object"):
        interpreter.interpret("Move the cube.")


def test_interpret_rejects_missing_top_level_fields():
    payload = make_valid_payload()
    del payload["criteria"]

    interpreter, _ = make_interpreter(payload)

    with pytest.raises(TaskInterpretationError, match="missing fields"):
        interpreter.interpret("Move the cube.")


def test_interpret_rejects_unknown_top_level_fields():
    payload = make_valid_payload()
    payload["verdict"] = "VERIFIED"

    interpreter, _ = make_interpreter(payload)

    with pytest.raises(TaskInterpretationError, match="unsupported fields"):
        interpreter.interpret("Move the cube.")


def test_interpret_rejects_unknown_entity_fields():
    payload = make_valid_payload()
    payload["entities"][0]["tracking_id"] = "entity-1"

    interpreter, _ = make_interpreter(payload)

    with pytest.raises(TaskInterpretationError, match="unsupported fields"):
        interpreter.interpret("Move the cube.")


def test_interpret_rejects_unknown_criterion_fields():
    payload = make_valid_payload()
    payload["criteria"][0]["required_evidence_types"] = [
        "COORDINATED_ENTITY_MOTION"
    ]

    interpreter, _ = make_interpreter(payload)

    with pytest.raises(TaskInterpretationError, match="unsupported fields"):
        interpreter.interpret("Move the cube.")


def test_interpret_rejects_unsupported_event_type():
    payload = make_valid_payload()
    payload["criteria"][0]["event_type"] = "GRASP_CONFIRMED"

    interpreter, _ = make_interpreter(payload)

    with pytest.raises(TaskInterpretationError, match="Unsupported event type"):
        interpreter.interpret("Move the cube.")


def test_interpret_rejects_unsupported_entity_role():
    payload = make_valid_payload()
    payload["entities"][0]["role"] = "camera"

    interpreter, _ = make_interpreter(payload)

    with pytest.raises(TaskInterpretationError, match="Unsupported entity role"):
        interpreter.interpret("Move the cube.")


def test_interpret_rejects_unsupported_region_type():
    payload = make_valid_payload()
    payload["regions"][0]["region_type"] = "FORBIDDEN_ZONE"

    interpreter, _ = make_interpreter(payload)

    with pytest.raises(TaskInterpretationError, match="Unsupported region type"):
        interpreter.interpret("Move the cube.")


def test_interpret_preserves_unresolved_items():
    payload = make_valid_payload()
    payload["unresolved_items"] = [
        "The target area's exact visual boundaries are not specified."
    ]

    interpreter, _ = make_interpreter(payload)

    result = interpreter.interpret("Move the cube.")

    assert result.unresolved_items == (
        "The target area's exact visual boundaries are not specified.",
    )


def test_interpret_rejects_invalid_unresolved_items():
    payload = make_valid_payload()
    payload["unresolved_items"] = ["", 123]

    interpreter, _ = make_interpreter(payload)

    with pytest.raises(TaskInterpretationError):
        interpreter.interpret("Move the cube.")


def test_interpret_rejects_non_array_entities():
    payload = make_valid_payload()
    payload["entities"] = {}

    interpreter, _ = make_interpreter(payload)

    with pytest.raises(TaskInterpretationError, match="JSON array"):
        interpreter.interpret("Move the cube.")


def test_interpret_rejects_criterion_without_required_region():
    payload = make_valid_payload()
    payload["criteria"][1]["region_role"] = None

    interpreter, _ = make_interpreter(payload)

    with pytest.raises(TaskInterpretationError, match="requires a region_role"):
        interpreter.interpret("Move the cube.")


def test_interpret_rejects_duplicate_entity_roles():
    payload = make_valid_payload()
    payload["entities"][1]["role"] = "object"

    interpreter, _ = make_interpreter(payload)

    with pytest.raises(TaskInterpretationError, match="Entity roles must be unique"):
        interpreter.interpret("Move the cube.")


def test_interpreter_requires_client_protocol():
    with pytest.raises(TypeError, match="generate_task_definition"):
        MuseSparkTaskInterpreter(object())
