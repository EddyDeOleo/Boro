
"""Tests for the task interpretation application service."""

from unittest.mock import Mock

import pytest

from src.application.services.task_interpretation_service import (
    TaskInterpretationResult,
    TaskInterpretationService,
)
from src.interpretation.task_definition_adapter import (
    AdaptedTaskDefinition,
    TaskDefinitionAdapterError,
)
from src.models.task_definition import (
    TaskCriterion,
    TaskDefinition,
    TaskEntity,
    TaskRegion,
)
from src.models.task_roles import TaskRoles


def make_task_definition():
    return TaskDefinition(
        task_summary="Move the red cube into the target area.",
        entities=(
            TaskEntity(
                role="object",
                description="The red cube.",
                visual_attributes={
                    "color": "red",
                    "shape": "cube",
                },
            ),
            TaskEntity(
                role="effector",
                description="The robot gripper.",
                visual_attributes={},
            ),
        ),
        regions=(
            TaskRegion(
                role="target_region",
                description="The intended destination.",
                region_type="TARGET_ZONE",
            ),
        ),
        criteria=(
            TaskCriterion(
                criterion_id="object_moves",
                description="The object moves.",
                event_type="OBJECT_MOVING",
                entity_roles=("object",),
            ),
        ),
    )


def make_adapted_task_definition():
    return AdaptedTaskDefinition(
        task_roles=TaskRoles(
            entity_roles={
                "object": "entity-1",
                "effector": "entity-2",
            },
            region_roles={
                "target_region": "region-1",
            },
        ),
        criteria=(),
    )


def make_service(
    task_definition=None,
    adapted_task_definition=None,
):
    interpreter = Mock()
    adapter = Mock()

    interpreter.interpret.return_value = (
        task_definition or make_task_definition()
    )

    adapter.adapt.return_value = (
        adapted_task_definition or make_adapted_task_definition()
    )

    service = TaskInterpretationService(
        interpreter=interpreter,
        adapter=adapter,
    )

    return service, interpreter, adapter


def test_interpret_returns_task_interpretation_result():
    task_definition = make_task_definition()
    adapted_task_definition = make_adapted_task_definition()

    service, _, _ = make_service(
        task_definition=task_definition,
        adapted_task_definition=adapted_task_definition,
    )

    result = service.interpret(
        task_instruction="Move the red cube into the target area.",
        entity_id_map={
            "object": "entity-1",
            "effector": "entity-2",
        },
        region_id_map={
            "target_region": "region-1",
        },
    )

    assert isinstance(result, TaskInterpretationResult)
    assert result.task_definition is task_definition
    assert (
        result.adapted_task_definition
        is adapted_task_definition
    )


def test_interpret_calls_interpreter_with_instruction():
    service, interpreter, _ = make_service()

    instruction = "Move the red cube into the target area."

    service.interpret(
        task_instruction=instruction,
        entity_id_map={
            "object": "entity-1",
            "effector": "entity-2",
        },
        region_id_map={
            "target_region": "region-1",
        },
    )

    interpreter.interpret.assert_called_once_with(instruction)


def test_interpret_passes_task_definition_to_adapter():
    task_definition = make_task_definition()

    service, _, adapter = make_service(
        task_definition=task_definition,
    )

    entity_id_map = {
        "object": "entity-1",
        "effector": "entity-2",
    }
    region_id_map = {
        "target_region": "region-1",
    }

    service.interpret(
        task_instruction="Move the red cube into the target area.",
        entity_id_map=entity_id_map,
        region_id_map=region_id_map,
    )

    adapter.adapt.assert_called_once_with(
        task_definition=task_definition,
        entity_id_map=entity_id_map,
        region_id_map=region_id_map,
    )


def test_interpret_preserves_original_task_definition():
    task_definition = make_task_definition()

    service, _, _ = make_service(
        task_definition=task_definition,
    )

    result = service.interpret(
        task_instruction="Move the red cube into the target area.",
        entity_id_map={
            "object": "entity-1",
            "effector": "entity-2",
        },
        region_id_map={
            "target_region": "region-1",
        },
    )

    assert result.task_definition.task_summary == (
        "Move the red cube into the target area."
    )
    assert len(result.task_definition.entities) == 2
    assert len(result.task_definition.regions) == 1
    assert len(result.task_definition.criteria) == 1


def test_interpret_does_not_call_adapter_when_interpreter_fails():
    interpreter = Mock()
    adapter = Mock()

    interpreter.interpret.side_effect = ValueError(
        "Task interpretation failed."
    )

    service = TaskInterpretationService(
        interpreter=interpreter,
        adapter=adapter,
    )

    with pytest.raises(
        ValueError,
        match="Task interpretation failed",
    ):
        service.interpret(
            task_instruction="Move the object.",
            entity_id_map={"object": "entity-1"},
            region_id_map={},
        )

    adapter.adapt.assert_not_called()


def test_interpret_propagates_adapter_errors():
    interpreter = Mock()
    adapter = Mock()

    interpreter.interpret.return_value = make_task_definition()
    adapter.adapt.side_effect = TaskDefinitionAdapterError(
        "Missing entity IDs."
    )

    service = TaskInterpretationService(
        interpreter=interpreter,
        adapter=adapter,
    )

    with pytest.raises(
        TaskDefinitionAdapterError,
        match="Missing entity IDs",
    ):
        service.interpret(
            task_instruction="Move the object.",
            entity_id_map={},
            region_id_map={},
        )


def test_interpret_rejects_invalid_interpreter_result():
    interpreter = Mock()
    adapter = Mock()

    interpreter.interpret.return_value = {
        "task_summary": "Move the object."
    }

    service = TaskInterpretationService(
        interpreter=interpreter,
        adapter=adapter,
    )

    with pytest.raises(
        TypeError,
        match="must return a TaskDefinition",
    ):
        service.interpret(
            task_instruction="Move the object.",
            entity_id_map={"object": "entity-1"},
            region_id_map={},
        )

    adapter.adapt.assert_not_called()


def test_constructor_rejects_invalid_interpreter():
    with pytest.raises(
        TypeError,
        match="interpret method",
    ):
        TaskInterpretationService(
            interpreter=object(),
            adapter=Mock(),
        )


def test_constructor_rejects_invalid_adapter():
    with pytest.raises(
        TypeError,
        match="adapt method",
    ):
        TaskInterpretationService(
            interpreter=Mock(),
            adapter=object(),
        )


def test_result_rejects_invalid_task_definition():
    with pytest.raises(
        TypeError,
        match="task_definition must be a TaskDefinition",
    ):
        TaskInterpretationResult(
            task_definition={},
            adapted_task_definition=make_adapted_task_definition(),
        )


def test_result_rejects_invalid_adapted_task_definition():
    with pytest.raises(
        TypeError,
        match="adapted_task_definition must be an AdaptedTaskDefinition",
    ):
        TaskInterpretationResult(
            task_definition=make_task_definition(),
            adapted_task_definition={},
        )
