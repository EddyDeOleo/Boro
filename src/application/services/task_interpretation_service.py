"""Application service for interpreting robot task instructions."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

from src.interpretation.muse_spark_task_interpreter import (
    MuseSparkTaskInterpreter,
)
from src.interpretation.task_definition_adapter import (
    AdaptedTaskDefinition,
    TaskDefinitionAdapter,
)
from src.models.task_definition import TaskDefinition


@dataclass(frozen=True)
class TaskInterpretationResult:
    """Contains the interpreted task and its verification-domain mapping."""

    task_definition: TaskDefinition
    adapted_task_definition: AdaptedTaskDefinition
    entity_id_map: Mapping[str, str] = field(default_factory=dict)
    region_id_map: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_definition, TaskDefinition):
            raise TypeError("task_definition must be a TaskDefinition.")

        if not isinstance(
            self.adapted_task_definition,
            AdaptedTaskDefinition,
        ):
            raise TypeError(
                "adapted_task_definition must be an AdaptedTaskDefinition."
            )

        for name, mapping in (
            ("entity_id_map", self.entity_id_map),
            ("region_id_map", self.region_id_map),
        ):
            if not isinstance(mapping, Mapping):
                raise TypeError(f"{name} must be a mapping.")

            if any(
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(value, str)
                or not value.strip()
                for key, value in mapping.items()
            ):
                raise ValueError(
                    f"{name} keys and values must be non-empty strings."
                )

            if len(set(mapping.values())) != len(mapping):
                raise ValueError(f"{name} values must be unique.")

        object.__setattr__(
            self,
            "entity_id_map",
            MappingProxyType(dict(self.entity_id_map)),
        )
        object.__setattr__(
            self,
            "region_id_map",
            MappingProxyType(dict(self.region_id_map)),
        )


class TaskInterpretationService:
    """Coordinates task interpretation and domain adaptation."""

    def __init__(
        self,
        interpreter: MuseSparkTaskInterpreter,
        adapter: TaskDefinitionAdapter,
    ) -> None:
        if not callable(getattr(interpreter, "interpret", None)):
            raise TypeError(
                "interpreter must provide an interpret method."
            )

        if not callable(getattr(adapter, "adapt", None)):
            raise TypeError(
                "adapter must provide an adapt method."
            )

        self._interpreter = interpreter
        self._adapter = adapter

    @staticmethod
    def _generate_identifier_map(
        roles: tuple[str, ...],
        prefix: str,
    ) -> dict[str, str]:
        """Generate a unique identifier for each semantic role."""

        return {
            role: f"{prefix}-{uuid4().hex}"
            for role in roles
        }

    def interpret(
        self,
        task_instruction: str,
        entity_id_map: Mapping[str, str] | None = None,
        region_id_map: Mapping[str, str] | None = None,
    ) -> TaskInterpretationResult:
        """Interpret an instruction and adapt it for verification."""

        task_definition = self._interpreter.interpret(
            task_instruction
        )

        if not isinstance(task_definition, TaskDefinition):
            raise TypeError(
                "interpreter.interpret must return a TaskDefinition."
            )

        resolved_entity_id_map = (
            dict(entity_id_map)
            if entity_id_map is not None
            else self._generate_identifier_map(
                tuple(entity.role for entity in task_definition.entities),
                "entity",
            )
        )

        resolved_region_id_map = (
            dict(region_id_map)
            if region_id_map is not None
            else self._generate_identifier_map(
                tuple(region.role for region in task_definition.regions),
                "region",
            )
        )

        adapted_task_definition = self._adapter.adapt(
            task_definition=task_definition,
            entity_id_map=resolved_entity_id_map,
            region_id_map=resolved_region_id_map,
        )

        return TaskInterpretationResult(
            task_definition=task_definition,
            adapted_task_definition=adapted_task_definition,
            entity_id_map=resolved_entity_id_map,
            region_id_map=resolved_region_id_map,
        )