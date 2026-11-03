
"""Tests for structured robotic task definitions."""

from dataclasses import FrozenInstanceError

import pytest

from src.models.task_definition import (
    SupportedEventType,
    TaskCriterion,
    TaskDefinition,
    TaskEntity,
    TaskEntityRole,
    TaskRegion,
    TaskRegionRole,
)


def make_object() -> TaskEntity:
    return TaskEntity(
        role=TaskEntityRole.OBJECT.value,
        description="The red cube",
        visual_attributes={
            "color": "red",
            "shape": "cube",
        },
    )


def make_effector() -> TaskEntity:
    return TaskEntity(
        role=TaskEntityRole.EFFECTOR.value,
        description="The robot gripper",
        visual_attributes={},
    )


def make_target_region() -> TaskRegion:
    return TaskRegion(
        role=TaskRegionRole.TARGET_REGION.value,
        description="The designated target zone",
        region_type="TARGET_ZONE",
    )


def make_stable_placement_criterion() -> TaskCriterion:
    return TaskCriterion(
        criterion_id="stable_placement",
        description="The object remains stable inside the target zone.",
        event_type=SupportedEventType.STABLE_PLACEMENT.value,
        entity_roles=("object",),
        region_role="target_region",
    )


def make_task_definition() -> TaskDefinition:
    return TaskDefinition(
        task_summary=(
            "Move the red cube into the target zone "
            "and leave it stable."
        ),
        entities=(make_object(), make_effector()),
        regions=(make_target_region(),),
        criteria=(
            TaskCriterion(
                criterion_id="object_movement",
                description=(
                    "The object moves in coordination "
                    "with the robot gripper."
                ),
                event_type=(
                    SupportedEventType.OBJECT_MOVING_WITH_EFFECTOR.value
                ),
                entity_roles=("object", "effector"),
            ),
            TaskCriterion(
                criterion_id="target_entry",
                description="The object enters the target zone.",
                event_type=SupportedEventType.REGION_ENTRY.value,
                entity_roles=("object",),
                region_role="target_region",
            ),
            make_stable_placement_criterion(),
        ),
    )


class TestTaskEntity:
    def test_create_valid_entity(self):
        entity = make_object()

        assert entity.role == "object"
        assert entity.description == "The red cube"
        assert entity.visual_attributes == {
            "color": "red",
            "shape": "cube",
        }
        assert entity.required is True

    def test_create_entity_without_visual_attributes(self):
        entity = make_effector()

        assert entity.visual_attributes == {}

    @pytest.mark.parametrize(
        "role",
        ["robot", "camera", "target_region", "", None],
    )
    def test_reject_unsupported_role(self, role):
        with pytest.raises(ValueError, match="Unsupported entity role"):
            TaskEntity(
                role=role,
                description="An entity",
                visual_attributes={},
            )

    @pytest.mark.parametrize(
        "description",
        ["", "   ", None],
    )
    def test_reject_invalid_description(self, description):
        with pytest.raises(ValueError, match="description"):
            TaskEntity(
                role="object",
                description=description,
                visual_attributes={},
            )

    def test_reject_invalid_visual_attributes(self):
        with pytest.raises(ValueError, match="visual_attributes"):
            TaskEntity(
                role="object",
                description="The red cube",
                visual_attributes="red",
            )

    def test_reject_empty_visual_attribute_key(self):
        with pytest.raises(ValueError, match="visual attribute key"):
            TaskEntity(
                role="object",
                description="The red cube",
                visual_attributes={"": "red"},
            )

    def test_reject_empty_visual_attribute_value(self):
        with pytest.raises(ValueError, match="visual attribute value"):
            TaskEntity(
                role="object",
                description="The red cube",
                visual_attributes={"color": ""},
            )

    def test_reject_non_boolean_required(self):
        with pytest.raises(ValueError, match="required"):
            TaskEntity(
                role="object",
                description="The red cube",
                visual_attributes={},
                required="yes",
            )

    def test_visual_attributes_are_immutable(self):
        entity = make_object()

        with pytest.raises(TypeError):
            entity.visual_attributes["color"] = "blue"

    def test_entity_is_frozen(self):
        entity = make_object()

        with pytest.raises(FrozenInstanceError):
            entity.description = "Another object"


class TestTaskRegion:
    def test_create_valid_target_region(self):
        region = make_target_region()

        assert region.role == "target_region"
        assert region.description == "The designated target zone"
        assert region.region_type == "TARGET_ZONE"
        assert region.required is True

    def test_reject_unsupported_region_role(self):
        with pytest.raises(ValueError, match="Unsupported region role"):
            TaskRegion(
                role="forbidden_region",
                description="A forbidden zone",
                region_type="TARGET_ZONE",
            )

    def test_reject_unsupported_region_type(self):
        with pytest.raises(ValueError, match="Unsupported region type"):
            TaskRegion(
                role="target_region",
                description="A target zone",
                region_type="WORK_AREA",
            )

    def test_reject_non_boolean_required(self):
        with pytest.raises(ValueError, match="required"):
            TaskRegion(
                role="target_region",
                description="A target zone",
                region_type="TARGET_ZONE",
                required=1,
            )

    def test_region_is_frozen(self):
        region = make_target_region()

        with pytest.raises(FrozenInstanceError):
            region.description = "Another region"


class TestTaskCriterion:
    def test_create_valid_stable_placement_criterion(self):
        criterion = make_stable_placement_criterion()

        assert criterion.criterion_id == "stable_placement"
        assert criterion.event_type == "STABLE_PLACEMENT"
        assert criterion.entity_roles == ("object",)
        assert criterion.region_role == "target_region"
        assert criterion.is_mandatory is True

    def test_normalize_entity_roles(self):
        criterion = TaskCriterion(
            criterion_id="object_movement",
            description="The object moves.",
            event_type="OBJECT_MOVING",
            entity_roles=["object"],
        )

        assert criterion.entity_roles == ("object",)

    def test_reject_unsupported_event_type(self):
        with pytest.raises(ValueError, match="Unsupported event type"):
            TaskCriterion(
                criterion_id="unknown_event",
                description="An unsupported event.",
                event_type="OBJECT_GRASPED",
                entity_roles=("object",),
            )

    def test_reject_unsupported_entity_role(self):
        with pytest.raises(ValueError, match="Unsupported entity roles"):
            TaskCriterion(
                criterion_id="object_movement",
                description="The object moves.",
                event_type="OBJECT_MOVING",
                entity_roles=("object", "robot"),
            )

    def test_reject_missing_required_entity_roles(self):
        with pytest.raises(ValueError, match="requires entity roles"):
            TaskCriterion(
                criterion_id="coordinated_movement",
                description="The object moves with the gripper.",
                event_type="OBJECT_MOVING_WITH_EFFECTOR",
                entity_roles=("object",),
            )

    def test_reject_duplicate_entity_roles(self):
        with pytest.raises(ValueError, match="cannot contain duplicates"):
            TaskCriterion(
                criterion_id="object_movement",
                description="The object moves.",
                event_type="OBJECT_MOVING",
                entity_roles=("object", "object"),
            )

    def test_region_entry_requires_region_role(self):
        with pytest.raises(ValueError, match="requires a region_role"):
            TaskCriterion(
                criterion_id="target_entry",
                description="The object enters the target zone.",
                event_type="REGION_ENTRY",
                entity_roles=("object",),
            )

    def test_stable_placement_requires_region_role(self):
        with pytest.raises(ValueError, match="requires a region_role"):
            TaskCriterion(
                criterion_id="stable_placement",
                description="The object remains stable.",
                event_type="STABLE_PLACEMENT",
                entity_roles=("object",),
            )

    def test_object_movement_does_not_accept_region_role(self):
        with pytest.raises(ValueError, match="does not support a region_role"):
            TaskCriterion(
                criterion_id="object_movement",
                description="The object moves.",
                event_type="OBJECT_MOVING",
                entity_roles=("object",),
                region_role="target_region",
            )

    def test_reject_unsupported_region_role(self):
        with pytest.raises(ValueError, match="Unsupported region role"):
            TaskCriterion(
                criterion_id="target_entry",
                description="The object enters a region.",
                event_type="REGION_ENTRY",
                entity_roles=("object",),
                region_role="forbidden_region",
            )

    def test_reject_non_boolean_mandatory(self):
        with pytest.raises(ValueError, match="is_mandatory"):
            TaskCriterion(
                criterion_id="object_movement",
                description="The object moves.",
                event_type="OBJECT_MOVING",
                entity_roles=("object",),
                is_mandatory=1,
            )

    def test_criterion_is_frozen(self):
        criterion = make_stable_placement_criterion()

        with pytest.raises(FrozenInstanceError):
            criterion.description = "Another description"


class TestTaskDefinition:
    def test_create_valid_task_definition(self):
        task = make_task_definition()

        assert task.task_summary.startswith("Move the red cube")
        assert len(task.entities) == 2
        assert len(task.regions) == 1
        assert len(task.criteria) == 3
        assert task.unresolved_items == ()

    def test_normalize_collections(self):
        task = TaskDefinition(
            task_summary="Move the object.",
            entities=[make_object()],
            regions=[make_target_region()],
            criteria=[
                TaskCriterion(
                    criterion_id="object_movement",
                    description="The object moves.",
                    event_type="OBJECT_MOVING",
                    entity_roles=["object"],
                )
            ],
            unresolved_items=["Confirm the target region."],
        )

        assert isinstance(task.entities, tuple)
        assert isinstance(task.regions, tuple)
        assert isinstance(task.criteria, tuple)
        assert task.unresolved_items == (
            "Confirm the target region.",
        )

    def test_reject_empty_task_summary(self):
        with pytest.raises(ValueError, match="task_summary"):
            TaskDefinition(
                task_summary="",
                entities=(make_object(),),
                regions=(),
                criteria=(
                    TaskCriterion(
                        criterion_id="object_movement",
                        description="The object moves.",
                        event_type="OBJECT_MOVING",
                        entity_roles=("object",),
                    ),
                ),
            )

    def test_reject_empty_entities(self):
        with pytest.raises(ValueError, match="entities must not be empty"):
            TaskDefinition(
                task_summary="Move the object.",
                entities=(),
                regions=(),
                criteria=(
                    TaskCriterion(
                        criterion_id="object_movement",
                        description="The object moves.",
                        event_type="OBJECT_MOVING",
                        entity_roles=("object",),
                    ),
                ),
            )

    def test_reject_empty_criteria(self):
        with pytest.raises(ValueError, match="criteria must not be empty"):
            TaskDefinition(
                task_summary="Move the object.",
                entities=(make_object(),),
                regions=(),
                criteria=(),
            )

    def test_reject_missing_object_entity(self):
        with pytest.raises(ValueError, match="object entity is required"):
            TaskDefinition(
                task_summary="Move the object.",
                entities=(make_effector(),),
                regions=(),
                criteria=(
                    TaskCriterion(
                        criterion_id="object_movement",
                        description="The object moves.",
                        event_type="OBJECT_MOVING_WITH_EFFECTOR",
                        entity_roles=("object", "effector"),
                    ),
                ),
            )

    def test_reject_duplicate_entity_roles(self):
        with pytest.raises(ValueError, match="Entity roles must be unique"):
            TaskDefinition(
                task_summary="Move the object.",
                entities=(make_object(), make_object()),
                regions=(),
                criteria=(
                    TaskCriterion(
                        criterion_id="object_movement",
                        description="The object moves.",
                        event_type="OBJECT_MOVING",
                        entity_roles=("object",),
                    ),
                ),
            )

    def test_reject_duplicate_region_roles(self):
        with pytest.raises(ValueError, match="Region roles must be unique"):
            TaskDefinition(
                task_summary="Move the object.",
                entities=(make_object(),),
                regions=(make_target_region(), make_target_region()),
                criteria=(
                    make_stable_placement_criterion(),
                ),
            )

    def test_reject_duplicate_criterion_ids(self):
        criterion = TaskCriterion(
            criterion_id="object_movement",
            description="The object moves.",
            event_type="OBJECT_MOVING",
            entity_roles=("object",),
        )

        with pytest.raises(ValueError, match="Criterion IDs must be unique"):
            TaskDefinition(
                task_summary="Move the object.",
                entities=(make_object(),),
                regions=(),
                criteria=(criterion, criterion),
            )

    def test_reject_criterion_referencing_missing_entity_role(self):
        criterion = TaskCriterion(
            criterion_id="coordinated_movement",
            description="The object moves with the gripper.",
            event_type="OBJECT_MOVING_WITH_EFFECTOR",
            entity_roles=("object", "effector"),
        )

        with pytest.raises(
            ValueError,
            match="references undefined entity roles",
        ):
            TaskDefinition(
                task_summary="Move the object.",
                entities=(make_object(),),
                regions=(),
                criteria=(criterion,),
            )

    def test_reject_criterion_referencing_missing_region(self):
        with pytest.raises(
            ValueError,
            match="references undefined region role",
        ):
            TaskDefinition(
                task_summary="Place the object.",
                entities=(make_object(),),
                regions=(),
                criteria=(make_stable_placement_criterion(),),
            )

    def test_allow_unresolved_items(self):
        task = TaskDefinition(
            task_summary="Move the red cube.",
            entities=(make_object(),),
            regions=(),
            criteria=(
                TaskCriterion(
                    criterion_id="object_movement",
                    description="The object moves.",
                    event_type="OBJECT_MOVING",
                    entity_roles=("object",),
                ),
            ),
            unresolved_items=(
                "The target region has not been identified.",
            ),
        )

        assert len(task.unresolved_items) == 1

    def test_reject_duplicate_unresolved_items(self):
        with pytest.raises(
            ValueError,
            match="cannot contain duplicates",
        ):
            TaskDefinition(
                task_summary="Move the object.",
                entities=(make_object(),),
                regions=(),
                criteria=(
                    TaskCriterion(
                        criterion_id="object_movement",
                        description="The object moves.",
                        event_type="OBJECT_MOVING",
                        entity_roles=("object",),
                    ),
                ),
                unresolved_items=("Identify the object.", "Identify the object."),
            )

    def test_task_definition_is_frozen(self):
        task = make_task_definition()

        with pytest.raises(FrozenInstanceError):
            task.task_summary = "Another task."
