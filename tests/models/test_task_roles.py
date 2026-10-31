
"""Tests for generic robotic task role configuration."""

import pytest

from src.models.task_roles import TaskRoles


class TestTaskRoles:
    """Tests for TaskRoles."""

    def test_create_task_roles(self):
        roles = TaskRoles(
            entity_roles={
                "manipulated_object": "red_cube",
                "manipulator": "gripper",
            },
            region_roles={
                "destination": "target_zone_01",
            },
        )

        assert roles.get_entity_id("manipulated_object") == "red_cube"
        assert roles.get_entity_id("manipulator") == "gripper"
        assert roles.get_region_id("destination") == "target_zone_01"

    def test_create_empty_task_roles(self):
        roles = TaskRoles()

        assert roles.entity_roles == {}
        assert roles.region_roles == {}

    def test_create_entity_roles_without_regions(self):
        roles = TaskRoles(
            entity_roles={
                "object": "component_01",
            }
        )

        assert roles.get_entity_id("object") == "component_01"
        assert roles.region_roles == {}

    def test_create_region_roles_without_entities(self):
        roles = TaskRoles(
            region_roles={
                "restricted_area": "zone_01",
            }
        )

        assert roles.get_region_id("restricted_area") == "zone_01"
        assert roles.entity_roles == {}

    def test_has_entity_role_returns_true(self):
        roles = TaskRoles(
            entity_roles={
                "manipulator": "gripper",
            }
        )

        assert roles.has_entity_role("manipulator") is True

    def test_has_entity_role_returns_false(self):
        roles = TaskRoles()

        assert roles.has_entity_role("manipulator") is False

    def test_has_region_role_returns_true(self):
        roles = TaskRoles(
            region_roles={
                "destination": "target_zone",
            }
        )

        assert roles.has_region_role("destination") is True

    def test_has_region_role_returns_false(self):
        roles = TaskRoles()

        assert roles.has_region_role("destination") is False

    def test_unknown_entity_role_raises_key_error(self):
        roles = TaskRoles()

        with pytest.raises(KeyError, match="No entity is configured"):
            roles.get_entity_id("unknown_object")

    def test_unknown_region_role_raises_key_error(self):
        roles = TaskRoles()

        with pytest.raises(KeyError, match="No region is configured"):
            roles.get_region_id("unknown_region")

    @pytest.mark.parametrize(
        "mapping",
        [
            {"": "entity_01"},
            {"object": ""},
            {"   ": "entity_01"},
            {"object": "   "},
        ],
    )
    def test_invalid_entity_role_mapping_raises_value_error(self, mapping):
        with pytest.raises(ValueError):
            TaskRoles(entity_roles=mapping)

    @pytest.mark.parametrize(
        "mapping",
        [
            {"": "region_01"},
            {"destination": ""},
            {"   ": "region_01"},
            {"destination": "   "},
        ],
    )
    def test_invalid_region_role_mapping_raises_value_error(self, mapping):
        with pytest.raises(ValueError):
            TaskRoles(region_roles=mapping)

    @pytest.mark.parametrize(
        "invalid_mapping",
        [
            None,
            [],
            "invalid",
            123,
        ],
    )
    def test_invalid_entity_roles_type_raises_value_error(
        self,
        invalid_mapping,
    ):
        with pytest.raises(ValueError, match="must be a mapping"):
            TaskRoles(entity_roles=invalid_mapping)

    @pytest.mark.parametrize(
        "invalid_mapping",
        [
            None,
            [],
            "invalid",
            123,
        ],
    )
    def test_invalid_region_roles_type_raises_value_error(
        self,
        invalid_mapping,
    ):
        with pytest.raises(ValueError, match="must be a mapping"):
            TaskRoles(region_roles=invalid_mapping)

    def test_entity_roles_are_immutable(self):
        roles = TaskRoles(
            entity_roles={
                "object": "red_cube",
            }
        )

        with pytest.raises(TypeError):
            roles.entity_roles["object"] = "blue_cube"

    def test_region_roles_are_immutable(self):
        roles = TaskRoles(
            region_roles={
                "destination": "target_zone",
            }
        )

        with pytest.raises(TypeError):
            roles.region_roles["destination"] = "another_zone"

    def test_external_mapping_changes_do_not_affect_roles(self):
        entity_mapping = {
            "object": "red_cube",
        }

        region_mapping = {
            "destination": "target_zone",
        }

        roles = TaskRoles(
            entity_roles=entity_mapping,
            region_roles=region_mapping,
        )

        entity_mapping["object"] = "blue_cube"
        region_mapping["destination"] = "another_zone"

        assert roles.get_entity_id("object") == "red_cube"
        assert roles.get_region_id("destination") == "target_zone"

    def test_same_entity_can_have_multiple_semantic_roles(self):
        roles = TaskRoles(
            entity_roles={
                "manipulator": "robot_arm",
                "effector": "robot_arm",
            }
        )

        assert roles.get_entity_id("manipulator") == "robot_arm"
        assert roles.get_entity_id("effector") == "robot_arm"

    def test_roles_support_different_robotic_tasks(self):
        roles = TaskRoles(
            entity_roles={
                "welding_tool": "tool_01",
                "welded_component": "component_01",
                "inspection_camera": "camera_01",
            },
            region_roles={
                "restricted_area": "zone_01",
                "inspection_area": "zone_02",
            },
        )

        assert roles.get_entity_id("welding_tool") == "tool_01"
        assert roles.get_entity_id("welded_component") == "component_01"
        assert roles.get_entity_id("inspection_camera") == "camera_01"
        assert roles.get_region_id("restricted_area") == "zone_01"
        assert roles.get_region_id("inspection_area") == "zone_02"
