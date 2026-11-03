
import pytest

from src.models.perception_config import PerceptionConfig


def test_creates_valid_configuration():
    config = PerceptionConfig(
        provider="meta",
        model="sam-3.1",
        entity_id_map={
            "1": "red_cube",
            "2": "gripper",
            "3": "target_box",
            "4": "obstacle",
        },
    )

    assert config.provider == "meta"
    assert config.model == "sam-3.1"
    assert config.entity_ids == (
        "red_cube",
        "gripper",
        "target_box",
        "obstacle",
    )
    assert config.entity_id_map["1"] == "red_cube"
    assert config.entity_id_map["4"] == "obstacle"
    assert config.include_spatial_relationships is True
    assert config.include_region_observations is True


def test_normalizes_entity_mapping():
    config = PerceptionConfig(
        provider="meta",
        model="sam-3.1",
        entity_id_map={
            "1": "red_cube",
            "2": "gripper",
        },
    )

    assert config.entity_ids == ("red_cube", "gripper")


@pytest.mark.parametrize(
    "provider",
    ["", "   ", None],
)
def test_rejects_invalid_provider(provider):
    with pytest.raises(ValueError):
        PerceptionConfig(
            provider=provider,
            model="sam-3.1",
            entity_id_map={"1": "red_cube"},
        )


@pytest.mark.parametrize(
    "model",
    ["", "   ", None],
)
def test_rejects_invalid_model(model):
    with pytest.raises(ValueError):
        PerceptionConfig(
            provider="meta",
            model=model,
            entity_id_map={"1": "red_cube"},
        )


@pytest.mark.parametrize(
    "entity_id_map",
    [
        {},
        None,
        [],
        {"1": ""},
        {"1": "   "},
        {"": "red_cube"},
        {"   ": "red_cube"},
        {"1": "red_cube", "2": "red_cube"},
    ],
)
def test_rejects_invalid_entity_mapping(entity_id_map):
    with pytest.raises(ValueError):
        PerceptionConfig(
            provider="meta",
            model="sam-3.1",
            entity_id_map=entity_id_map,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "include_spatial_relationships",
        "include_region_observations",
    ],
)
def test_rejects_non_boolean_options(field_name):
    with pytest.raises(ValueError):
        PerceptionConfig(
            provider="meta",
            model="sam-3.1",
            entity_id_map={"1": "red_cube"},
            **{field_name: 1},
        )


def test_configuration_is_immutable():
    config = PerceptionConfig(
        provider="meta",
        model="sam-3.1",
        entity_id_map={"1": "red_cube"},
    )

    with pytest.raises(AttributeError):
        config.provider = "other"


def test_entity_mapping_is_immutable():
    config = PerceptionConfig(
        provider="meta",
        model="sam-3.1",
        entity_id_map={"1": "red_cube"},
    )

    with pytest.raises(TypeError):
        config.entity_id_map["2"] = "gripper"


def test_normalizes_mapping_without_mutating_original():
    original_mapping = {
        "1": "red_cube",
        "2": "gripper",
    }

    config = PerceptionConfig(
        provider="meta",
        model="sam-3.1",
        entity_id_map=original_mapping,
    )

    original_mapping["3"] = "target_box"

    assert config.entity_ids == ("red_cube", "gripper")
    assert "3" not in config.entity_id_map
