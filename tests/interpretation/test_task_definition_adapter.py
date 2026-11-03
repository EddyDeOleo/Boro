
"""Tests for adapting task definitions into verification domain models."""

import pytest

from src.interpretation.task_definition_adapter import (
    EVENT_REQUIRED_EVIDENCE_TYPES,
    AdaptedTaskDefinition,
    TaskDefinitionAdapter,
    TaskDefinitionAdapterError,
)
from src.models.task_definition import (
    TaskCriterion,
    TaskDefinition,
    TaskEntity,
    TaskRegion,
)
from src.models.verification_criteria import CriterionRequirement


def make_object():
    return TaskEntity(
        role="object",
        description="The red cube.",
        visual_attributes={
            "color": "red",
            "shape": "cube",
        },
    )


def make_effector():
    return TaskEntity(
        role="effector",
        description="The robot gripper.",
        visual_attributes={},
    )


def make_target_region():
    return TaskRegion(
        role="target_region",
        description="The intended destination.",
        region_type="TARGET_ZONE",
    )


def make_task_definition(
    unresolved_items=(),
    include_effector=True,
    include_region=True,
):
    entities = [make_object()]

    if include_effector:
        entities.append(make_effector())

    regions = [make_target_region()] if include_region else []

    criteria = [
        TaskCriterion(
            criterion_id="object_moves",
            description="The object moves.",
            event_type="OBJECT_MOVING",
            entity_roles=("object",),
        )
    ]

    if include_effector:
        criteria.append(
            TaskCriterion(
                criterion_id="object_moves_with_effector",
                description="The object moves with the effector.",
                event_type="OBJECT_MOVING_WITH_EFFECTOR",
                entity_roles=("object", "effector"),
            )
        )

    if include_region:
        criteria.append(
            TaskCriterion(
                criterion_id="object_stabilizes",
                description="The object remains stable in the target region.",
                event_type="STABLE_PLACEMENT",
                entity_roles=("object",),
                region_role="target_region",
            )
        )

    return TaskDefinition(
        task_summary="Move the red cube into the target area.",
        entities=tuple(entities),
        regions=tuple(regions),
        criteria=tuple(criteria),
        unresolved_items=tuple(unresolved_items),
    )


def make_adapter():
    return TaskDefinitionAdapter()


def test_adapt_returns_domain_configuration():
    task_definition = make_task_definition()

    result = make_adapter().adapt(
        task_definition=task_definition,
        entity_id_map={
            "object": "entity-1",
            "effector": "entity-2",
        },
        region_id_map={
            "target_region": "region-1",
        },
    )

    assert isinstance(result, AdaptedTaskDefinition)
    assert result.task_roles.get_entity_id("object") == "entity-1"
    assert result.task_roles.get_entity_id("effector") == "entity-2"
    assert result.task_roles.get_region_id("target_region") == "region-1"
    assert len(result.criteria) == 3


def test_adapt_maps_event_evidence_types():
    task_definition = make_task_definition()

    result = make_adapter().adapt(
        task_definition,
        entity_id_map={
            "object": "entity-1",
            "effector": "entity-2",
        },
        region_id_map={
            "target_region": "region-1",
        },
    )

    for adapted_criterion in result.criteria:
        assert adapted_criterion.required_evidence_types == (
            EVENT_REQUIRED_EVIDENCE_TYPES[
                adapted_criterion.event_type
            ]
        )


@pytest.mark.parametrize(
    ("event_type", "expected_evidence"),
    [
        ("OBJECT_MOVING", ("CENTROID_DISPLACEMENT",)),
        (
            "OBJECT_MOVING_WITH_EFFECTOR",
            ("COORDINATED_ENTITY_MOTION",),
        ),
        (
            "RELATIVE_SEPARATION",
            ("RELATIVE_CENTROID_SEPARATION",),
        ),
        ("REGION_ENTRY", ("ENTITY_ENTERED_REGION",)),
        ("STABLE_PLACEMENT", ("STABLE_IN_REGION",)),
    ],
)
def test_evidence_mapping_matches_event_detector(
    event_type,
    expected_evidence,
):
    assert EVENT_REQUIRED_EVIDENCE_TYPES[event_type] == expected_evidence


def test_adapt_uses_required_criterion_requirement():
    task_definition = make_task_definition()

    result = make_adapter().adapt(
        task_definition,
        entity_id_map={
            "object": "entity-1",
            "effector": "entity-2",
        },
        region_id_map={
            "target_region": "region-1",
        },
    )

    assert all(
        criterion.requirement == CriterionRequirement.REQUIRED
        for criterion in result.criteria
    )


def test_adapt_preserves_criterion_mandatory_flag():
    task_definition = TaskDefinition(
        task_summary="Move the object.",
        entities=(make_object(),),
        regions=(),
        criteria=(
            TaskCriterion(
                criterion_id="object_moves",
                description="The object moves.",
                event_type="OBJECT_MOVING",
                entity_roles=("object",),
                is_mandatory=False,
            ),
        ),
    )

    result = make_adapter().adapt(
        task_definition,
        entity_id_map={"object": "entity-1"},
        region_id_map={},
    )

    assert result.criteria[0].is_mandatory is False


def test_adapt_preserves_criterion_roles():
    task_definition = make_task_definition()

    result = make_adapter().adapt(
        task_definition,
        entity_id_map={
            "object": "entity-1",
            "effector": "entity-2",
        },
        region_id_map={
            "target_region": "region-1",
        },
    )

    moving_criterion = next(
        criterion
        for criterion in result.criteria
        if criterion.criterion_id == "object_moves_with_effector"
    )

    assert moving_criterion.entity_roles == ("object", "effector")


def test_adapt_rejects_wrong_task_definition_type():
    with pytest.raises(TypeError, match="TaskDefinition"):
        make_adapter().adapt(
            task_definition={},
            entity_id_map={},
            region_id_map={},
        )


def test_adapt_rejects_unresolved_task_definition():
    task_definition = make_task_definition(
        unresolved_items=("Target boundaries are unknown.",)
    )

    with pytest.raises(
        TaskDefinitionAdapterError,
        match="unresolved items",
    ):
        make_adapter().adapt(
            task_definition,
            entity_id_map={
                "object": "entity-1",
                "effector": "entity-2",
            },
            region_id_map={
                "target_region": "region-1",
            },
        )


@pytest.mark.parametrize(
    "entity_id_map",
    [None, [], "invalid"],
)
def test_adapt_rejects_invalid_entity_mapping(entity_id_map):
    task_definition = make_task_definition()

    with pytest.raises(TaskDefinitionAdapterError, match="entity_id_map"):
        make_adapter().adapt(
            task_definition,
            entity_id_map=entity_id_map,
            region_id_map={"target_region": "region-1"},
        )


@pytest.mark.parametrize(
    "region_id_map",
    [None, [], "invalid"],
)
def test_adapt_rejects_invalid_region_mapping(region_id_map):
    task_definition = make_task_definition()

    with pytest.raises(TaskDefinitionAdapterError, match="region_id_map"):
        make_adapter().adapt(
            task_definition,
            entity_id_map={
                "object": "entity-1",
                "effector": "entity-2",
            },
            region_id_map=region_id_map,
        )


def test_adapt_rejects_missing_entity_id():
    task_definition = make_task_definition()

    with pytest.raises(
        TaskDefinitionAdapterError,
        match="Missing entity IDs",
    ):
        make_adapter().adapt(
            task_definition,
            entity_id_map={"object": "entity-1"},
            region_id_map={"target_region": "region-1"},
        )


def test_adapt_rejects_missing_region_id():
    task_definition = make_task_definition()

    with pytest.raises(
        TaskDefinitionAdapterError,
        match="Missing region IDs",
    ):
        make_adapter().adapt(
            task_definition,
            entity_id_map={
                "object": "entity-1",
                "effector": "entity-2",
            },
            region_id_map={},
        )


def test_adapt_rejects_undeclared_entity_role():
    task_definition = make_task_definition()

    with pytest.raises(
        TaskDefinitionAdapterError,
        match="undeclared roles",
    ):
        make_adapter().adapt(
            task_definition,
            entity_id_map={
                "object": "entity-1",
                "effector": "entity-2",
                "camera": "entity-3",
            },
            region_id_map={"target_region": "region-1"},
        )


def test_adapt_rejects_undeclared_region_role():
    task_definition = make_task_definition()

    with pytest.raises(
        TaskDefinitionAdapterError,
        match="undeclared roles",
    ):
        make_adapter().adapt(
            task_definition,
            entity_id_map={
                "object": "entity-1",
                "effector": "entity-2",
            },
            region_id_map={
                "target_region": "region-1",
                "forbidden_region": "region-2",
            },
        )


def test_adapt_rejects_duplicate_entity_ids():
    task_definition = make_task_definition()

    with pytest.raises(
        TaskDefinitionAdapterError,
        match="same ID",
    ):
        make_adapter().adapt(
            task_definition,
            entity_id_map={
                "object": "same-id",
                "effector": "same-id",
            },
            region_id_map={"target_region": "region-1"},
        )


def test_adapt_accepts_single_region_id():
    """A single declared target region receives its mapped domain ID."""
    task_definition = TaskDefinition(
        task_summary="Move the object.",
        entities=(make_object(),),
        regions=(make_target_region(),),
        criteria=(
            TaskCriterion(
                criterion_id="object_enters_target",
                description="The object enters the target.",
                event_type="REGION_ENTRY",
                entity_roles=("object",),
                region_role="target_region",
            ),
        ),
    )

    result = make_adapter().adapt(
        task_definition,
        entity_id_map={"object": "entity-1"},
        region_id_map={"target_region": "region-1"},
    )

    assert result.task_roles.get_region_id("target_region") == "region-1"
    assert result.criteria[0].region_role == "target_region"


def test_adapt_rejects_criterion_using_optional_entity():
    task_definition = TaskDefinition(
        task_summary="Move the object.",
        entities=(
            make_object(),
            TaskEntity(
                role="effector",
                description="Optional gripper.",
                visual_attributes={},
                required=False,
            ),
        ),
        regions=(),
        criteria=(
            TaskCriterion(
                criterion_id="object_moves_with_effector",
                description="The object moves with the effector.",
                event_type="OBJECT_MOVING_WITH_EFFECTOR",
                entity_roles=("object", "effector"),
            ),
        ),
    )

    with pytest.raises(
        TaskDefinitionAdapterError,
        match="optional entity role",
    ):
        make_adapter().adapt(
            task_definition,
            entity_id_map={
                "object": "entity-1",
                "effector": "entity-2",
            },
            region_id_map={},
        )


def test_adapt_rejects_criterion_using_optional_region():
    task_definition = TaskDefinition(
        task_summary="Move the object.",
        entities=(make_object(),),
        regions=(
            TaskRegion(
                role="target_region",
                description="Optional target area.",
                region_type="TARGET_ZONE",
                required=False,
            ),
        ),
        criteria=(
            TaskCriterion(
                criterion_id="object_enters_target",
                description="The object enters the target.",
                event_type="REGION_ENTRY",
                entity_roles=("object",),
                region_role="target_region",
            ),
        ),
    )

    with pytest.raises(
        TaskDefinitionAdapterError,
        match="optional region role",
    ):
        make_adapter().adapt(
            task_definition,
            entity_id_map={"object": "entity-1"},
            region_id_map={"target_region": "region-1"},
        )


def test_adapt_supports_task_without_region():
    task_definition = make_task_definition(
        include_effector=False,
        include_region=False,
    )

    result = make_adapter().adapt(
        task_definition,
        entity_id_map={"object": "entity-1"},
        region_id_map={},
    )

    assert result.task_roles.get_entity_id("object") == "entity-1"
    assert not result.criteria[0].region_role


def test_adapt_does_not_create_region_geometry():
    task_definition = make_task_definition()

    result = make_adapter().adapt(
        task_definition,
        entity_id_map={
            "object": "entity-1",
            "effector": "entity-2",
        },
        region_id_map={"target_region": "region-1"},
    )

    assert isinstance(result, AdaptedTaskDefinition)
    assert not hasattr(result, "regions")
