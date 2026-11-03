
"""Tests for declarative verification criteria."""

import pytest

from src.models.verification_criteria import (
    CriterionRequirement,
    VerificationCriteria,
)


def create_criteria(**overrides) -> VerificationCriteria:
    values = {
        "criterion_id": "object_reaches_target",
        "description": "The object must reach the target region.",
        "event_type": "STABLE_PLACEMENT",
        "requirement": CriterionRequirement.REQUIRED,
        "entity_roles": ("object",),
        "region_role": "target_region",
        "required_evidence_types": ("STABLE_IN_REGION",),
        "is_mandatory": True,
    }
    values.update(overrides)
    return VerificationCriteria(**values)


def test_creates_valid_required_criterion():
    criterion = create_criteria()

    assert criterion.criterion_id == "object_reaches_target"
    assert criterion.requirement == CriterionRequirement.REQUIRED
    assert criterion.is_mandatory is True
    assert criterion.entity_roles == ("object",)


def test_creates_valid_forbidden_criterion():
    criterion = create_criteria(
        criterion_id="avoid_restricted_area",
        event_type="REGION_ENTRY",
        requirement=CriterionRequirement.FORBIDDEN,
        region_role="forbidden_region",
    )

    assert criterion.requirement == CriterionRequirement.FORBIDDEN


def test_creates_optional_criterion():
    criterion = create_criteria(is_mandatory=False)

    assert criterion.is_mandatory is False


def test_creates_criterion_without_entity_roles():
    criterion = create_criteria(entity_roles=())

    assert criterion.entity_roles == ()


def test_creates_criterion_without_region_role():
    criterion = create_criteria(region_role=None)

    assert criterion.region_role is None


def test_creates_criterion_without_required_evidence_types():
    criterion = create_criteria(required_evidence_types=())

    assert criterion.required_evidence_types == ()


@pytest.mark.parametrize(
    "field,value",
    [
        ("criterion_id", ""),
        ("criterion_id", "   "),
        ("description", ""),
        ("description", "   "),
        ("event_type", ""),
        ("event_type", "   "),
    ],
)
def test_rejects_empty_required_strings(field, value):
    with pytest.raises(ValueError):
        create_criteria(**{field: value})


def test_rejects_invalid_requirement():
    with pytest.raises(ValueError):
        create_criteria(requirement="REQUIRED")


@pytest.mark.parametrize("value", [1, 0, None, "true"])
def test_rejects_non_boolean_mandatory_flag(value):
    with pytest.raises(ValueError):
        create_criteria(is_mandatory=value)


@pytest.mark.parametrize(
    "value",
    [
        ["object"],
        "object",
        None,
    ],
)
def test_rejects_non_tuple_entity_roles(value):
    with pytest.raises(ValueError):
        create_criteria(entity_roles=value)


@pytest.mark.parametrize(
    "value",
    [
        ("",),
        ("   ",),
        (1,),
        (None,),
    ],
)
def test_rejects_invalid_entity_roles(value):
    with pytest.raises(ValueError):
        create_criteria(entity_roles=value)


def test_rejects_duplicate_entity_roles():
    with pytest.raises(ValueError):
        create_criteria(entity_roles=("object", "object"))


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        123,
        (),
    ],
)
def test_rejects_invalid_region_role(value):
    with pytest.raises(ValueError):
        create_criteria(region_role=value)


@pytest.mark.parametrize(
    "value",
    [
        ["STABLE_IN_REGION"],
        "STABLE_IN_REGION",
        None,
    ],
)
def test_rejects_non_tuple_required_evidence_types(value):
    with pytest.raises(ValueError):
        create_criteria(required_evidence_types=value)


@pytest.mark.parametrize(
    "value",
    [
        ("",),
        ("   ",),
        (123,),
        (None,),
    ],
)
def test_rejects_invalid_required_evidence_types(value):
    with pytest.raises(ValueError):
        create_criteria(required_evidence_types=value)


def test_rejects_duplicate_required_evidence_types():
    with pytest.raises(ValueError):
        create_criteria(
            required_evidence_types=(
                "STABLE_IN_REGION",
                "STABLE_IN_REGION",
            )
        )


def test_criterion_is_immutable():
    criterion = create_criteria()

    with pytest.raises(AttributeError):
        criterion.criterion_id = "modified"

def test_creates_criterion_with_one_of_evidence_types():
    criterion = create_criteria(
        required_evidence_types=(),
        one_of_evidence_types=(
            "MASK_DISTANCE_SEPARATION",
            "RELATIVE_CENTROID_SEPARATION",
        ),
    )

    assert criterion.one_of_evidence_types == (
        "MASK_DISTANCE_SEPARATION",
        "RELATIVE_CENTROID_SEPARATION",
    )


@pytest.mark.parametrize(
    "value",
    [
        ["MASK_DISTANCE_SEPARATION"],
        "MASK_DISTANCE_SEPARATION",
        None,
    ],
)
def test_rejects_non_tuple_one_of_evidence_types(value):
    with pytest.raises(ValueError):
        create_criteria(one_of_evidence_types=value)


@pytest.mark.parametrize(
    "value",
    [
        ("",),
        ("   ",),
        (123,),
        (None,),
    ],
)
def test_rejects_invalid_one_of_evidence_types(value):
    with pytest.raises(ValueError):
        create_criteria(one_of_evidence_types=value)


def test_rejects_duplicate_one_of_evidence_types():
    with pytest.raises(ValueError):
        create_criteria(
            one_of_evidence_types=(
                "MASK_DISTANCE_SEPARATION",
                "MASK_DISTANCE_SEPARATION",
            )
        )