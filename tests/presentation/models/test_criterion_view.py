
import pytest

from src.models.verification_criteria import CriterionRequirement
from src.models.verification_result import CriterionResultStatus
from src.presentation.models.criterion_view import CriterionView


@pytest.fixture
def valid_criterion():
    return CriterionView(
        criterion_id="stable_target_placement",
        requirement=CriterionRequirement.REQUIRED.value,
        status=CriterionResultStatus.INSUFFICIENT_EVIDENCE.value,
        is_mandatory=True,
        explanation="Stable placement could not be confirmed.",
        event_ids=("event-001",),
        evidence_ids=("evidence-001",),
    )


def test_creates_valid_criterion_view(valid_criterion):
    assert valid_criterion.criterion_id == "stable_target_placement"
    assert valid_criterion.requirement == CriterionRequirement.REQUIRED.value
    assert (
        valid_criterion.status
        == CriterionResultStatus.INSUFFICIENT_EVIDENCE.value
    )
    assert valid_criterion.is_mandatory is True
    assert valid_criterion.explanation == (
        "Stable placement could not be confirmed."
    )
    assert valid_criterion.event_ids == ("event-001",)
    assert valid_criterion.evidence_ids == ("evidence-001",)


@pytest.mark.parametrize(
    "field,value",
    [
        ("criterion_id", ""),
        ("requirement", ""),
        ("status", ""),
        ("explanation", ""),
    ],
)
def test_rejects_empty_required_strings(valid_criterion, field, value):
    values = {
        "criterion_id": valid_criterion.criterion_id,
        "requirement": valid_criterion.requirement,
        "status": valid_criterion.status,
        "is_mandatory": valid_criterion.is_mandatory,
        "explanation": valid_criterion.explanation,
        "event_ids": valid_criterion.event_ids,
        "evidence_ids": valid_criterion.evidence_ids,
    }
    values[field] = value

    with pytest.raises(ValueError):
        CriterionView(**values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("requirement", "OPTIONAL"),
        ("status", "UNKNOWN"),
    ],
)
def test_rejects_invalid_enum_values(valid_criterion, field, value):
    values = {
        "criterion_id": valid_criterion.criterion_id,
        "requirement": valid_criterion.requirement,
        "status": valid_criterion.status,
        "is_mandatory": valid_criterion.is_mandatory,
        "explanation": valid_criterion.explanation,
        "event_ids": valid_criterion.event_ids,
        "evidence_ids": valid_criterion.evidence_ids,
    }
    values[field] = value

    with pytest.raises(ValueError):
        CriterionView(**values)


def test_rejects_non_boolean_mandatory_flag(valid_criterion):
    values = {
        "criterion_id": valid_criterion.criterion_id,
        "requirement": valid_criterion.requirement,
        "status": valid_criterion.status,
        "is_mandatory": 1,
        "explanation": valid_criterion.explanation,
        "event_ids": valid_criterion.event_ids,
        "evidence_ids": valid_criterion.evidence_ids,
    }

    with pytest.raises((ValueError, TypeError)):
        CriterionView(**values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_ids", ("event-001", "event-001")),
        ("evidence_ids", ("evidence-001", "evidence-001")),
        ("event_ids", ("",)),
        ("evidence_ids", ("",)),
    ],
)
def test_rejects_invalid_reference_tuples(valid_criterion, field, value):
    values = {
        "criterion_id": valid_criterion.criterion_id,
        "requirement": valid_criterion.requirement,
        "status": valid_criterion.status,
        "is_mandatory": valid_criterion.is_mandatory,
        "explanation": valid_criterion.explanation,
        "event_ids": valid_criterion.event_ids,
        "evidence_ids": valid_criterion.evidence_ids,
    }
    values[field] = value

    with pytest.raises(ValueError):
        CriterionView(**values)


def test_rejects_non_tuple_reference_collections(valid_criterion):
    values = {
        "criterion_id": valid_criterion.criterion_id,
        "requirement": valid_criterion.requirement,
        "status": valid_criterion.status,
        "is_mandatory": valid_criterion.is_mandatory,
        "explanation": valid_criterion.explanation,
        "event_ids": ["event-001"],
        "evidence_ids": ("evidence-001",),
    }

    with pytest.raises((ValueError, TypeError)):
        CriterionView(**values)


def test_criterion_view_is_immutable(valid_criterion):
    with pytest.raises((AttributeError, TypeError)):
        valid_criterion.status = CriterionResultStatus.SATISFIED.value
