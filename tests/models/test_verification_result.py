
"""Tests for verification result models."""

import pytest

from src.models.observation_context import CoverageScope
from src.models.verification_criteria import CriterionRequirement
from src.models.verification_result import (
    CriterionResult,
    CriterionResultStatus,
    VerificationResult,
    VerificationVerdict,
)


def create_criterion_result(
    criterion_id="criterion_01",
    requirement=CriterionRequirement.REQUIRED,
    is_mandatory=True,
    status=CriterionResultStatus.SATISFIED,
    explanation="The criterion is supported by sufficient evidence.",
    event_ids=("event_01",),
    evidence_ids=("evidence_01",),
):
    return CriterionResult(
        criterion_id=criterion_id,
        requirement=requirement,
        is_mandatory=is_mandatory,
        status=status,
        explanation=explanation,
        event_ids=event_ids,
        evidence_ids=evidence_ids,
    )


def create_verification_result(
    verdict=VerificationVerdict.VERIFIED,
    criterion_results=None,
    summary="The task satisfies all mandatory criteria.",
    coverage_scope=CoverageScope.COMPLETE_TASK,
    covers_entire_video=True,
):
    if criterion_results is None:
        criterion_results = (
            create_criterion_result(),
        )

    return VerificationResult(
        verdict=verdict,
        criterion_results=criterion_results,
        summary=summary,
        coverage_scope=coverage_scope,
        covers_entire_video=covers_entire_video,
    )


# CriterionResult validation


def test_creates_valid_criterion_result():
    result = create_criterion_result()

    assert result.criterion_id == "criterion_01"
    assert result.status == CriterionResultStatus.SATISFIED
    assert result.requirement == CriterionRequirement.REQUIRED
    assert result.is_mandatory is True


def test_creates_criterion_result_without_event_ids():
    result = create_criterion_result(event_ids=())

    assert result.event_ids == ()


def test_creates_criterion_result_without_evidence_ids():
    result = create_criterion_result(evidence_ids=())

    assert result.evidence_ids == ()


@pytest.mark.parametrize(
    "field,value",
    [
        ("criterion_id", ""),
        ("criterion_id", "   "),
        ("explanation", ""),
        ("explanation", "   "),
    ],
)
def test_rejects_empty_criterion_result_strings(field, value):
    with pytest.raises(ValueError):
        create_criterion_result(**{field: value})


def test_rejects_invalid_criterion_requirement():
    with pytest.raises(ValueError):
        create_criterion_result(requirement="REQUIRED")


def test_rejects_invalid_criterion_status():
    with pytest.raises(ValueError):
        create_criterion_result(status="SATISFIED")


@pytest.mark.parametrize("value", [1, 0, None, "true"])
def test_rejects_non_boolean_mandatory_flag(value):
    with pytest.raises(ValueError):
        create_criterion_result(is_mandatory=value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_ids", ["event_01"]),
        ("event_ids", "event_01"),
        ("evidence_ids", ["evidence_01"]),
        ("evidence_ids", "evidence_01"),
    ],
)
def test_rejects_non_tuple_identifiers(field, value):
    with pytest.raises(ValueError):
        create_criterion_result(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_ids", ("",)),
        ("event_ids", ("   ",)),
        ("event_ids", (None,)),
        ("evidence_ids", ("",)),
        ("evidence_ids", ("   ",)),
        ("evidence_ids", (None,)),
    ],
)
def test_rejects_invalid_identifiers(field, value):
    with pytest.raises(ValueError):
        create_criterion_result(**{field: value})


def test_rejects_duplicate_event_ids():
    with pytest.raises(ValueError):
        create_criterion_result(
            event_ids=("event_01", "event_01")
        )


def test_rejects_duplicate_evidence_ids():
    with pytest.raises(ValueError):
        create_criterion_result(
            evidence_ids=("evidence_01", "evidence_01")
        )


def test_criterion_result_is_immutable():
    result = create_criterion_result()

    with pytest.raises(AttributeError):
        result.criterion_id = "modified"


# VerificationResult validation


def test_creates_verified_result():
    result = create_verification_result()

    assert result.verdict == VerificationVerdict.VERIFIED
    assert result.coverage_scope == CoverageScope.COMPLETE_TASK
    assert result.covers_entire_video is True


def test_creates_failed_result():
    criterion = create_criterion_result(
        status=CriterionResultStatus.VIOLATED,
    )

    result = create_verification_result(
        verdict=VerificationVerdict.FAILED,
        criterion_results=(criterion,),
        summary="A mandatory criterion was violated.",
    )

    assert result.verdict == VerificationVerdict.FAILED


def test_creates_insufficient_evidence_result():
    criterion = create_criterion_result(
        status=CriterionResultStatus.INSUFFICIENT_EVIDENCE,
        event_ids=(),
        evidence_ids=(),
    )

    result = create_verification_result(
        verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
        criterion_results=(criterion,),
        summary="There is insufficient evidence.",
        coverage_scope=CoverageScope.PARTIAL_SEGMENT,
        covers_entire_video=False,
    )

    assert result.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE


def test_rejects_invalid_verdict():
    with pytest.raises(ValueError):
        create_verification_result(verdict="VERIFIED")


def test_rejects_empty_criterion_results():
    with pytest.raises(ValueError):
        create_verification_result(criterion_results=())


def test_rejects_non_tuple_criterion_results():
    with pytest.raises(ValueError):
        create_verification_result(
            criterion_results=[
                create_criterion_result()
            ]
        )


def test_rejects_invalid_criterion_result_items():
    with pytest.raises(ValueError):
        create_verification_result(
            criterion_results=("invalid",)
        )


def test_rejects_duplicate_criterion_ids():
    first = create_criterion_result(criterion_id="duplicate")
    second = create_criterion_result(criterion_id="duplicate")

    with pytest.raises(ValueError):
        create_verification_result(
            criterion_results=(first, second)
        )


def test_rejects_empty_summary():
    with pytest.raises(ValueError):
        create_verification_result(summary="   ")


# Global verdict consistency


def test_verified_requires_all_mandatory_criteria_satisfied():
    satisfied = create_criterion_result(
        criterion_id="criterion_01",
        status=CriterionResultStatus.SATISFIED,
    )

    unresolved = create_criterion_result(
        criterion_id="criterion_02",
        status=CriterionResultStatus.INSUFFICIENT_EVIDENCE,
    )

    with pytest.raises(ValueError):
        create_verification_result(
            verdict=VerificationVerdict.VERIFIED,
            criterion_results=(satisfied, unresolved),
        )


def test_mandatory_violation_requires_failed_verdict():
    criterion = create_criterion_result(
        status=CriterionResultStatus.VIOLATED,
    )

    with pytest.raises(ValueError):
        create_verification_result(
            verdict=VerificationVerdict.VERIFIED,
            criterion_results=(criterion,),
        )


def test_optional_required_criterion_does_not_block_verification():
    mandatory = create_criterion_result(
        criterion_id="mandatory",
        is_mandatory=True,
        status=CriterionResultStatus.SATISFIED,
    )

    optional = create_criterion_result(
        criterion_id="optional",
        is_mandatory=False,
        status=CriterionResultStatus.INSUFFICIENT_EVIDENCE,
        event_ids=(),
        evidence_ids=(),
    )

    result = create_verification_result(
        verdict=VerificationVerdict.VERIFIED,
        criterion_results=(mandatory, optional),
    )

    assert result.verdict == VerificationVerdict.VERIFIED


def test_optional_required_violation_does_not_block_verification():
    mandatory = create_criterion_result(
        criterion_id="mandatory",
        status=CriterionResultStatus.SATISFIED,
    )

    optional = create_criterion_result(
        criterion_id="optional",
        is_mandatory=False,
        status=CriterionResultStatus.VIOLATED,
    )

    result = create_verification_result(
        verdict=VerificationVerdict.VERIFIED,
        criterion_results=(mandatory, optional),
    )

    assert result.verdict == VerificationVerdict.VERIFIED


def test_optional_forbidden_violation_requires_failed_verdict():
    mandatory = create_criterion_result(
        criterion_id="mandatory",
        status=CriterionResultStatus.SATISFIED,
    )

    forbidden = create_criterion_result(
        criterion_id="forbidden",
        requirement=CriterionRequirement.FORBIDDEN,
        is_mandatory=False,
        status=CriterionResultStatus.VIOLATED,
    )

    result = create_verification_result(
        verdict=VerificationVerdict.FAILED,
        criterion_results=(mandatory, forbidden),
        summary="A prohibited event was detected.",
    )

    assert result.verdict == VerificationVerdict.FAILED


def test_all_optional_criteria_cannot_produce_verified():
    optional = create_criterion_result(
        is_mandatory=False,
        status=CriterionResultStatus.SATISFIED,
    )

    with pytest.raises(ValueError):
        create_verification_result(
            verdict=VerificationVerdict.VERIFIED,
            criterion_results=(optional,),
        )


def test_all_optional_criteria_require_insufficient_verdict():
    optional = create_criterion_result(
        is_mandatory=False,
        status=CriterionResultStatus.SATISFIED,
    )

    result = create_verification_result(
        verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
        criterion_results=(optional,),
        summary="No mandatory criteria were defined.",
    )

    assert result.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE


def test_failed_verdict_takes_priority_over_insufficient_evidence():
    violated = create_criterion_result(
        criterion_id="violated",
        status=CriterionResultStatus.VIOLATED,
    )

    unresolved = create_criterion_result(
        criterion_id="unresolved",
        status=CriterionResultStatus.INSUFFICIENT_EVIDENCE,
        event_ids=(),
        evidence_ids=(),
    )

    result = create_verification_result(
        verdict=VerificationVerdict.FAILED,
        criterion_results=(violated, unresolved),
        summary="A mandatory violation was established.",
    )

    assert result.verdict == VerificationVerdict.FAILED


# Coverage validation


@pytest.mark.parametrize(
    "coverage_scope,covers_entire_video",
    [
        (CoverageScope.UNKNOWN, False),
        (CoverageScope.PARTIAL_SEGMENT, False),
        (CoverageScope.COMPLETE_TASK, False),
        (CoverageScope.PARTIAL_SEGMENT, True),
    ],
)
def test_verified_requires_complete_task_coverage(
    coverage_scope,
    covers_entire_video,
):
    with pytest.raises(ValueError):
        create_verification_result(
            verdict=VerificationVerdict.VERIFIED,
            coverage_scope=coverage_scope,
            covers_entire_video=covers_entire_video,
        )


def test_complete_task_coverage_allows_verified_result():
    result = create_verification_result(
        verdict=VerificationVerdict.VERIFIED,
        coverage_scope=CoverageScope.COMPLETE_TASK,
        covers_entire_video=True,
    )

    assert result.verdict == VerificationVerdict.VERIFIED


def test_failed_result_does_not_require_complete_coverage():
    violated = create_criterion_result(
        status=CriterionResultStatus.VIOLATED,
    )

    result = create_verification_result(
        verdict=VerificationVerdict.FAILED,
        criterion_results=(violated,),
        coverage_scope=CoverageScope.PARTIAL_SEGMENT,
        covers_entire_video=False,
    )

    assert result.verdict == VerificationVerdict.FAILED


@pytest.mark.parametrize(
    "coverage_scope",
    [
        "COMPLETE_TASK",
        "PARTIAL_SEGMENT",
        "UNKNOWN",
        None,
    ],
)
def test_rejects_invalid_coverage_scope(coverage_scope):
    with pytest.raises(ValueError):
        create_verification_result(
            coverage_scope=coverage_scope,
        )


@pytest.mark.parametrize(
    "covers_entire_video",
    [
        1,
        0,
        None,
        "true",
    ],
)
def test_rejects_non_boolean_video_coverage(covers_entire_video):
    with pytest.raises(ValueError):
        create_verification_result(
            covers_entire_video=covers_entire_video,
        )


def test_verification_result_is_immutable():
    result = create_verification_result()

    with pytest.raises(AttributeError):
        result.verdict = VerificationVerdict.FAILED
