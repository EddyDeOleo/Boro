"""Tests for the ReportOverview presentation model."""

from dataclasses import replace

import pytest

from src.presentation.models.report_overview import ReportOverview


@pytest.fixture
def valid_overview():
    return ReportOverview(
        verdict="INSUFFICIENT_EVIDENCE",
        summary="The execution could not be fully verified.",
        coverage_scope="PARTIAL_SEGMENT",
        covers_entire_context=False,
        start_time_seconds=0.0,
        end_time_seconds=2.0,
        duration_seconds=2.0,
        fps=30.0,
        context_frame_count=300,
        observed_frame_count=61,
    )


def test_creates_valid_report_overview(valid_overview):
    assert valid_overview.verdict == "INSUFFICIENT_EVIDENCE"
    assert valid_overview.summary == (
        "The execution could not be fully verified."
    )
    assert valid_overview.coverage_scope == "PARTIAL_SEGMENT"
    assert valid_overview.covers_entire_context is False
    assert valid_overview.start_time_seconds == 0.0
    assert valid_overview.end_time_seconds == 2.0
    assert valid_overview.duration_seconds == 2.0
    assert valid_overview.fps == 30.0
    assert valid_overview.context_frame_count == 300
    assert valid_overview.observed_frame_count == 61


@pytest.mark.parametrize(
    "field,value",
    [
        ("verdict", ""),
        ("summary", ""),
        ("coverage_scope", ""),
    ],
)
def test_rejects_empty_required_strings(valid_overview, field, value):
    with pytest.raises(ValueError):
        replace(valid_overview, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_time_seconds", -1.0),
        ("end_time_seconds", -1.0),
        ("duration_seconds", -1.0),
        ("fps", 0.0),
        ("fps", -30.0),
        ("context_frame_count", 0),
        ("context_frame_count", -1),
        ("observed_frame_count", 0),
        ("observed_frame_count", -1),
    ],
)
def test_rejects_invalid_numeric_values(valid_overview, field, value):
    with pytest.raises(ValueError):
        replace(valid_overview, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_time_seconds", float("inf")),
        ("end_time_seconds", float("nan")),
        ("duration_seconds", float("inf")),
        ("fps", float("nan")),
        ("start_time_seconds", True),
        ("end_time_seconds", False),
        ("duration_seconds", True),
        ("fps", False),
    ],
)
def test_rejects_non_finite_or_boolean_numeric_values(
    valid_overview,
    field,
    value,
):
    with pytest.raises(ValueError):
        replace(valid_overview, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_frame_count", True),
        ("context_frame_count", 300.0),
        ("observed_frame_count", False),
        ("observed_frame_count", 61.0),
    ],
)
def test_rejects_non_integer_frame_counts(
    valid_overview,
    field,
    value,
):
    with pytest.raises(ValueError):
        replace(valid_overview, **{field: value})


def test_rejects_observed_frames_exceeding_context_frame_count(
    valid_overview,
):
    with pytest.raises(ValueError):
        replace(
            valid_overview,
            observed_frame_count=301,
        )


def test_accepts_observing_entire_context(valid_overview):
    overview = replace(
        valid_overview,
        covers_entire_context=True,
        context_frame_count=61,
        observed_frame_count=61,
    )

    assert overview.context_frame_count == 61
    assert overview.observed_frame_count == 61
    assert overview.covers_entire_context is True


def test_rejects_end_time_before_start_time(valid_overview):
    with pytest.raises(ValueError):
        replace(
            valid_overview,
            start_time_seconds=2.0,
            end_time_seconds=1.0,
            duration_seconds=1.0,
        )


def test_rejects_duration_exceeding_time_interval(valid_overview):
    with pytest.raises(ValueError):
        replace(
            valid_overview,
            start_time_seconds=0.0,
            end_time_seconds=2.0,
            duration_seconds=2.5,
        )


def test_rejects_non_boolean_context_coverage(valid_overview):
    with pytest.raises(ValueError):
        replace(
            valid_overview,
            covers_entire_context=1,
        )


def test_report_overview_is_immutable(valid_overview):
    with pytest.raises((AttributeError, TypeError)):
        valid_overview.verdict = "VERIFIED"