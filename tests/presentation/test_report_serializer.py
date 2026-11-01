
"""Tests for the presentation report JSON serializer."""

import json

import pytest

from src.models.event import EventStatus
from src.models.verification_criteria import CriterionRequirement
from src.models.verification_result import CriterionResultStatus
from src.presentation.models import (
    CriterionView,
    EvidenceView,
    PresentationReport,
    ReportOverview,
    TimelineEventView,
)
from src.presentation.report_serializer import (
    ReportSerializationError,
    ReportSerializer,
)


@pytest.fixture
def presentation_report() -> PresentationReport:
    evidence = EvidenceView(
        evidence_id="evidence_001",
        evidence_type="MASK_OBSERVATION",
        description="The object mask was observed in the target area.",
        frame_start=20,
        frame_end=25,
        timestamp_start_seconds=6.4667,
        timestamp_end_seconds=6.6333,
        observation_ids=("observation_001", "observation_002"),
        entity_ids=("red_cube",),
        metrics={
            "mean_overlap_ratio": 0.94,
            "observed_frames": 6,
            "region_confirmed": True,
        },
        limitations=("Visual overlap does not prove physical contact.",),
    )

    event = TimelineEventView(
        event_id="event_001",
        event_type="REGION_ENTRY",
        status=EventStatus.DETECTED.value,
        frame_start=20,
        frame_end=25,
        timestamp_start=6.4667,
        timestamp_end=6.6333,
        entity_ids=("red_cube",),
        confidence=0.92,
        evidence_ids=("evidence_001",),
        attributes={
            "overlap_ratio": 0.94,
            "region_id": "target_zone_01",
            "transition_detected": True,
        },
    )

    criterion = CriterionView(
        criterion_id="target_entry",
        requirement=CriterionRequirement.REQUIRED.value,
        status=CriterionResultStatus.SATISFIED.value,
        is_mandatory=True,
        explanation="The object entered the target region.",
        event_ids=("event_001",),
        evidence_ids=("evidence_001",),
    )

    overview = ReportOverview(
        verdict="INSUFFICIENT_EVIDENCE",
        summary="The object entered the target region, but stable placement "
        "could not be verified.",
        coverage_scope="PARTIAL_SEGMENT",
        covers_entire_context=True,
        start_time_seconds=5.8,
        end_time_seconds=7.4667,
        duration_seconds=1.6667,
        fps=30.0,
        context_frame_count=51,
        observed_frame_count=51,
    )

    return PresentationReport(
        overview=overview,
        criteria=(criterion,),
        timeline=(event,),
        evidence=(evidence,),
        limitations=(
            "The analyzed clip does not cover the complete task.",
            "Visual evidence cannot establish physical contact.",
        ),
    )


def test_to_dict_returns_json_compatible_dictionary(
    presentation_report: PresentationReport,
) -> None:
    result = ReportSerializer.to_dict(presentation_report)

    assert result["schema_version"] == ReportSerializer.SCHEMA_VERSION
    assert set(result) == {"schema_version", "report"}

    report_data = result["report"]

    assert set(report_data) == {
        "overview",
        "criteria",
        "timeline",
        "evidence",
        "limitations",
    }

    assert isinstance(report_data["criteria"], list)
    assert isinstance(report_data["timeline"], list)
    assert isinstance(report_data["evidence"], list)

    # The returned structure must be accepted by the standard JSON encoder.
    json.dumps(result, allow_nan=False)


def test_to_json_returns_valid_json(
    presentation_report: PresentationReport,
) -> None:
    result = ReportSerializer.to_json(presentation_report)

    parsed = json.loads(result)

    assert parsed["schema_version"] == ReportSerializer.SCHEMA_VERSION
    assert parsed["report"]["overview"]["verdict"] == (
        presentation_report.overview.verdict
    )


def test_from_dict_restores_presentation_report(
    presentation_report: PresentationReport,
) -> None:
    serialized = ReportSerializer.to_dict(presentation_report)

    restored = ReportSerializer.from_dict(serialized)

    assert isinstance(restored, PresentationReport)
    assert restored == presentation_report


def test_json_round_trip_preserves_report(
    presentation_report: PresentationReport,
) -> None:
    serialized = ReportSerializer.to_json(presentation_report)

    restored = ReportSerializer.from_json(serialized)

    assert restored == presentation_report


def test_round_trip_preserves_timeline_attributes(
    presentation_report: PresentationReport,
) -> None:
    serialized = ReportSerializer.to_json(presentation_report)
    restored = ReportSerializer.from_json(serialized)

    attributes = restored.timeline[0].attributes

    assert attributes["overlap_ratio"] == 0.94
    assert attributes["region_id"] == "target_zone_01"
    assert attributes["transition_detected"] is True

    with pytest.raises(TypeError):
        attributes["new_attribute"] = "not allowed"


def test_round_trip_preserves_evidence_metrics(
    presentation_report: PresentationReport,
) -> None:
    serialized = ReportSerializer.to_json(presentation_report)
    restored = ReportSerializer.from_json(serialized)

    metrics = restored.evidence[0].metrics

    assert metrics["mean_overlap_ratio"] == 0.94
    assert metrics["observed_frames"] == 6
    assert metrics["region_confirmed"] is True

    with pytest.raises(TypeError):
        metrics["new_metric"] = 1


def test_to_dict_rejects_wrong_input_type() -> None:
    with pytest.raises(ReportSerializationError):
        ReportSerializer.to_dict({"not": "a report"})


def test_from_dict_rejects_non_mapping_input() -> None:
    with pytest.raises(ReportSerializationError):
        ReportSerializer.from_dict([])


def test_from_dict_rejects_missing_top_level_key(
    presentation_report: PresentationReport,
) -> None:
    serialized = ReportSerializer.to_dict(presentation_report)
    del serialized["report"]

    with pytest.raises(
        ReportSerializationError,
        match="missing keys",
    ):
        ReportSerializer.from_dict(serialized)


def test_from_dict_rejects_unexpected_top_level_key(
    presentation_report: PresentationReport,
) -> None:
    serialized = ReportSerializer.to_dict(presentation_report)
    serialized["unexpected"] = "value"

    with pytest.raises(
        ReportSerializationError,
        match="unexpected keys",
    ):
        ReportSerializer.from_dict(serialized)


def test_from_dict_rejects_unsupported_schema_version(
    presentation_report: PresentationReport,
) -> None:
    serialized = ReportSerializer.to_dict(presentation_report)
    serialized["schema_version"] = "2.0"

    with pytest.raises(
        ReportSerializationError,
        match="Unsupported schema version",
    ):
        ReportSerializer.from_dict(serialized)


def test_from_dict_rejects_unexpected_overview_field(
    presentation_report: PresentationReport,
) -> None:
    serialized = ReportSerializer.to_dict(presentation_report)
    serialized["report"]["overview"]["unexpected"] = "value"

    with pytest.raises(
        ReportSerializationError,
        match="unexpected keys",
    ):
        ReportSerializer.from_dict(serialized)


def test_from_dict_rejects_invalid_criteria_collection(
    presentation_report: PresentationReport,
) -> None:
    serialized = ReportSerializer.to_dict(presentation_report)
    serialized["report"]["criteria"] = "not-an-array"

    with pytest.raises(
        ReportSerializationError,
        match="must be an array",
    ):
        ReportSerializer.from_dict(serialized)


def test_from_dict_rejects_non_string_event_ids(
    presentation_report: PresentationReport,
) -> None:
    serialized = ReportSerializer.to_dict(presentation_report)
    serialized["report"]["criteria"][0]["event_ids"] = [123]

    with pytest.raises(
        ReportSerializationError,
        match="must contain strings only",
    ):
        ReportSerializer.from_dict(serialized)


def test_from_dict_rejects_broken_criterion_event_reference(
    presentation_report: PresentationReport,
) -> None:
    serialized = ReportSerializer.to_dict(presentation_report)
    serialized["report"]["criteria"][0]["event_ids"] = [
        "event_does_not_exist"
    ]

    with pytest.raises(
        ReportSerializationError,
        match="violates PresentationReport validation rules",
    ):
        ReportSerializer.from_dict(serialized)


def test_from_json_rejects_invalid_json() -> None:
    with pytest.raises(
        ReportSerializationError,
        match="Input is not valid JSON",
    ):
        ReportSerializer.from_json('{"schema_version":')


def test_from_json_rejects_non_string_input() -> None:
    with pytest.raises(
        ReportSerializationError,
        match="must be a string",
    ):
        ReportSerializer.from_json(123)


def test_from_dict_rejects_non_finite_timeline_attribute(
    presentation_report: PresentationReport,
) -> None:
    serialized = ReportSerializer.to_dict(presentation_report)
    serialized["report"]["timeline"][0]["attributes"]["invalid"] = float(
        "nan"
    )

    with pytest.raises(
        ReportSerializationError,
        match="Invalid report.timeline",
    ):
        ReportSerializer.from_dict(serialized)
