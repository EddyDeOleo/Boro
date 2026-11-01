

"""Tests for the PresentationReport model."""

import pytest

from src.models.event import EventStatus
from src.models.verification_criteria import CriterionRequirement
from src.models.verification_result import CriterionResultStatus

from src.presentation.models.criterion_view import CriterionView
from src.presentation.models.evidence_view import EvidenceView
from src.presentation.models.presentation_report import PresentationReport
from src.presentation.models.report_overview import ReportOverview
from src.presentation.models.timeline_event_view import TimelineEventView


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
        context_frame_count=61,
        observed_frame_count=61,
    )


@pytest.fixture
def valid_evidence():
    return EvidenceView(
        evidence_id="evidence-001",
        evidence_type="STABLE_IN_REGION",
        description="The object remained stable inside the target region.",
        frame_start=10,
        frame_end=40,
        timestamp_start_seconds=0.333333,
        timestamp_end_seconds=1.333333,
        observation_ids=("observation-001",),
        entity_ids=("entity-001",),
        metrics={"mean_speed": 0.12},
        limitations=("Visual evidence only.",),
    )


@pytest.fixture
def valid_event():
    return TimelineEventView(
        event_id="event-001",
        event_type="STABLE_PLACEMENT",
        status=EventStatus.DETECTED,
        frame_start=10,
        frame_end=40,
        timestamp_start=0.333333,
        timestamp_end=1.333333,
        entity_ids=("entity-001",),
        confidence=0.95,
        evidence_ids=("evidence-001",),
        attributes={"region_id": "target_zone_01"},
    )


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


@pytest.fixture
def valid_report(
    valid_overview,
    valid_criterion,
    valid_event,
    valid_evidence,
):
    return PresentationReport(
        overview=valid_overview,
        criteria=(valid_criterion,),
        timeline=(valid_event,),
        evidence=(valid_evidence,),
        limitations=("Visual evidence only.",),
    )


def test_creates_valid_presentation_report(valid_report):
    assert isinstance(valid_report.overview, ReportOverview)
    assert len(valid_report.criteria) == 1
    assert len(valid_report.timeline) == 1
    assert len(valid_report.evidence) == 1

    assert valid_report.criteria[0].criterion_id == (
        "stable_target_placement"
    )
    assert valid_report.timeline[0].event_id == "event-001"
    assert valid_report.evidence[0].evidence_id == "evidence-001"

    assert valid_report.limitations == ("Visual evidence only.",)


def test_allows_empty_collections(valid_overview):
    report = PresentationReport(
        overview=valid_overview,
        criteria=(),
        timeline=(),
        evidence=(),
        limitations=(),
    )

    assert report.criteria == ()
    assert report.timeline == ()
    assert report.evidence == ()
    assert report.limitations == ()


def test_rejects_invalid_overview(
    valid_criterion,
    valid_event,
    valid_evidence,
):
    with pytest.raises(ValueError, match="overview"):
        PresentationReport(
            overview="invalid",
            criteria=(valid_criterion,),
            timeline=(valid_event,),
            evidence=(valid_evidence,),
            limitations=(),
        )


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("criteria", [None], "criteria must be a tuple"),
        ("timeline", [None], "timeline must be a tuple"),
        ("evidence", [None], "evidence must be a tuple"),
    ],
)
def test_rejects_non_tuple_collections(
    valid_overview,
    field,
    value,
    expected_message,
):
    values = {
        "overview": valid_overview,
        "criteria": (),
        "timeline": (),
        "evidence": (),
        "limitations": (),
    }
    values[field] = value

    with pytest.raises(ValueError, match=expected_message):
        PresentationReport(**values)


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("criteria", (None,), "Every item in criteria"),
        ("timeline", (None,), "Every item in timeline"),
        ("evidence", (None,), "Every item in evidence"),
    ],
)
def test_rejects_invalid_model_instances(
    valid_overview,
    field,
    value,
    expected_message,
):
    values = {
        "overview": valid_overview,
        "criteria": (),
        "timeline": (),
        "evidence": (),
        "limitations": (),
    }
    values[field] = value

    with pytest.raises(ValueError, match=expected_message):
        PresentationReport(**values)


def test_rejects_non_tuple_limitations(valid_overview):
    with pytest.raises(ValueError, match="limitations must be a tuple"):
        PresentationReport(
            overview=valid_overview,
            criteria=(),
            timeline=(),
            evidence=(),
            limitations=["Visual evidence only."],
        )


@pytest.mark.parametrize(
    "limitations",
    [
        ("",),
        ("   ",),
        ("Visual evidence only.", "Visual evidence only."),
    ],
)
def test_rejects_invalid_limitations(
    valid_overview,
    limitations,
):
    with pytest.raises(ValueError):
        PresentationReport(
            overview=valid_overview,
            criteria=(),
            timeline=(),
            evidence=(),
            limitations=limitations,
        )


def test_rejects_duplicate_criterion_ids(
    valid_overview,
    valid_criterion,
):
    duplicate = CriterionView(
        criterion_id=valid_criterion.criterion_id,
        requirement=valid_criterion.requirement,
        status=valid_criterion.status,
        is_mandatory=valid_criterion.is_mandatory,
        explanation="Another criterion with the same ID.",
        event_ids=(),
        evidence_ids=(),
    )

    with pytest.raises(ValueError, match="criteria must contain unique IDs"):
        PresentationReport(
            overview=valid_overview,
            criteria=(valid_criterion, duplicate),
            timeline=(),
            evidence=(),
            limitations=(),
        )


def test_rejects_duplicate_event_ids(
    valid_overview,
    valid_event,
):
    duplicate = TimelineEventView(
        event_id=valid_event.event_id,
        event_type="OBJECT_MOVING",
        status=EventStatus.NOT_OBSERVABLE,
        frame_start=0,
        frame_end=5,
        timestamp_start=0.0,
        timestamp_end=0.166667,
        entity_ids=(),
        confidence=None,
        evidence_ids=(),
        attributes={},
    )

    with pytest.raises(ValueError, match="timeline must contain unique IDs"):
        PresentationReport(
            overview=valid_overview,
            criteria=(),
            timeline=(valid_event, duplicate),
            evidence=(),
            limitations=(),
        )


def test_rejects_duplicate_evidence_ids(
    valid_overview,
    valid_evidence,
):
    duplicate = EvidenceView(
        evidence_id=valid_evidence.evidence_id,
        evidence_type="MOTION",
        description="Another evidence item with the same ID.",
        frame_start=50,
        frame_end=60,
        timestamp_start_seconds=1.666667,
        timestamp_end_seconds=2.0,
    )

    with pytest.raises(ValueError, match="evidence must contain unique IDs"):
        PresentationReport(
            overview=valid_overview,
            criteria=(),
            timeline=(),
            evidence=(valid_evidence, duplicate),
            limitations=(),
        )


def test_rejects_criterion_referencing_unknown_event(
    valid_overview,
    valid_criterion,
    valid_evidence,
):
    with pytest.raises(
        ValueError,
        match="references unknown events",
    ):
        PresentationReport(
            overview=valid_overview,
            criteria=(valid_criterion,),
            timeline=(),
            evidence=(valid_evidence,),
            limitations=(),
        )


def test_rejects_criterion_referencing_unknown_evidence(
    valid_overview,
    valid_event,
    valid_criterion,
):
    criterion = CriterionView(
        criterion_id="stable_target_placement",
        requirement=CriterionRequirement.REQUIRED.value,
        status=CriterionResultStatus.INSUFFICIENT_EVIDENCE.value,
        is_mandatory=True,
        explanation="Stable placement could not be confirmed.",
        event_ids=("event-001",),
        evidence_ids=("missing-evidence",),
    )

    with pytest.raises(
        ValueError,
        match="references unknown evidence",
    ):
        PresentationReport(
            overview=valid_overview,
            criteria=(criterion,),
            timeline=(valid_event,),
            evidence=(),
            limitations=(),
        )


def test_rejects_criterion_evidence_not_associated_with_its_events(
    valid_overview,
):
    event = TimelineEventView(
        event_id="event-001",
        event_type="STABLE_PLACEMENT",
        status=EventStatus.DETECTED,
        frame_start=10,
        frame_end=40,
        timestamp_start=0.333333,
        timestamp_end=1.333333,
        entity_ids=("entity-001",),
        confidence=0.95,
        evidence_ids=("evidence-001",),
        attributes={},
    )

    unrelated_evidence = EvidenceView(
        evidence_id="evidence-002",
        evidence_type="MOTION",
        description="Evidence unrelated to the referenced event.",
        frame_start=10,
        frame_end=40,
        timestamp_start_seconds=0.333333,
        timestamp_end_seconds=1.333333,
    )

    criterion = CriterionView(
        criterion_id="stable_target_placement",
        requirement=CriterionRequirement.REQUIRED.value,
        status=CriterionResultStatus.INSUFFICIENT_EVIDENCE.value,
        is_mandatory=True,
        explanation="Stable placement could not be confirmed.",
        event_ids=("event-001",),
        evidence_ids=("evidence-002",),
    )

    with pytest.raises(
        ValueError,
        match="not associated with its events",
    ):
        PresentationReport(
            overview=valid_overview,
            criteria=(criterion,),
            timeline=(event,),
            evidence=(
                EvidenceView(
                    evidence_id="evidence-001",
                    evidence_type="STABLE_IN_REGION",
                    description="Evidence associated with the event.",
                    frame_start=10,
                    frame_end=40,
                    timestamp_start_seconds=0.333333,
                    timestamp_end_seconds=1.333333,
                ),
                unrelated_evidence,
            ),
            limitations=(),
        )


def test_rejects_event_referencing_unknown_evidence(
    valid_overview,
):
    event = TimelineEventView(
        event_id="event-001",
        event_type="STABLE_PLACEMENT",
        status=EventStatus.DETECTED,
        frame_start=10,
        frame_end=40,
        timestamp_start=0.333333,
        timestamp_end=1.333333,
        entity_ids=("entity-001",),
        confidence=0.95,
        evidence_ids=("missing-evidence",),
        attributes={},
    )

    with pytest.raises(
        ValueError,
        match="references unknown evidence",
    ):
        PresentationReport(
            overview=valid_overview,
            criteria=(),
            timeline=(event,),
            evidence=(),
            limitations=(),
        )


def test_allows_criterion_to_reference_subset_of_event_evidence(
    valid_overview,
):
    first_evidence = EvidenceView(
        evidence_id="evidence-001",
        evidence_type="MOTION",
        description="Object motion evidence.",
        frame_start=0,
        frame_end=10,
        timestamp_start_seconds=0.0,
        timestamp_end_seconds=0.333333,
    )

    second_evidence = EvidenceView(
        evidence_id="evidence-002",
        evidence_type="STABLE_IN_REGION",
        description="Stable placement evidence.",
        frame_start=10,
        frame_end=40,
        timestamp_start_seconds=0.333333,
        timestamp_end_seconds=1.333333,
    )

    event = TimelineEventView(
        event_id="event-001",
        event_type="STABLE_PLACEMENT",
        status=EventStatus.DETECTED,
        frame_start=0,
        frame_end=40,
        timestamp_start=0.0,
        timestamp_end=1.333333,
        entity_ids=("entity-001",),
        confidence=0.95,
        evidence_ids=("evidence-001", "evidence-002"),
        attributes={},
    )

    criterion = CriterionView(
        criterion_id="stable_target_placement",
        requirement=CriterionRequirement.REQUIRED.value,
        status=CriterionResultStatus.INSUFFICIENT_EVIDENCE.value,
        is_mandatory=True,
        explanation="Stable placement could not be confirmed.",
        event_ids=("event-001",),
        evidence_ids=("evidence-002",),
    )

    report = PresentationReport(
        overview=valid_overview,
        criteria=(criterion,),
        timeline=(event,),
        evidence=(first_evidence, second_evidence),
        limitations=(),
    )

    assert report.criteria[0].evidence_ids == ("evidence-002",)


def test_rejects_events_out_of_chronological_order(
    valid_overview,
    valid_evidence,
):
    later_event = TimelineEventView(
        event_id="event-002",
        event_type="STABLE_PLACEMENT",
        status=EventStatus.DETECTED,
        frame_start=30,
        frame_end=40,
        timestamp_start=1.0,
        timestamp_end=1.333333,
        entity_ids=("entity-001",),
        confidence=0.95,
        evidence_ids=("evidence-001",),
        attributes={},
    )

    earlier_event = TimelineEventView(
        event_id="event-001",
        event_type="OBJECT_MOVING",
        status=EventStatus.DETECTED,
        frame_start=0,
        frame_end=20,
        timestamp_start=0.0,
        timestamp_end=0.666667,
        entity_ids=("entity-001",),
        confidence=0.9,
        evidence_ids=("evidence-001",),
        attributes={},
    )

    with pytest.raises(
        ValueError,
        match="chronologically ordered",
    ):
        PresentationReport(
            overview=valid_overview,
            criteria=(),
            timeline=(later_event, earlier_event),
            evidence=(valid_evidence,),
            limitations=(),
        )


def test_rejects_events_with_incorrect_tie_break_order(
    valid_overview,
    valid_evidence,
):
    event_b = TimelineEventView(
        event_id="event-b",
        event_type="OBJECT_MOVING",
        status=EventStatus.DETECTED,
        frame_start=0,
        frame_end=10,
        timestamp_start=0.0,
        timestamp_end=0.333333,
        entity_ids=("entity-001",),
        confidence=0.9,
        evidence_ids=("evidence-001",),
        attributes={},
    )

    event_a = TimelineEventView(
        event_id="event-a",
        event_type="OBJECT_MOVING",
        status=EventStatus.DETECTED,
        frame_start=0,
        frame_end=10,
        timestamp_start=0.0,
        timestamp_end=0.333333,
        entity_ids=("entity-001",),
        confidence=0.9,
        evidence_ids=("evidence-001",),
        attributes={},
    )

    with pytest.raises(
        ValueError,
        match="chronologically ordered",
    ):
        PresentationReport(
            overview=valid_overview,
            criteria=(),
            timeline=(event_b, event_a),
            evidence=(valid_evidence,),
            limitations=(),
        )


def test_rejects_evidence_out_of_frame_order(
    valid_overview,
):
    later_evidence = EvidenceView(
        evidence_id="evidence-002",
        evidence_type="STABLE_IN_REGION",
        description="Evidence from a later frame interval.",
        frame_start=30,
        frame_end=40,
        timestamp_start_seconds=1.0,
        timestamp_end_seconds=1.333333,
    )

    earlier_evidence = EvidenceView(
        evidence_id="evidence-001",
        evidence_type="MOTION",
        description="Evidence from an earlier frame interval.",
        frame_start=0,
        frame_end=20,
        timestamp_start_seconds=0.0,
        timestamp_end_seconds=0.666667,
    )

    with pytest.raises(
        ValueError,
        match="ordered by frame interval",
    ):
        PresentationReport(
            overview=valid_overview,
            criteria=(),
            timeline=(),
            evidence=(later_evidence, earlier_evidence),
            limitations=(),
        )


def test_rejects_evidence_with_incorrect_tie_break_order(
    valid_overview,
):
    evidence_b = EvidenceView(
        evidence_id="evidence-b",
        evidence_type="MOTION",
        description="Evidence B.",
        frame_start=0,
        frame_end=10,
        timestamp_start_seconds=0.0,
        timestamp_end_seconds=0.333333,
    )

    evidence_a = EvidenceView(
        evidence_id="evidence-a",
        evidence_type="MOTION",
        description="Evidence A.",
        frame_start=0,
        frame_end=10,
        timestamp_start_seconds=0.0,
        timestamp_end_seconds=0.333333,
    )

    with pytest.raises(
        ValueError,
        match="ordered by frame interval",
    ):
        PresentationReport(
            overview=valid_overview,
            criteria=(),
            timeline=(),
            evidence=(evidence_b, evidence_a),
            limitations=(),
        )


def test_presentation_report_is_immutable(valid_report):
    with pytest.raises((AttributeError, TypeError)):
        valid_report.overview = None
