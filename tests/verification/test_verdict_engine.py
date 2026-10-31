"""Tests for the VerdictEngine."""

import pytest

from src.models.event import EventStatus, TemporalEvent
from src.models.evidence import Evidence
from src.models.observation_context import (
    CoverageScope,
    ObservationContext,
)
from src.models.task_roles import TaskRoles
from src.models.verification_criteria import (
    CriterionRequirement,
    VerificationCriteria,
)
from src.models.verification_result import (
    CriterionResultStatus,
    VerificationVerdict,
)
from src.verification.verdict_engine import VerdictEngine


@pytest.fixture
def engine() -> VerdictEngine:
    return VerdictEngine()


@pytest.fixture
def task_roles() -> TaskRoles:
    return TaskRoles(
        entity_roles={
            "target_object": "cube_01",
            "effector": "gripper_01",
        },
        region_roles={
            "target_region": "target_zone_01",
        },
    )


@pytest.fixture
def complete_context() -> ObservationContext:
    return ObservationContext(
        fps=30.0,
        frame_count=10,
        frame_start=0,
        frame_end=9,
        time_offset_seconds=0.0,
        coverage_scope=CoverageScope.COMPLETE_TASK,
    )


@pytest.fixture
def partial_context() -> ObservationContext:
    return ObservationContext(
        fps=30.0,
        frame_count=10,
        frame_start=2,
        frame_end=7,
        time_offset_seconds=0.0,
        coverage_scope=CoverageScope.PARTIAL_SEGMENT,
    )


def make_evidence(
    evidence_id: str = "evidence_01",
    evidence_type: str = "OBJECT_MOTION",
    frame_start: int = 0,
    frame_end: int = 9,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        description="Test evidence.",
        frame_start=frame_start,
        frame_end=frame_end,
        observation_ids=(),
        entity_ids=("cube_01",),
        metrics={},
        limitations=(),
    )


def make_event(
    event_id: str = "event_01",
    event_type: str = "OBJECT_MOVING",
    status: EventStatus = EventStatus.DETECTED,
    entity_ids: tuple[str, ...] = ("cube_01",),
    region_id: str | None = None,
    evidence: tuple[Evidence, ...] | None = None,
    frame_start: int = 0,
    frame_end: int = 9,
) -> TemporalEvent:
    attributes = {}

    if region_id is not None:
        attributes["region_id"] = region_id

    if evidence is None:
        evidence = (make_evidence(),)

    return TemporalEvent(
        event_id=event_id,
        event_type=event_type,
        status=status,
        frame_start=frame_start,
        frame_end=frame_end,
        timestamp_start=frame_start / 30.0,
        timestamp_end=frame_end / 30.0,
        entity_ids=entity_ids,
        confidence=0.95 if status == EventStatus.DETECTED else None,
        evidence=evidence,
        attributes=attributes,
    )


def make_criterion(
    criterion_id: str = "criterion_01",
    event_type: str = "OBJECT_MOVING",
    requirement: CriterionRequirement = CriterionRequirement.REQUIRED,
    entity_roles: tuple[str, ...] = ("target_object",),
    region_role: str | None = None,
    required_evidence_types: tuple[str, ...] = (),
    is_mandatory: bool = True,
) -> VerificationCriteria:
    return VerificationCriteria(
        criterion_id=criterion_id,
        description="Test criterion.",
        event_type=event_type,
        requirement=requirement,
        entity_roles=entity_roles,
        region_role=region_role,
        required_evidence_types=required_evidence_types,
        is_mandatory=is_mandatory,
    )


def make_observable_absence_event(
    event_id: str = "event_absence",
    event_type: str = "OBJECT_MOVING",
    entity_ids: tuple[str, ...] = ("cube_01",),
    region_id: str | None = None,
    frame_start: int = 0,
    frame_end: int = 9,
) -> TemporalEvent:
    evidence = make_evidence(
        evidence_id="observability_01",
        evidence_type="OBSERVABILITY_CONFIRMED",
        frame_start=frame_start,
        frame_end=frame_end,
    )

    return make_event(
        event_id=event_id,
        event_type=event_type,
        status=EventStatus.NOT_DETECTED,
        entity_ids=entity_ids,
        region_id=region_id,
        evidence=(evidence,),
        frame_start=frame_start,
        frame_end=frame_end,
    )


def test_required_detected_event_satisfies_criterion(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion()
    event = make_event()

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=complete_context,
    )

    criterion_result = result.criterion_results[0]

    assert criterion_result.status == CriterionResultStatus.SATISFIED
    assert criterion_result.event_ids == ("event_01",)
    assert criterion_result.evidence_ids == ("evidence_01",)
    assert result.verdict == VerificationVerdict.VERIFIED


def test_forbidden_detected_event_causes_failure(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion(
        requirement=CriterionRequirement.FORBIDDEN,
    )
    event = make_event()

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.VIOLATED
    )
    assert result.verdict == VerificationVerdict.FAILED


def test_required_not_detected_with_complete_observability_is_violated(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion()
    event = make_observable_absence_event()

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.VIOLATED
    )
    assert result.verdict == VerificationVerdict.FAILED


def test_forbidden_not_detected_with_complete_observability_is_satisfied(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion(
        requirement=CriterionRequirement.FORBIDDEN,
    )
    event = make_observable_absence_event()

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.SATISFIED
    )
    assert result.verdict == VerificationVerdict.VERIFIED


def test_not_detected_without_observability_is_insufficient(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion()

    event = make_event(
        status=EventStatus.NOT_DETECTED,
        evidence=(
            make_evidence(
                evidence_type="GENERIC_EVIDENCE",
            ),
        ),
    )

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.INSUFFICIENT_EVIDENCE
    )
    assert result.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE


def test_not_observable_event_is_insufficient(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion()

    event = make_event(
        status=EventStatus.NOT_OBSERVABLE,
        evidence=(),
    )

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.INSUFFICIENT_EVIDENCE
    )


def test_missing_event_does_not_mean_criterion_was_violated(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion()

    result = engine.evaluate(
        criteria=(criterion,),
        events=(),
        task_roles=task_roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.INSUFFICIENT_EVIDENCE
    )
    assert result.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE


def test_wrong_event_type_does_not_match_criterion(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion(event_type="REGION_ENTRY")
    event = make_event(event_type="OBJECT_MOVING")

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.INSUFFICIENT_EVIDENCE
    )


def test_wrong_entity_role_does_not_match_criterion(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion()
    event = make_event(entity_ids=("another_cube",))

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.INSUFFICIENT_EVIDENCE
    )


def test_wrong_region_does_not_match_criterion(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion(region_role="target_region")
    event = make_event(region_id="another_region")

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.INSUFFICIENT_EVIDENCE
    )


def test_missing_required_evidence_type_does_not_satisfy_criterion(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion(
        required_evidence_types=("STABLE_IN_REGION",),
    )

    event = make_event(
        evidence=(
            make_evidence(evidence_type="OBJECT_MOTION"),
        ),
    )

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.INSUFFICIENT_EVIDENCE
    )


def test_partial_coverage_prevents_verified_verdict(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    partial_context: ObservationContext,
) -> None:
    criterion = make_criterion()
    event = make_event()

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=partial_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.SATISFIED
    )
    assert result.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE


def test_unknown_coverage_prevents_verified_verdict(
    engine: VerdictEngine,
    task_roles: TaskRoles,
) -> None:
    context = ObservationContext(
        fps=30.0,
        frame_count=10,
        frame_start=0,
        frame_end=9,
        coverage_scope=CoverageScope.UNKNOWN,
    )

    result = engine.evaluate(
        criteria=(make_criterion(),),
        events=(make_event(),),
        task_roles=task_roles,
        context=context,
    )

    assert result.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE


def test_complete_task_scope_without_full_video_coverage_is_insufficient(
    engine: VerdictEngine,
    task_roles: TaskRoles,
) -> None:
    context = ObservationContext(
        fps=30.0,
        frame_count=10,
        frame_start=2,
        frame_end=7,
        coverage_scope=CoverageScope.COMPLETE_TASK,
    )

    result = engine.evaluate(
        criteria=(make_criterion(),),
        events=(make_event(),),
        task_roles=task_roles,
        context=context,
    )

    assert result.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE


def test_forbidden_event_detected_in_partial_video_causes_failure(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    partial_context: ObservationContext,
) -> None:
    criterion = make_criterion(
        requirement=CriterionRequirement.FORBIDDEN,
    )

    result = engine.evaluate(
        criteria=(criterion,),
        events=(make_event(),),
        task_roles=task_roles,
        context=partial_context,
    )

    assert result.verdict == VerificationVerdict.FAILED


def test_optional_required_violation_does_not_fail_task(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion(
        is_mandatory=False,
    )

    event = make_observable_absence_event()

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.VIOLATED
    )
    assert result.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE


def test_optional_forbidden_violation_causes_failure(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion(
        requirement=CriterionRequirement.FORBIDDEN,
        is_mandatory=False,
    )

    result = engine.evaluate(
        criteria=(criterion,),
        events=(make_event(),),
        task_roles=task_roles,
        context=complete_context,
    )

    assert result.verdict == VerificationVerdict.FAILED


def test_all_mandatory_criteria_satisfied_with_complete_coverage(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criteria = (
        make_criterion(
            criterion_id="criterion_motion",
            event_type="OBJECT_MOVING",
        ),
        make_criterion(
            criterion_id="criterion_separation",
            event_type="RELATIVE_SEPARATION",
        ),
    )

    events = (
        make_event(
            event_id="event_motion",
            event_type="OBJECT_MOVING",
        ),
        make_event(
            event_id="event_separation",
            event_type="RELATIVE_SEPARATION",
        ),
    )

    result = engine.evaluate(
        criteria=criteria,
        events=events,
        task_roles=task_roles,
        context=complete_context,
    )

    assert result.verdict == VerificationVerdict.VERIFIED
    assert all(
        criterion_result.status == CriterionResultStatus.SATISFIED
        for criterion_result in result.criterion_results
    )


def test_missing_entity_role_mapping_is_insufficient(
    engine: VerdictEngine,
    complete_context: ObservationContext,
) -> None:
    roles = TaskRoles(
        entity_roles={},
        region_roles={},
    )

    result = engine.evaluate(
        criteria=(make_criterion(),),
        events=(make_event(),),
        task_roles=roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.INSUFFICIENT_EVIDENCE
    )


def test_missing_region_role_mapping_is_insufficient(
    engine: VerdictEngine,
    complete_context: ObservationContext,
) -> None:
    roles = TaskRoles(
        entity_roles={"target_object": "cube_01"},
        region_roles={},
    )

    criterion = make_criterion(region_role="target_region")
    event = make_event(region_id="target_zone_01")

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.INSUFFICIENT_EVIDENCE
    )


def test_not_detected_event_must_cover_complete_context_interval(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion()

    event = make_observable_absence_event(
        frame_start=1,
        frame_end=8,
    )

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.INSUFFICIENT_EVIDENCE
    )


def test_observability_evidence_must_cover_complete_context_interval(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion()

    event = make_event(
        status=EventStatus.NOT_DETECTED,
        evidence=(
            make_evidence(
                evidence_type="OBSERVABILITY_CONFIRMED",
                frame_start=1,
                frame_end=8,
            ),
        ),
    )

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.INSUFFICIENT_EVIDENCE
    )


def test_observability_proof_is_not_accepted_for_partial_coverage(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    partial_context: ObservationContext,
) -> None:
    criterion = make_criterion()

    event = make_observable_absence_event(
        frame_start=2,
        frame_end=7,
    )

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=partial_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.INSUFFICIENT_EVIDENCE
    )


def test_event_and_evidence_references_are_deduplicated(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion()

    shared_evidence = make_evidence(
        evidence_id="shared_evidence",
    )

    first_event = make_event(
        event_id="event_01",
        evidence=(shared_evidence,),
    )

    second_event = make_event(
        event_id="event_02",
        evidence=(shared_evidence,),
    )

    result = engine.evaluate(
        criteria=(criterion,),
        events=(first_event, second_event),
        task_roles=task_roles,
        context=complete_context,
    )

    criterion_result = result.criterion_results[0]

    assert criterion_result.event_ids == ("event_01", "event_02")
    assert criterion_result.evidence_ids == ("shared_evidence",)


def test_duplicate_criterion_ids_are_rejected(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion_a = make_criterion(criterion_id="duplicate")
    criterion_b = make_criterion(criterion_id="duplicate")

    with pytest.raises(ValueError, match="unique criterion IDs"):
        engine.evaluate(
            criteria=(criterion_a, criterion_b),
            events=(),
            task_roles=task_roles,
            context=complete_context,
        )


def test_duplicate_event_ids_are_rejected(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    event_a = make_event(event_id="duplicate")
    event_b = make_event(event_id="duplicate")

    with pytest.raises(ValueError, match="unique event IDs"):
        engine.evaluate(
            criteria=(make_criterion(),),
            events=(event_a, event_b),
            task_roles=task_roles,
            context=complete_context,
        )


def test_empty_criteria_are_rejected(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    with pytest.raises(ValueError, match="criteria must not be empty"):
        engine.evaluate(
            criteria=(),
            events=(),
            task_roles=task_roles,
            context=complete_context,
        )


@pytest.mark.parametrize(
    "invalid_criteria",
    [
        "not a sequence",
        None,
        123,
    ],
)
def test_invalid_criteria_input_is_rejected(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
    invalid_criteria: object,
) -> None:
    with pytest.raises(ValueError):
        engine.evaluate(
            criteria=invalid_criteria,
            events=(),
            task_roles=task_roles,
            context=complete_context,
        )


@pytest.mark.parametrize(
    "invalid_events",
    [
        "not a sequence",
        None,
        123,
    ],
)
def test_invalid_events_input_is_rejected(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
    invalid_events: object,
) -> None:
    with pytest.raises(ValueError):
        engine.evaluate(
            criteria=(make_criterion(),),
            events=invalid_events,
            task_roles=task_roles,
            context=complete_context,
        )


def test_result_preserves_coverage_metadata(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    result = engine.evaluate(
        criteria=(make_criterion(),),
        events=(make_event(),),
        task_roles=task_roles,
        context=complete_context,
    )

    assert result.coverage_scope == CoverageScope.COMPLETE_TASK
    assert result.covers_entire_video is True


def test_required_criterion_can_be_satisfied_without_region_constraint(
    engine: VerdictEngine,
    task_roles: TaskRoles,
    complete_context: ObservationContext,
) -> None:
    criterion = make_criterion(region_role=None)
    event = make_event(region_id=None)

    result = engine.evaluate(
        criteria=(criterion,),
        events=(event,),
        task_roles=task_roles,
        context=complete_context,
    )

    assert (
        result.criterion_results[0].status
        == CriterionResultStatus.SATISFIED
    )