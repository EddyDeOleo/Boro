
"""Run the Boro verification pipeline on episode 000025."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import analyze_dual_masks as analysis

from src.models.observation_context import (
    CoverageScope,
    ObservationContext,
)
from src.models.task_roles import TaskRoles
from src.models.verification_criteria import (
    CriterionRequirement,
    VerificationCriteria,
)
from src.verification.verification_pipeline import VerificationPipeline


def build_task_roles() -> TaskRoles:
    """Define semantic roles for the observed episode."""

    return TaskRoles(
        entity_roles={
            "object": "red_cube",
            "effector": "gripper",
        },
        region_roles={
            "target": "target_zone_01",
        },
    )


def build_verification_criteria() -> tuple[VerificationCriteria, ...]:
    """Define task and diagnostic criteria."""

    return (
        VerificationCriteria(
            criterion_id="coordinated_transport",
            description=(
                "The object moves in coordination with the robot effector."
            ),
            event_type="OBJECT_MOVING_WITH_EFFECTOR",
            requirement=CriterionRequirement.REQUIRED,
            entity_roles=("object", "effector"),
            required_evidence_types=(
                "COORDINATED_ENTITY_MOTION",
            ),
            is_mandatory=False,
        ),
        VerificationCriteria(
            criterion_id="relative_separation",
            description=(
                "The centroid distance between the object and effector "
                "increases over time."
            ),
            event_type="RELATIVE_SEPARATION",
            requirement=CriterionRequirement.REQUIRED,
            entity_roles=("object", "effector"),
            required_evidence_types=(
                "RELATIVE_CENTROID_SEPARATION",
            ),
            is_mandatory=False,
        ),
        VerificationCriteria(
            criterion_id="stable_target_placement",
            description=(
                "The object remains visually stable inside the target zone."
            ),
            event_type="STABLE_PLACEMENT",
            requirement=CriterionRequirement.REQUIRED,
            entity_roles=("object",),
            region_role="target",
            required_evidence_types=(
                "STABLE_IN_REGION",
            ),
            is_mandatory=True,
        ),
    )


def print_pipeline_result(result) -> None:
    """Print observations, detected events, and verification results."""

    print("\n========== BORO PIPELINE RESULT ==========")

    print("\nObservation counts:")
    print(f"  Entity observations: {len(result.entity_observations)}")
    print(f"  Spatial relationships: {len(result.spatial_relationships)}")
    print(f"  Region observations: {len(result.region_observations)}")

    print("\nContext:")
    print(f"  FPS: {result.context.fps}")
    print(f"  Frames: {result.context.frame_start}-{result.context.frame_end}")
    print(f"  Offset: {result.context.time_offset_seconds:.3f}s")
    print(f"  Coverage scope: {result.context.coverage_scope.value}")
    print(
        "  Covers complete task: "
        f"{result.context.coverage_scope == CoverageScope.COMPLETE_TASK}"
    )

    print(f"\nDetected events: {len(result.events)}")

    if not result.events:
        print("  No events detected.")

    for event in result.events:
        print(f"\n  Event: {event.event_type}")
        print(f"    Status: {event.status.value}")
        print(f"    Frames: {event.frame_start}-{event.frame_end}")
        print(
            f"    Time: {event.timestamp_start:.3f}s"
            f"-{event.timestamp_end:.3f}s"
        )
        print(f"    Entities: {event.entity_ids}")
        print(f"    Confidence: {event.confidence}")

        evidence_types = [
            evidence.evidence_type
            for evidence in event.evidence
        ]
        print(f"    Evidence: {evidence_types}")

        if event.attributes:
            print(f"    Attributes: {dict(event.attributes)}")

    verification = result.verification

    print("\nCriterion results:")

    for criterion_result in verification.criterion_results:
        print(f"\n  Criterion: {criterion_result.criterion_id}")
        print(f"    Requirement: {criterion_result.requirement.value}")
        print(f"    Mandatory: {criterion_result.is_mandatory}")
        print(f"    Status: {criterion_result.status.value}")
        print(f"    Explanation: {criterion_result.explanation}")
        print(f"    Event IDs: {criterion_result.event_ids}")
        print(f"    Evidence IDs: {criterion_result.evidence_ids}")

    print("\nFinal verdict:")
    print(f"  {verification.verdict.value}")
    print(f"  {verification.summary}")

    print("\n==========================================")


def main() -> None:
    """Load the episode, execute Boro, and print its result."""

    print("Loading SAM response...")

    raw_text = analysis.load_sam_response()
    masks = analysis.parse_masks(raw_text)
    grouped_masks = analysis.group_masks_by_frame(masks)

    height, width, fps = analysis.get_video_metadata()

    print("\nEpisode metadata:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.3f}")
    print(f"  Frames: {analysis.EXPECTED_FRAMES}")
    print(f"  SAM masks: {len(masks)}")

    print("\nLoading target region...")

    target_region = analysis.load_target_region()

    if (
        target_region.frame_width != width
        or target_region.frame_height != height
    ):
        raise RuntimeError(
            "Target region dimensions do not match the video."
        )

    print(f"  Region: {target_region.region_id}")

    print("\nAnalyzing SAM masks...")

    (
        _,
        _,
        _,
        entity_metric_records,
        relative_metric_records,
        relationship_metric_records,
        masks_by_frame,
        confidences,
    ) = analysis.analyze_episode(
        grouped_masks,
        (height, width),
    )

    context = ObservationContext(
        fps=fps,
        frame_count=analysis.EXPECTED_FRAMES,
        frame_start=0,
        frame_end=analysis.EXPECTED_FRAMES - 1,
        time_offset_seconds=analysis.VIDEO_TIME_OFFSET_SECONDS,
        coverage_scope=CoverageScope.PARTIAL_SEGMENT,
    )

    task_roles = build_task_roles()
    criteria = build_verification_criteria()

    print("\nRunning Boro verification pipeline...")

    pipeline = VerificationPipeline()

    result = pipeline.run(
        object_metrics=entity_metric_records,
        relative_metrics=relative_metric_records,
        relationship_metrics=relationship_metric_records,
        masks_by_frame=masks_by_frame,
        object_id_map=analysis.OBJECT_IDS,
        confidences=confidences,
        relative_entity_ids=(
            analysis.CUBE_ID,
            analysis.GRIPPER_ID,
        ),
        task_roles=task_roles,
        criteria=criteria,
        context=context,
        region=target_region,
        region_sam_object_id=analysis.CUBE_ID,
        source=analysis.SAM_SOURCE,
    )

    if len(result.entity_observations) != (
        analysis.EXPECTED_FRAMES * len(analysis.OBJECT_IDS)
    ):
        raise RuntimeError("Unexpected entity observation count.")

    if len(result.spatial_relationships) != analysis.EXPECTED_FRAMES:
        raise RuntimeError("Unexpected spatial relationship count.")

    if len(result.region_observations) != analysis.EXPECTED_FRAMES:
        raise RuntimeError("Unexpected region observation count.")

    print_pipeline_result(result)


if __name__ == "__main__":
    main()
