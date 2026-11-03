
"""Run the complete Boro architecture using cached SAM 3.1 results."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts.analysis import analyze_dual_masks as analysis

from src.application.services.verification_service import (
    VerificationService,
)
from src.application.services.video_preparation_service import (
    VideoPreparationService,
)
from src.models.observation_context import CoverageScope
from src.models.perception_config import PerceptionConfig
from src.models.region import RegionDefinition
from src.models.task_roles import TaskRoles
from src.models.verification_criteria import (
    CriterionRequirement,
    VerificationCriteria,
)
from src.perception.cached_sam31_inference import (
    CachedSAM31Inference,
)
from src.perception.sam31_perception_provider import (
    Sam31PerceptionProvider,
)


VIDEO_PATH = analysis.VIDEO_PATH

SAM_RESPONSE_PATH = (
    ROOT
    / "evaluation"
    / "sam_episode_000025_dual_response.json"
)

REPORT_PATH = (
    ROOT
    / "evaluation"
    / "boro_episode_000025_report.json"
)

EXPECTED_ENTITY_OBSERVATIONS = (
    analysis.EXPECTED_FRAMES * len(analysis.OBJECT_IDS)
)

EXPECTED_SPATIAL_RELATIONSHIPS = analysis.EXPECTED_FRAMES


def build_perception_config() -> PerceptionConfig:
    """Configure the SAM provider and semantic entity mapping."""

    return PerceptionConfig(
        provider="meta",
        model=analysis.SAM_SOURCE,
        entity_id_map=analysis.OBJECT_IDS,
        include_spatial_relationships=True,
        include_region_observations=True,
    )


def build_task_roles() -> TaskRoles:
    """Define semantic roles for the observed robotic task."""

    return TaskRoles(
        entity_roles={
            "object": "red_cube",
            "effector": "gripper",
        },
        region_roles={
            "target_region": "target_zone_01",
        },
    )


def build_verification_criteria() -> tuple[VerificationCriteria, ...]:
    """Define the criteria used to evaluate the execution."""

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
            region_role="target_region",
            required_evidence_types=(
                "STABLE_IN_REGION",
            ),
            is_mandatory=True,
        ),
    )


def load_target_region() -> RegionDefinition:
    """Load the manually annotated target region."""

    region = analysis.load_target_region()

    if not isinstance(region, RegionDefinition):
        raise TypeError(
            "The target region loader did not return a RegionDefinition."
        )

    return region


def validate_paths() -> None:
    """Validate all required input files."""

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(
            f"Episode video does not exist: {VIDEO_PATH}"
        )

    if not SAM_RESPONSE_PATH.is_file():
        raise FileNotFoundError(
            f"Cached SAM response does not exist: {SAM_RESPONSE_PATH}"
        )


def validate_perception_result(
    perception_result: Any,
) -> None:
    """Validate the expected perception output for this episode."""

    actual_entity_count = len(perception_result.observations)

    if actual_entity_count != EXPECTED_ENTITY_OBSERVATIONS:
        raise RuntimeError(
            "Unexpected entity observation count: "
            f"expected {EXPECTED_ENTITY_OBSERVATIONS}, "
            f"received {actual_entity_count}."
        )

    actual_relationship_count = len(
        perception_result.spatial_relationships
    )

    if actual_relationship_count != EXPECTED_SPATIAL_RELATIONSHIPS:
        raise RuntimeError(
            "Unexpected spatial relationship count: "
            f"expected {EXPECTED_SPATIAL_RELATIONSHIPS}, "
            f"received {actual_relationship_count}."
        )

    if not perception_result.region_observations:
        raise RuntimeError(
            "No target-region observations were generated."
        )


def save_report(report: dict[str, Any]) -> None:
    """Serialize the verification report to the evaluation directory."""

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialized_report = json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )

    REPORT_PATH.write_text(
        serialized_report,
        encoding="utf-8",
    )


def print_execution_summary(
    perception_result: Any,
    report: dict[str, Any],
) -> None:
    """Print a concise summary of the complete execution."""

    report_payload = report.get("report", {})
    overview = report_payload.get("overview", {})

    print("\n========== BORO INTEGRATION RESULT ==========")

    print("\nPerception:")
    print(f"  Provider: {perception_result.provider}")
    print(f"  Model: {perception_result.model}")
    print(f"  Entity observations: {len(perception_result.observations)}")
    print(
        "  Spatial relationships: "
        f"{len(perception_result.spatial_relationships)}"
    )
    print(
        "  Region observations: "
        f"{len(perception_result.region_observations)}"
    )

    context = perception_result.context

    print("\nObservation context:")
    print(f"  FPS: {context.fps}")
    print(f"  Frame count: {context.frame_count}")
    print(f"  Frame range: {context.frame_start}-{context.frame_end}")
    print(f"  Offset: {context.time_offset_seconds:.3f}s")
    print(f"  Coverage: {context.coverage_scope.value}")
    print(f"  Start time: {context.start_time_seconds:.3f}s")
    print(f"  End time: {context.end_time_seconds:.3f}s")

    print("\nVerification:")
    print(f"  Verdict: {overview.get('verdict', 'See report JSON')}")
    print(f"  Summary: {overview.get('summary', 'See report JSON')}")

    print("\nReport:")
    print(f"  {REPORT_PATH}")

    print("\n=============================================")


def main() -> None:
    """Execute the complete local Boro verification workflow."""

    print("Validating input files...")
    validate_paths()

    print("\nLoading target region...")
    target_region = load_target_region()

    print(f"  Region: {target_region.region_id}")
    print(
        f"  Dimensions: "
        f"{target_region.frame_width}x{target_region.frame_height}"
    )

    print("\nInitializing cached SAM inference...")
    inference = CachedSAM31Inference(
        response_path=SAM_RESPONSE_PATH,
    )

    perception_provider = Sam31PerceptionProvider(
        inference=inference,
    )

    perception_config = build_perception_config()
    task_roles = build_task_roles()
    criteria = build_verification_criteria()

    print("\nPreparing episode video...")

    preparation_service = VideoPreparationService()

    with preparation_service.prepare(
        video_path=VIDEO_PATH,
        coverage_scope=CoverageScope.PARTIAL_SEGMENT,
        frame_start=0,
        frame_end=None,
        time_offset_seconds=analysis.VIDEO_TIME_OFFSET_SECONDS,
    ) as prepared_video:

        print("\nEpisode metadata:")
        print(f"  Video: {prepared_video.source_path}")
        print(
            f"  Resolution: {prepared_video.metadata.width}x"
            f"{prepared_video.metadata.height}"
        )
        print(f"  FPS: {prepared_video.metadata.fps:.3f}")
        print(f"  Frames: {prepared_video.metadata.frame_count}")

        if prepared_video.metadata.frame_count != analysis.EXPECTED_FRAMES:
            raise RuntimeError(
                "Video frame count does not match the cached SAM episode."
            )

        print("\nRunning SAM perception provider...")
        print("  Using cached response. No external API calls will be made.")

        perception_result = perception_provider.perceive(
            video_path=prepared_video.source_path,
            context=prepared_video.context,
            config=perception_config,
            regions=(target_region,),
        )

        print("\nValidating perception output...")
        validate_perception_result(perception_result)

        print("\n========== TARGET REGION DIAGNOSTICS ==========")

        object_region_observations = [
            observation
            for observation in perception_result.region_observations
            if observation.entity_id == "red_cube"
            and observation.region_id == "target_zone_01"
        ]

        relation_counts = {}

        for observation in object_region_observations:
            relation = observation.relation.value
            relation_counts[relation] = relation_counts.get(relation, 0) + 1

        print("\nRegion relation counts:")

        for relation, count in sorted(relation_counts.items()):
            print(f"  {relation}: {count}")

        qualifying_frames = [
            observation.frame_index
            for observation in object_region_observations
            if observation.overlap_ratio >= 0.80
        ]

        print("\nTarget overlap:")
        print(f"  Observations: {len(object_region_observations)}")
        print(f"  Frames with overlap >= 0.80: {len(qualifying_frames)}")
        print(f"  Qualifying frames: {qualifying_frames}")

        if object_region_observations:
            overlap_values = [
                observation.overlap_ratio
                for observation in object_region_observations
            ]

            print(f"  Maximum overlap: {max(overlap_values):.4f}")
            print(f"  Minimum overlap: {min(overlap_values):.4f}")

        print("\n===============================================")

        print("\nRunning verification service...")

        verification_service = VerificationService()

        report = verification_service.execute(
            perception_result=perception_result,
            task_roles=task_roles,
            criteria=criteria,
        )

    print("\nSaving serialized execution report...")
    save_report(report)

    print_execution_summary(
        perception_result=perception_result,
        report=report,
    )


if __name__ == "__main__":
    main()
