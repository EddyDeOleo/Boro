

"""Analyze dual SAM masks and adapt them into Boro observation contracts."""

from __future__ import annotations

import sys
import json
import csv

from pathlib import Path
from collections import defaultdict

# ============================================================
# PATH AND IMPORT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import meta_sam_parser as sam

from src.perception.mask_analyzer import (
    calculate_mask_metrics,
    calculate_relative_metrics,
    calculate_mask_relationship_metrics,
)

from src.adapters.sam_observation_adapter import (
    SAMObservationBundle,
    adapt_sam_observations,
)

from src.models.region import (
    RegionDefinition,
    RegionType,
)


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = ROOT / "videos" / "episode_000025_critical.mp4"

EVENTS_PATH = (
    ROOT / "evaluation" / "sam_episode_000025_dual_events.jsonl"
)

REGION_PATH = ROOT / "evaluation" / "target_region.json"

OUTPUT_DIR = ROOT / "evaluation" / "dual_tracking_analysis"

OBJECT_IDS = {
    "1": "red_cube",
    "2": "gripper",
}

CUBE_ID = "1"
GRIPPER_ID = "2"

EXPECTED_FRAMES = 51

SAM_SOURCE = "sam-3.1"

# Timestamps are relative to the beginning of the extracted clip
# when this value is 0.0.
#
# Set this to the clip's starting timestamp in the original video
# if Boro needs timestamps in the original video timeline.
VIDEO_TIME_OFFSET_SECONDS = 5.8

# Expected frame indexes are zero-based.
EXPECTED_FRAME_INDEXES = set(range(EXPECTED_FRAMES))


# ============================================================
# LOAD SAM RESPONSE
# ============================================================

def load_sam_response() -> str:
    """Recover the completed SAM response from the JSONL event stream."""

    if not EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {EVENTS_PATH}"
        )

    events = []

    with EVENTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in event stream at line {line_number}."
                ) from error

    completed = next(
        (
            event
            for event in reversed(events)
            if event.get("type") == "response.completed"
        ),
        None,
    )

    if completed is None:
        raise RuntimeError(
            "response.completed event was not found."
        )

    response = completed.get("response")

    if not isinstance(response, dict):
        raise RuntimeError(
            "The completed event does not contain a valid response."
        )

    parts = []

    for item in response.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")

            if isinstance(text, str):
                parts.append(text)

    raw_text = "".join(parts)

    if not raw_text.strip():
        raise RuntimeError(
            "The SAM response is empty."
        )

    return raw_text


# ============================================================
# PARSE MASKS
# ============================================================

def parse_masks(raw_text: str):
    """Parse segmentation records from the completed SAM response."""

    parser = sam.video_segmentation_format().create_parser()

    parser.push(
        raw_text
        if raw_text.endswith("\n")
        else raw_text + "\n"
    )

    result = parser.finish(
        sam.CompletedOutcome()
    ).result

    if result.diagnostics:
        for diagnostic in result.diagnostics:
            print(
                "DIAGNOSTIC:",
                diagnostic.severity,
                diagnostic.code,
                diagnostic.message,
            )

    masks = [
        record
        for record in result.records
        if isinstance(
            record,
            sam.SegmentationMaskRecord,
        )
    ]

    expected_count = EXPECTED_FRAMES * len(OBJECT_IDS)

    if len(masks) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} masks, "
            f"but found {len(masks)}."
        )

    return masks


# ============================================================
# RECONSTRUCT FULL-FRAME MASK
# ============================================================

def reconstruct_mask(record, frame_shape):
    """Reconstruct a SAM mask in full-frame image coordinates."""

    height, width = frame_shape[:2]

    bounds = record.bounds
    mask = record.mask

    left = int(bounds.left)
    top = int(bounds.top)
    right = int(bounds.right)
    bottom = int(bounds.bottom)

    bounds_width = right - left
    bounds_height = bottom - top

    if (
        mask.width != bounds_width
        or mask.height != bounds_height
    ):
        raise ValueError(
            f"Incompatible dimensions in frame "
            f"{record.identity.frame_index}, "
            f"object {record.object_id}."
        )

    if not (
        0 <= left < right <= width
        and 0 <= top < bottom <= height
    ):
        raise ValueError(
            f"Mask bounds are outside the frame: "
            f"{left}, {top}, {right}, {bottom}."
        )

    raster_bytes = sam.decode_mask_to_raster(mask)

    raster = np.frombuffer(
        raster_bytes,
        dtype=np.uint8,
    ).reshape(
        (mask.height, mask.width)
    )

    full_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    full_mask[top:bottom, left:right] = (
        raster > 0
    ).astype(np.uint8)

    return full_mask


# ============================================================
# GET VIDEO METADATA
# ============================================================

def get_video_metadata():
    """Return video height, width, and FPS after validating metadata."""

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid video dimensions: {width}x{height}."
        )

    if frame_count != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Expected {EXPECTED_FRAMES} frames, "
            f"but the video contains {frame_count}."
        )

    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(
            f"Invalid video FPS: {fps}."
        )

    return height, width, fps


# ============================================================
# LOAD TARGET REGION
# ============================================================

def load_target_region() -> RegionDefinition:
    """Load and validate the manually annotated target region."""

    if not REGION_PATH.exists():
        raise FileNotFoundError(
            f"Target region file not found: {REGION_PATH}"
        )

    with REGION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        region_data = json.load(file)

    try:
        region_data["region_type"] = RegionType(
            region_data["region_type"]
        )
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Invalid or missing region_type in target_region.json."
        ) from error

    return RegionDefinition(**region_data)


# ============================================================
# GROUP MASKS BY FRAME
# ============================================================

def group_masks_by_frame(masks):
    """Group SAM records and validate complete frame/object coverage."""

    grouped = defaultdict(dict)

    for record in masks:
        frame = record.identity.frame_index
        object_id = str(record.object_id)

        if not isinstance(frame, int):
            raise ValueError(
                f"Invalid frame index: {frame!r}."
            )

        if frame not in EXPECTED_FRAME_INDEXES:
            raise ValueError(
                f"Unexpected frame index: {frame}."
            )

        if object_id not in OBJECT_IDS:
            raise ValueError(
                f"Unexpected object ID: {object_id}."
            )

        if object_id in grouped[frame]:
            raise ValueError(
                f"Duplicate mask: frame={frame}, object={object_id}."
            )

        grouped[frame][object_id] = record

    actual_frames = set(grouped.keys())

    if actual_frames != EXPECTED_FRAME_INDEXES:
        missing = sorted(EXPECTED_FRAME_INDEXES - actual_frames)
        unexpected = sorted(actual_frames - EXPECTED_FRAME_INDEXES)

        raise RuntimeError(
            f"Invalid frame coverage. "
            f"Missing frames: {missing}. "
            f"Unexpected frames: {unexpected}."
        )

    expected_ids = set(OBJECT_IDS.keys())

    for frame in range(EXPECTED_FRAMES):
        actual_ids = set(grouped[frame].keys())

        if actual_ids != expected_ids:
            raise RuntimeError(
                f"Invalid object coverage in frame {frame}. "
                f"Expected {sorted(expected_ids)}, "
                f"found {sorted(actual_ids)}."
            )

    return grouped


# ============================================================
# ANALYZE ALL FRAMES
# ============================================================

def analyze_episode(grouped_masks, frame_shape):
    """Calculate mask, centroid, and spatial relationship metrics."""

    object_metrics_rows = []
    relative_metrics_rows = []
    relationship_metrics_rows = []

    entity_metric_records = []
    relative_metric_records = []
    relationship_metric_records = []

    masks_by_frame = {}
    confidences = {}

    previous_masks = {}

    for frame in range(EXPECTED_FRAMES):
        current_metrics = {}
        current_masks = {}

        for object_id, object_name in OBJECT_IDS.items():
            record = grouped_masks[frame][object_id]

            current_mask = reconstruct_mask(
                record,
                frame_shape,
            )

            previous_mask = previous_masks.get(object_id)

            metrics = calculate_mask_metrics(
                mask=current_mask,
                frame=frame,
                object_id=object_id,
                previous_mask=previous_mask,
            )

            current_metrics[object_id] = metrics
            current_masks[object_id] = current_mask

            entity_metric_records.append(metrics)

            confidences[(frame, object_id)] = record.confidence

            object_metrics_rows.append({
                "frame": frame,
                "object_id": object_id,
                "object_name": object_name,
                "area": metrics.area,
                "centroid_x": metrics.centroid_x,
                "centroid_y": metrics.centroid_y,
                "bbox_x_min": metrics.bbox[0],
                "bbox_y_min": metrics.bbox[1],
                "bbox_x_max": metrics.bbox[2],
                "bbox_y_max": metrics.bbox[3],
                "iou_previous": metrics.iou_previous,
                "area_ratio": metrics.area_ratio,
                "confidence": record.confidence,
            })

            previous_masks[object_id] = current_mask

        # Only the cube mask is required for target-region analysis.
        # The gripper mask is already represented in the relationship metrics.
        masks_by_frame[frame] = {
            CUBE_ID: current_masks[CUBE_ID].copy()
        }

        # ----------------------------------------------------
        # Centroid-based relative metrics.
        # ----------------------------------------------------

        relative = calculate_relative_metrics(
            current_metrics[CUBE_ID],
            current_metrics[GRIPPER_ID],
        )

        relative_metric_records.append(relative)

        relative_metrics_rows.append({
            "frame": frame,
            "distance": relative.distance,
            "delta_x": relative.delta_x,
            "delta_y": relative.delta_y,
        })

        # ----------------------------------------------------
        # Mask-based geometric relationship metrics.
        # ----------------------------------------------------

        relationship = calculate_mask_relationship_metrics(
            mask_a=current_masks[CUBE_ID],
            mask_b=current_masks[GRIPPER_ID],
            frame=frame,
            object_a_id=CUBE_ID,
            object_b_id=GRIPPER_ID,
        )

        relationship_metric_records.append(relationship)

        relationship_metrics_rows.append({
            "frame": frame,
            "object_a": OBJECT_IDS[CUBE_ID],
            "object_b": OBJECT_IDS[GRIPPER_ID],
            "min_distance": relationship.min_distance,
            "intersection_area": relationship.intersection_area,
            "iou": relationship.iou,
            "overlap_a": relationship.overlap_a,
            "overlap_b": relationship.overlap_b,
            "normalized_delta_x": relationship.normalized_delta_x,
            "normalized_delta_y": relationship.normalized_delta_y,
        })

    return (
        object_metrics_rows,
        relative_metrics_rows,
        relationship_metrics_rows,
        entity_metric_records,
        relative_metric_records,
        relationship_metric_records,
        masks_by_frame,
        confidences,
    )


# ============================================================
# BUILD BORO OBSERVATION CONTRACTS
# ============================================================

def build_observation_bundle(
    entity_metric_records,
    relative_metric_records,
    relationship_metric_records,
    masks_by_frame,
    confidences,
    fps,
    target_region,
):
    """Adapt the analyzed SAM episode into Boro observation contracts."""

    bundle = adapt_sam_observations(
        object_metrics=entity_metric_records,
        relative_metrics=relative_metric_records,
        relationship_metrics=relationship_metric_records,
        masks_by_frame=masks_by_frame,
        object_id_map=OBJECT_IDS,
        confidences=confidences,
        fps=fps,
        relative_entity_ids=(CUBE_ID, GRIPPER_ID),
        region=target_region,
        region_sam_object_id=CUBE_ID,
        time_offset_seconds=VIDEO_TIME_OFFSET_SECONDS,
        source=SAM_SOURCE,
    )

    validate_observation_bundle(
        bundle=bundle,
        fps=fps,
    )

    return bundle


# ============================================================
# VALIDATE BORO OBSERVATION CONTRACTS
# ============================================================

def validate_observation_bundle(
    bundle: SAMObservationBundle,
    fps: float,
):
    """Validate counts, frame coverage, identities, and timestamps."""

    expected_entity_count = EXPECTED_FRAMES * len(OBJECT_IDS)

    if len(bundle.entity_observations) != expected_entity_count:
        raise RuntimeError(
            "Unexpected entity observation count: "
            f"{len(bundle.entity_observations)} "
            f"(expected {expected_entity_count})."
        )

    if len(bundle.spatial_relationships) != EXPECTED_FRAMES:
        raise RuntimeError(
            "Unexpected spatial relationship count: "
            f"{len(bundle.spatial_relationships)} "
            f"(expected {EXPECTED_FRAMES})."
        )

    if len(bundle.region_observations) != EXPECTED_FRAMES:
        raise RuntimeError(
            "Unexpected region observation count: "
            f"{len(bundle.region_observations)} "
            f"(expected {EXPECTED_FRAMES})."
        )

    # --------------------------------------------------------
    # Entity observations.
    # --------------------------------------------------------

    seen_entity_keys = set()

    for observation in bundle.entity_observations:
        key = (
            observation.frame_index,
            observation.entity_id,
        )

        if key in seen_entity_keys:
            raise RuntimeError(
                f"Duplicate entity observation: {key}."
            )

        seen_entity_keys.add(key)

        if observation.frame_index not in EXPECTED_FRAME_INDEXES:
            raise RuntimeError(
                f"Unexpected entity frame: {observation.frame_index}."
            )

        if observation.entity_id not in OBJECT_IDS.values():
            raise RuntimeError(
                f"Unexpected semantic entity ID: {observation.entity_id}."
            )

        expected_timestamp = (
            observation.frame_index / fps
            + VIDEO_TIME_OFFSET_SECONDS
        )

        if not np.isclose(
            observation.timestamp_seconds,
            expected_timestamp,
            atol=1e-9,
        ):
            raise RuntimeError(
                f"Invalid timestamp for observation "
                f"{observation.observation_id}."
            )

        if observation.source != SAM_SOURCE:
            raise RuntimeError(
                f"Unexpected observation source: {observation.source}."
            )

    expected_entity_keys = {
        (frame, entity_id)
        for frame in EXPECTED_FRAME_INDEXES
        for entity_id in OBJECT_IDS.values()
    }

    if seen_entity_keys != expected_entity_keys:
        raise RuntimeError(
            "Entity observation coverage is incomplete."
        )

    # --------------------------------------------------------
    # Spatial relationships.
    # --------------------------------------------------------

    relationship_frames = set()

    for relationship in bundle.spatial_relationships:
        if relationship.frame_index in relationship_frames:
            raise RuntimeError(
                f"Duplicate spatial relationship in frame "
                f"{relationship.frame_index}."
            )

        relationship_frames.add(relationship.frame_index)

        if relationship.entity_a_id != OBJECT_IDS[CUBE_ID]:
            raise RuntimeError(
                "Unexpected entity_a_id in spatial relationship."
            )

        if relationship.entity_b_id != OBJECT_IDS[GRIPPER_ID]:
            raise RuntimeError(
                "Unexpected entity_b_id in spatial relationship."
            )

        if relationship.attributes.get("physical_contact_established") is not False:
            raise RuntimeError(
                "Spatial relationship must not claim physical contact."
            )

        if relationship.attributes.get("source") != SAM_SOURCE:
            raise RuntimeError(
                "Unexpected spatial relationship source."
            )

    if relationship_frames != EXPECTED_FRAME_INDEXES:
        raise RuntimeError(
            "Spatial relationship frame coverage is incomplete."
        )

    # --------------------------------------------------------
    # Region observations.
    # --------------------------------------------------------

    region_frames = set()

    for observation in bundle.region_observations:
        if observation.frame_index in region_frames:
            raise RuntimeError(
                f"Duplicate region observation in frame "
                f"{observation.frame_index}."
            )

        region_frames.add(observation.frame_index)

        if observation.entity_id != OBJECT_IDS[CUBE_ID]:
            raise RuntimeError(
                "Unexpected entity in target-region observation."
            )

    if region_frames != EXPECTED_FRAME_INDEXES:
        raise RuntimeError(
            "Region observation frame coverage is incomplete."
        )


# ============================================================
# PRINT OBSERVATION SUMMARY
# ============================================================

def print_observation_summary(bundle: SAMObservationBundle):
    """Print a summary of the generated Boro contracts."""

    print("\n========== BORO OBSERVATION CONTRACTS ==========")

    print(
        f"Entity observations: "
        f"{len(bundle.entity_observations)}"
    )

    print(
        f"Spatial relationships: "
        f"{len(bundle.spatial_relationships)}"
    )

    print(
        f"Region observations: "
        f"{len(bundle.region_observations)}"
    )

    entity_counts = defaultdict(int)

    for observation in bundle.entity_observations:
        entity_counts[observation.entity_id] += 1

    print("\nEntity observations by entity:")

    for entity_id, count in sorted(entity_counts.items()):
        print(f"  {entity_id}: {count}")

    region_counts = defaultdict(int)

    for observation in bundle.region_observations:
        region_counts[observation.relation.value] += 1

    print("\nTarget region relations:")

    for relation, count in sorted(region_counts.items()):
        print(f"  {relation}: {count}")

    print("\n=================================================")


# ============================================================
# EXPORT CSV
# ============================================================

def export_csv(filename, rows):
    """Export analysis rows to a CSV file."""

    if not rows:
        raise ValueError(
            f"No data available for export: {filename}"
        )

    output_path = OUTPUT_DIR / filename

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV generated: {output_path}")


# ============================================================
# AREA PLOT
# ============================================================

def generate_area_plot(object_metrics):
    frames = list(range(EXPECTED_FRAMES))

    plt.figure(figsize=(12, 5))

    for object_id, object_name in OBJECT_IDS.items():
        rows = [
            row
            for row in object_metrics
            if row["object_id"] == object_id
        ]

        areas = [row["area"] for row in rows]

        plt.plot(
            frames,
            areas,
            marker=".",
            label=object_name,
        )

    plt.title("Segmented area per frame")
    plt.xlabel("Frame")
    plt.ylabel("Area (pixels)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "mask_area.png"

    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Plot generated: {path}")


# ============================================================
# CENTROID DISTANCE PLOT
# ============================================================

def generate_distance_plot(relative_metrics):
    frames = [row["frame"] for row in relative_metrics]

    distances = [
        row["distance"]
        for row in relative_metrics
    ]

    plt.figure(figsize=(12, 5))

    plt.plot(
        frames,
        distances,
        marker=".",
        label="Cube-gripper centroid distance",
    )

    plt.title("Centroid distance between cube and gripper")
    plt.xlabel("Frame")
    plt.ylabel("Distance (pixels)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "relative_distance.png"

    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Plot generated: {path}")


# ============================================================
# RELATIVE POSITION PLOT
# ============================================================

def generate_relative_position_plot(relative_metrics):
    frames = [row["frame"] for row in relative_metrics]

    delta_x = [
        row["delta_x"]
        for row in relative_metrics
    ]

    delta_y = [
        row["delta_y"]
        for row in relative_metrics
    ]

    plt.figure(figsize=(12, 5))

    plt.plot(
        frames,
        delta_x,
        label="Delta X",
    )

    plt.plot(
        frames,
        delta_y,
        label="Delta Y",
    )

    plt.title("Cube position relative to gripper")
    plt.xlabel("Frame")
    plt.ylabel("Relative position (pixels)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "relative_position.png"

    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Plot generated: {path}")


# ============================================================
# MINIMUM MASK DISTANCE PLOT
# ============================================================

def generate_min_distance_plot(relationship_metrics):
    frames = [
        row["frame"]
        for row in relationship_metrics
    ]

    distances = [
        row["min_distance"]
        for row in relationship_metrics
    ]

    plt.figure(figsize=(12, 5))

    plt.plot(
        frames,
        distances,
        marker=".",
        label="Minimum mask distance",
    )

    plt.title("Minimum distance between cube and gripper masks")
    plt.xlabel("Frame")
    plt.ylabel("Minimum distance (pixels)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "mask_min_distance.png"

    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Plot generated: {path}")


# ============================================================
# MASK OVERLAP PLOT
# ============================================================

def generate_overlap_plot(relationship_metrics):
    frames = [
        row["frame"]
        for row in relationship_metrics
    ]

    iou_values = [
        row["iou"]
        for row in relationship_metrics
    ]

    overlap_a_values = [
        row["overlap_a"]
        for row in relationship_metrics
    ]

    overlap_b_values = [
        row["overlap_b"]
        for row in relationship_metrics
    ]

    plt.figure(figsize=(12, 5))

    plt.plot(
        frames,
        iou_values,
        marker=".",
        label="IoU",
    )

    plt.plot(
        frames,
        overlap_a_values,
        marker=".",
        label="Cube overlap",
    )

    plt.plot(
        frames,
        overlap_b_values,
        marker=".",
        label="Gripper overlap",
    )

    plt.title("Geometric overlap between cube and gripper")
    plt.xlabel("Frame")
    plt.ylabel("Overlap ratio")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "mask_overlap.png"

    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Plot generated: {path}")


# ============================================================
# NORMALIZED RELATIVE POSITION PLOT
# ============================================================

def generate_normalized_position_plot(relationship_metrics):
    frames = [
        row["frame"]
        for row in relationship_metrics
    ]

    delta_x = [
        row["normalized_delta_x"]
        for row in relationship_metrics
    ]

    delta_y = [
        row["normalized_delta_y"]
        for row in relationship_metrics
    ]

    plt.figure(figsize=(12, 5))

    plt.plot(
        frames,
        delta_x,
        marker=".",
        label="Normalized Delta X",
    )

    plt.plot(
        frames,
        delta_y,
        marker=".",
        label="Normalized Delta Y",
    )

    plt.title("Normalized cube position relative to gripper")
    plt.xlabel("Frame")
    plt.ylabel("Relative position (normalized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "mask_normalized_position.png"

    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Plot generated: {path}")


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    object_metrics,
    relative_metrics,
    relationship_metrics,
):
    """Print geometric and temporal mask analysis statistics."""

    print("\n========== EPISODE ANALYSIS ==========")

    for object_id, object_name in OBJECT_IDS.items():
        rows = [
            row
            for row in object_metrics
            if row["object_id"] == object_id
        ]

        areas = [
            row["area"]
            for row in rows
        ]

        print(f"\n{object_name}")
        print(f"Frames analyzed: {len(rows)}")
        print(f"Minimum area: {min(areas)}")
        print(f"Maximum area: {max(areas)}")
        print(f"Average area: {np.mean(areas):.2f}")

        iou_values = [
            row["iou_previous"]
            for row in rows
            if row["iou_previous"] is not None
        ]

        if iou_values:
            print(
                f"Average temporal IoU: "
                f"{np.mean(iou_values):.4f}"
            )

    centroid_distances = [
        row["distance"]
        for row in relative_metrics
    ]

    print("\nCUBE - GRIPPER CENTROID RELATIONSHIP")
    print(
        f"Initial distance: "
        f"{centroid_distances[0]:.2f} px"
    )
    print(
        f"Final distance: "
        f"{centroid_distances[-1]:.2f} px"
    )
    print(
        f"Minimum distance: "
        f"{min(centroid_distances):.2f} px"
    )
    print(
        f"Maximum distance: "
        f"{max(centroid_distances):.2f} px"
    )

    mask_distances = [
        row["min_distance"]
        for row in relationship_metrics
    ]

    iou_values = [
        row["iou"]
        for row in relationship_metrics
    ]

    intersection_values = [
        row["intersection_area"]
        for row in relationship_metrics
    ]

    overlap_a_values = [
        row["overlap_a"]
        for row in relationship_metrics
    ]

    overlap_b_values = [
        row["overlap_b"]
        for row in relationship_metrics
    ]

    print("\nCUBE - GRIPPER MASK RELATIONSHIP")
    print(
        f"Initial minimum mask distance: "
        f"{mask_distances[0]:.2f} px"
    )
    print(
        f"Final minimum mask distance: "
        f"{mask_distances[-1]:.2f} px"
    )
    print(
        f"Minimum mask distance across episode: "
        f"{min(mask_distances):.2f} px"
    )
    print(
        f"Maximum mask distance across episode: "
        f"{max(mask_distances):.2f} px"
    )
    print(
        f"Frames with mask intersection: "
        f"{sum(value > 0 for value in intersection_values)}"
        f"/{EXPECTED_FRAMES}"
    )
    print(
        f"Maximum intersection area: "
        f"{max(intersection_values)} pixels"
    )
    print(
        f"Maximum IoU: {max(iou_values):.4f}"
    )
    print(
        f"Maximum cube overlap: "
        f"{max(overlap_a_values):.4f}"
    )
    print(
        f"Maximum gripper overlap: "
        f"{max(overlap_b_values):.4f}"
    )

    print("\n======================================")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading SAM response...")

    raw_text = load_sam_response()

    print("Processing masks...")

    masks = parse_masks(raw_text)

    grouped_masks = group_masks_by_frame(masks)

    height, width, fps = get_video_metadata()

    print(f"Masks: {len(masks)}")
    print(f"Frames: {EXPECTED_FRAMES}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.3f}")
    print(
        f"Video time offset: "
        f"{VIDEO_TIME_OFFSET_SECONDS:.3f} seconds"
    )

    print("\nLoading target region...")

    target_region = load_target_region()

    if (
        target_region.frame_width != width
        or target_region.frame_height != height
    ):
        raise RuntimeError(
            "Target region dimensions do not match the video: "
            f"region={target_region.frame_width}x"
            f"{target_region.frame_height}, "
            f"video={width}x{height}."
        )

    print(
        f"Region loaded: {target_region.region_id}"
    )

    print("\nAnalyzing episode...")

    (
        object_metrics,
        relative_metrics,
        relationship_metrics,
        entity_metric_records,
        relative_metric_records,
        relationship_metric_records,
        masks_by_frame,
        confidences,
    ) = analyze_episode(
        grouped_masks,
        (height, width),
    )

    # --------------------------------------------------------
    # Build Boro observation contracts.
    # --------------------------------------------------------

    print("\nBuilding Boro observation contracts...")

    observation_bundle = build_observation_bundle(
        entity_metric_records=entity_metric_records,
        relative_metric_records=relative_metric_records,
        relationship_metric_records=relationship_metric_records,
        masks_by_frame=masks_by_frame,
        confidences=confidences,
        fps=fps,
        target_region=target_region,
    )

    print_observation_summary(observation_bundle)

    # --------------------------------------------------------
    # Export analysis CSV files.
    # --------------------------------------------------------

    export_csv(
        "mask_metrics.csv",
        object_metrics,
    )

    export_csv(
        "relative_metrics.csv",
        relative_metrics,
    )

    export_csv(
        "mask_relationship_metrics.csv",
        relationship_metrics,
    )

    # --------------------------------------------------------
    # Generate plots.
    # --------------------------------------------------------

    generate_area_plot(object_metrics)
    generate_distance_plot(relative_metrics)
    generate_relative_position_plot(relative_metrics)

    generate_min_distance_plot(relationship_metrics)
    generate_overlap_plot(relationship_metrics)
    generate_normalized_position_plot(relationship_metrics)

    # --------------------------------------------------------
    # Print analysis summary.
    # --------------------------------------------------------

    print_summary(
        object_metrics,
        relative_metrics,
        relationship_metrics,
    )

    print(
        "\nAnalysis and observation adaptation completed."
    )


if __name__ == "__main__":
    main()
