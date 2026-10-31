
import json
from pathlib import Path

import cv2
import numpy as np
import meta_sam_parser as sam


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = ROOT / "videos" / "episode_000025_critical.mp4"

EVENTS_PATH = (
    ROOT / "evaluation" / "sam_episode_000025_dual_events.jsonl"
)

OUTPUT_DIR = ROOT / "evaluation" / "dual_mask_visualization"

FRAMES_TO_VISUALIZE = [0, 20, 32, 40, 50]

# Colors in BGR format (OpenCV).
OBJECT_COLORS = {
    "1": (0, 0, 255),      # Cube: red
    "2": (0, 255, 0),      # Gripper: green
}

OBJECT_NAMES = {
    "1": "RED CUBE",
    "2": "GRIPPER",
}

ALPHA = 0.45

DETAIL_HALF_SIZE = 35
DETAIL_SCALE = 6


# ============================================================
# LOAD SAM RESPONSE
# ============================================================

def load_sam_response():
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"SAM events file not found: {EVENTS_PATH}"
        )

    events = [
        json.loads(line)
        for line in EVENTS_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

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

    parts = []

    for item in completed["response"].get("output", []):
        for content in item.get("content", []):
            text = content.get("text")

            if isinstance(text, str):
                parts.append(text)

    raw_text = "".join(parts)

    if not raw_text:
        raise RuntimeError(
            "The final SAM response is empty."
        )

    return raw_text


# ============================================================
# PARSE MASKS
# ============================================================

def parse_masks(raw_text):
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

    if len(masks) != 102:
        raise RuntimeError(
            f"Expected 102 masks, "
            f"but found {len(masks)}."
        )

    return masks


# ============================================================
# INDEX MASKS
# ============================================================

def index_masks(masks):
    indexed = {}

    for record in masks:
        frame_number = record.identity.frame_index
        object_id = record.object_id

        key = (frame_number, object_id)

        if key in indexed:
            raise RuntimeError(
                f"Duplicate mask: frame={frame_number}, "
                f"object_id={object_id}"
            )

        indexed[key] = record

    return indexed


# ============================================================
# RECONSTRUCT MASK
# ============================================================

def reconstruct_mask(record, frame_shape):
    frame_height, frame_width = frame_shape[:2]

    bounds = record.bounds
    mask = record.mask

    left = int(bounds.left)
    top = int(bounds.top)
    right = int(bounds.right)
    bottom = int(bounds.bottom)

    expected_width = right - left
    expected_height = bottom - top

    # Do not resize masks.
    # Stop processing if the dimensions do not match.
    if (
        mask.width != expected_width
        or mask.height != expected_height
    ):
        raise ValueError(
            "Mask dimensions do not match its bounds: "
            f"{mask.width}x{mask.height} "
            f"vs {expected_width}x{expected_height}"
        )

    if not (
        0 <= left < right <= frame_width
        and 0 <= top < bottom <= frame_height
    ):
        raise ValueError(
            f"Mask bounds are outside the frame: "
            f"({left}, {top}, {right}, {bottom})"
        )

    raster_bytes = sam.decode_mask_to_raster(mask)

    raster = np.frombuffer(
        raster_bytes,
        dtype=np.uint8,
    ).reshape(
        (mask.height, mask.width)
    )

    # Create a full-frame mask.
    full_mask = np.zeros(
        (frame_height, frame_width),
        dtype=np.uint8,
    )

    # Non-zero pixels represent the segmented region.
    full_mask[top:bottom, left:right] = (
        raster > 0
    ).astype(np.uint8)

    return full_mask


# ============================================================
# CLOSEST MASK PIXELS
# ============================================================

def find_closest_mask_pixels(mask_a, mask_b):
    """Find the closest pair of pixels between two binary masks."""

    y_a, x_a = np.where(mask_a > 0)
    y_b, x_b = np.where(mask_b > 0)

    if len(x_a) == 0 or len(x_b) == 0:
        raise ValueError(
            "Cannot compare empty masks."
        )

    points_a = np.column_stack(
        (x_a, y_a)
    ).astype(np.float64)

    points_b = np.column_stack(
        (x_b, y_b)
    ).astype(np.float64)

    min_distance_squared = float("inf")
    closest_pair = None

    chunk_size = 256

    for start in range(0, len(points_a), chunk_size):
        chunk = points_a[start:start + chunk_size]

        dx = (
            chunk[:, None, 0]
            - points_b[None, :, 0]
        )

        dy = (
            chunk[:, None, 1]
            - points_b[None, :, 1]
        )

        distances_squared = dx * dx + dy * dy

        flat_index = int(
            np.argmin(distances_squared)
        )

        row_index, column_index = np.unravel_index(
            flat_index,
            distances_squared.shape,
        )

        block_min = float(
            distances_squared[row_index, column_index]
        )

        if block_min < min_distance_squared:
            min_distance_squared = block_min

            closest_pair = (
                tuple(
                    chunk[row_index].astype(int)
                ),
                tuple(
                    points_b[column_index].astype(int)
                ),
            )

        if min_distance_squared == 0:
            break

    return (
        float(np.sqrt(min_distance_squared)),
        closest_pair,
    )


# ============================================================
# DRAW MASK CONTOURS
# ============================================================

def draw_mask_contour(image, mask, color):
    """Draw the actual segmentation boundary."""

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        image,
        contours,
        -1,
        color,
        1,
    )


# ============================================================
# CREATE MAGNIFIED CONTACT DETAIL
# ============================================================

def create_contact_detail(
    frame,
    mask_a,
    mask_b,
    distance,
    closest_pair,
    frame_number,
):
    """Create a magnified view of the closest mask pixels."""

    point_a, point_b = closest_pair

    center_x = int(
        (point_a[0] + point_b[0]) / 2
    )

    center_y = int(
        (point_a[1] + point_b[1]) / 2
    )

    height, width = frame.shape[:2]

    left = max(
        0,
        center_x - DETAIL_HALF_SIZE,
    )

    right = min(
        width,
        center_x + DETAIL_HALF_SIZE + 1,
    )

    top = max(
        0,
        center_y - DETAIL_HALF_SIZE,
    )

    bottom = min(
        height,
        center_y + DETAIL_HALF_SIZE + 1,
    )

    detail = frame[top:bottom, left:right].copy()

    local_a = mask_a[top:bottom, left:right]
    local_b = mask_b[top:bottom, left:right]

    # Overlay the actual segmentation pixels.
    overlay = detail.copy()

    overlay[local_a > 0] = OBJECT_COLORS["1"]
    overlay[local_b > 0] = OBJECT_COLORS["2"]

    detail = cv2.addWeighted(
        overlay,
        ALPHA,
        detail,
        1 - ALPHA,
        0,
    )

    # Convert global coordinates to crop coordinates.
    local_point_a = (
        point_a[0] - left,
        point_a[1] - top,
    )

    local_point_b = (
        point_b[0] - left,
        point_b[1] - top,
    )

    # Mark the closest pixel from each mask.
    cv2.circle(
        detail,
        local_point_a,
        2,
        (255, 255, 255),
        -1,
    )

    cv2.circle(
        detail,
        local_point_b,
        2,
        (0, 255, 255),
        -1,
    )

    # Enlarge using nearest-neighbor interpolation.
    # This preserves the original pixel boundaries.
    detail = cv2.resize(
        detail,
        None,
        fx=DETAIL_SCALE,
        fy=DETAIL_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.putText(
        detail,
        (
            f"Frame {frame_number} | "
            f"Min distance: {distance:.2f} px"
        ),
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return detail


# ============================================================
# MASK OVERLAY
# ============================================================

def draw_masks(frame, frame_number, indexed_masks):
    output = frame.copy()

    reconstructed_masks = {}

    for object_id in ["1", "2"]:
        key = (frame_number, object_id)

        if key not in indexed_masks:
            raise RuntimeError(
                f"No mask found for {key}"
            )

        record = indexed_masks[key]

        mask = reconstruct_mask(
            record,
            frame.shape,
        )

        reconstructed_masks[object_id] = mask

        color = OBJECT_COLORS[object_id]

        overlay = output.copy()
        overlay[mask > 0] = color

        output = cv2.addWeighted(
            overlay,
            ALPHA,
            output,
            1 - ALPHA,
            0,
        )

        draw_mask_contour(
            output,
            mask,
            color,
        )

        confidence = record.confidence
        bounds = record.bounds

        left = int(bounds.left)
        top = int(bounds.top)

        label = (
            f"ID {object_id} - "
            f"{OBJECT_NAMES[object_id]} "
            f"({confidence:.3f})"
        )

        cv2.putText(
            output,
            label,
            (left, max(25, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    mask_a = reconstructed_masks["1"]
    mask_b = reconstructed_masks["2"]

    distance, closest_pair = find_closest_mask_pixels(
        mask_a,
        mask_b,
    )

    cv2.putText(
        output,
        f"Minimum mask distance: {distance:.2f} px",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        f"FRAME {frame_number}",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    detail = create_contact_detail(
        frame,
        mask_a,
        mask_b,
        distance,
        closest_pair,
        frame_number,
    )

    return output, detail


# ============================================================
# FRAME EXTRACTION
# ============================================================

def read_frame(cap, frame_number):
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number,
    )

    success, frame = cap.read()

    if not success:
        raise RuntimeError(
            f"Could not read frame {frame_number}."
        )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video file not found: {VIDEO_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading SAM response...")

    raw_text = load_sam_response()

    print("Processing masks...")

    masks = parse_masks(raw_text)

    indexed_masks = index_masks(masks)

    print(f"Masks loaded: {len(masks)}")

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    visualized_frames = []
    detail_frames = []

    try:
        for frame_number in FRAMES_TO_VISUALIZE:
            frame = read_frame(
                cap,
                frame_number,
            )

            result, detail = draw_masks(
                frame,
                frame_number,
                indexed_masks,
            )

            output_path = (
                OUTPUT_DIR
                / f"frame_{frame_number:02d}_overlay.jpg"
            )

            detail_path = (
                OUTPUT_DIR
                / f"frame_{frame_number:02d}_detail.jpg"
            )

            if not cv2.imwrite(
                str(output_path),
                result,
            ):
                raise RuntimeError(
                    f"Could not save image: {output_path}"
                )

            if not cv2.imwrite(
                str(detail_path),
                detail,
            ):
                raise RuntimeError(
                    f"Could not save image: {detail_path}"
                )

            preview = cv2.resize(
                result,
                (640, 480),
            )

            visualized_frames.append(preview)
            detail_frames.append(detail)

            print(f"Generated: {output_path}")
            print(f"Generated: {detail_path}")

    finally:
        cap.release()

    comparison = np.hstack(
        visualized_frames
    )

    comparison_path = (
        OUTPUT_DIR
        / "dual_tracking_comparison.jpg"
    )

    if not cv2.imwrite(
        str(comparison_path),
        comparison,
    ):
        raise RuntimeError(
            "Could not save the comparison image."
        )

    detail_comparison = np.hstack(
        detail_frames
    )

    detail_comparison_path = (
        OUTPUT_DIR
        / "dual_tracking_contact_details.jpg"
    )

    if not cv2.imwrite(
        str(detail_comparison_path),
        detail_comparison,
    ):
        raise RuntimeError(
            "Could not save the detail comparison image."
        )

    print("\nVisualization completed.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Comparison image: {comparison_path}")
    print(f"Detail comparison image: {detail_comparison_path}")


if __name__ == "__main__":
    main()
