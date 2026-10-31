
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class MaskMetrics:
    """Geometric metrics for a segmented object."""

    frame: int
    object_id: str
    area: int
    centroid_x: float
    centroid_y: float
    bbox: tuple[int, int, int, int]
    iou_previous: Optional[float]
    area_ratio: Optional[float]


@dataclass(frozen=True)
class RelativeMetrics:
    """Relative position between the centroids of two objects."""

    frame: int
    distance: float
    delta_x: float
    delta_y: float


@dataclass(frozen=True)
class MaskRelationshipMetrics:
    """Geometric relationship between two masks in the same frame."""

    frame: int

    object_a_id: str
    object_b_id: str

    min_distance: float

    intersection_area: int
    iou: float

    overlap_a: float
    overlap_b: float

    normalized_delta_x: float
    normalized_delta_y: float


def _validate_mask(mask: np.ndarray, name: str) -> np.ndarray:
    """Validate a mask and return its binary representation."""

    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray.")

    if mask.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")

    binary_mask = mask > 0

    if not np.any(binary_mask):
        raise ValueError(f"{name} is empty.")

    return binary_mask


def calculate_mask_metrics(
    mask: np.ndarray,
    frame: int,
    object_id: str,
    previous_mask: Optional[np.ndarray] = None,
) -> MaskMetrics:
    """
    Calculate geometric metrics for a binary mask.

    The mask must have the same dimensions as the full frame.
    Non-zero pixels represent the segmented object.
    """

    binary_mask = _validate_mask(mask, "mask")

    y_coords, x_coords = np.where(binary_mask)

    area = int(len(x_coords))

    centroid_x = float(np.mean(x_coords))
    centroid_y = float(np.mean(y_coords))

    x_min = int(np.min(x_coords))
    y_min = int(np.min(y_coords))
    x_max = int(np.max(x_coords)) + 1
    y_max = int(np.max(y_coords)) + 1

    iou_previous = None
    area_ratio = None

    if previous_mask is not None:
        previous_binary = _validate_mask(
            previous_mask,
            "previous_mask",
        )

        if previous_binary.shape != binary_mask.shape:
            raise ValueError(
                "The previous mask must have the same dimensions."
            )

        intersection = np.count_nonzero(
            binary_mask & previous_binary
        )

        union = np.count_nonzero(
            binary_mask | previous_binary
        )

        iou_previous = float(intersection / union)

        previous_area = int(np.count_nonzero(previous_binary))

        area_ratio = float(area / previous_area)

    return MaskMetrics(
        frame=frame,
        object_id=str(object_id),
        area=area,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        bbox=(x_min, y_min, x_max, y_max),
        iou_previous=iou_previous,
        area_ratio=area_ratio,
    )


def calculate_relative_metrics(
    cube_metrics: MaskMetrics,
    gripper_metrics: MaskMetrics,
) -> RelativeMetrics:
    """
    Calculate the distance and relative position between two objects
    using their centroids.

    Parameter names are preserved for compatibility with existing code.
    """

    if cube_metrics.frame != gripper_metrics.frame:
        raise ValueError(
            "Both metrics must belong to the same frame."
        )

    delta_x = (
        cube_metrics.centroid_x
        - gripper_metrics.centroid_x
    )

    delta_y = (
        cube_metrics.centroid_y
        - gripper_metrics.centroid_y
    )

    distance = float(np.hypot(delta_x, delta_y))

    return RelativeMetrics(
        frame=cube_metrics.frame,
        distance=distance,
        delta_x=float(delta_x),
        delta_y=float(delta_y),
    )


def calculate_mask_relationship_metrics(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    frame: int,
    object_a_id: str,
    object_b_id: str,
) -> MaskRelationshipMetrics:
    """
    Calculate the geometric relationship between two masks.

    Includes:
    - Minimum distance between pixels in both masks.
    - Intersection area.
    - Intersection over Union (IoU).
    - Overlap ratio relative to each object.
    - Normalized relative position of A with respect to B's bounding box.

    Distance is measured in pixels between pixel centers.
    A value of zero means that the masks share at least one pixel.

    These metrics do not establish physical contact.
    """

    binary_a = _validate_mask(mask_a, "mask_a")
    binary_b = _validate_mask(mask_b, "mask_b")

    if binary_a.shape != binary_b.shape:
        raise ValueError(
            "Both masks must have the same dimensions."
        )

    # Extract pixel coordinates for both objects.
    y_a, x_a = np.where(binary_a)
    y_b, x_b = np.where(binary_b)

    area_a = len(x_a)
    area_b = len(x_b)

    # Calculate intersection and union.
    intersection_area = int(
        np.count_nonzero(binary_a & binary_b)
    )

    union_area = int(
        np.count_nonzero(binary_a | binary_b)
    )

    iou = float(intersection_area / union_area)

    overlap_a = float(intersection_area / area_a)
    overlap_b = float(intersection_area / area_b)

    # Calculate the minimum Euclidean distance between pixels.
    # Process points in chunks to limit memory consumption.
    points_a = np.column_stack((x_a, y_a)).astype(np.float64)
    points_b = np.column_stack((x_b, y_b)).astype(np.float64)

    min_distance_squared = float("inf")
    chunk_size = 256

    for start in range(0, len(points_a), chunk_size):
        chunk = points_a[start:start + chunk_size]

        dx = chunk[:, None, 0] - points_b[None, :, 0]
        dy = chunk[:, None, 1] - points_b[None, :, 1]

        distances_squared = dx * dx + dy * dy

        block_min = float(np.min(distances_squared))

        if block_min < min_distance_squared:
            min_distance_squared = block_min

        if min_distance_squared == 0:
            break

    min_distance = float(np.sqrt(min_distance_squared))

    # Calculate the centroid of each mask.
    centroid_a_x = float(np.mean(x_a))
    centroid_a_y = float(np.mean(y_a))

    centroid_b_x = float(np.mean(x_b))
    centroid_b_y = float(np.mean(y_b))

    # Calculate B's bounding box using half-open coordinates.
    width_b = int(np.max(x_b) - np.min(x_b) + 1)
    height_b = int(np.max(y_b) - np.min(y_b) + 1)

    # Normalize A's position relative to B's dimensions.
    normalized_delta_x = (
        (centroid_a_x - centroid_b_x) / width_b
    )

    normalized_delta_y = (
        (centroid_a_y - centroid_b_y) / height_b
    )

    return MaskRelationshipMetrics(
        frame=frame,
        object_a_id=str(object_a_id),
        object_b_id=str(object_b_id),
        min_distance=min_distance,
        intersection_area=intersection_area,
        iou=iou,
        overlap_a=overlap_a,
        overlap_b=overlap_b,
        normalized_delta_x=float(normalized_delta_x),
        normalized_delta_y=float(normalized_delta_y),
    )
