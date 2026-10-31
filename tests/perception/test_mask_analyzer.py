
import numpy as np
import pytest

from src.perception.mask_analyzer import (
    calculate_mask_metrics,
    calculate_relative_metrics,
    calculate_mask_relationship_metrics,
)


# ============================================================
# INDIVIDUAL MASK METRICS
# ============================================================

def test_calculate_mask_metrics():
    mask = np.zeros((100, 100), dtype=np.uint8)

    mask[10:20, 30:40] = 1

    result = calculate_mask_metrics(
        mask=mask,
        frame=0,
        object_id="1",
    )

    assert result.area == 100

    assert result.centroid_x == pytest.approx(34.5)
    assert result.centroid_y == pytest.approx(14.5)

    assert result.bbox == (30, 10, 40, 20)

    assert result.iou_previous is None
    assert result.area_ratio is None


def test_calculate_temporal_metrics():
    previous = np.zeros((100, 100), dtype=np.uint8)
    current = np.zeros((100, 100), dtype=np.uint8)

    previous[10:20, 10:20] = 1
    current[10:20, 15:25] = 1

    result = calculate_mask_metrics(
        mask=current,
        frame=1,
        object_id="1",
        previous_mask=previous,
    )

    assert result.area == 100
    assert result.iou_previous == pytest.approx(1 / 3)
    assert result.area_ratio == pytest.approx(1.0)


def test_empty_mask_raises_error():
    mask = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(ValueError):
        calculate_mask_metrics(
            mask=mask,
            frame=0,
            object_id="1",
        )


# ============================================================
# RELATIVE POSITION BETWEEN OBJECTS
# ============================================================

def create_mask(x, y):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[y:y + 10, x:x + 10] = 1
    return mask


def test_calculate_relative_metrics():
    cube = calculate_mask_metrics(
        mask=create_mask(10, 10),
        frame=0,
        object_id="1",
    )

    gripper = calculate_mask_metrics(
        mask=create_mask(20, 30),
        frame=0,
        object_id="2",
    )

    result = calculate_relative_metrics(
        cube,
        gripper,
    )

    assert result.delta_x == pytest.approx(-10)
    assert result.delta_y == pytest.approx(-20)
    assert result.distance == pytest.approx(np.sqrt(500))


def test_different_frame_relative_metrics_raises_error():
    cube = calculate_mask_metrics(
        create_mask(10, 10),
        frame=0,
        object_id="1",
    )

    gripper = calculate_mask_metrics(
        create_mask(20, 20),
        frame=1,
        object_id="2",
    )

    with pytest.raises(ValueError):
        calculate_relative_metrics(cube, gripper)


# ============================================================
# GEOMETRIC RELATIONSHIP BETWEEN MASKS
# ============================================================

def test_mask_relationship_metrics():
    mask_a = np.zeros((100, 100), dtype=np.uint8)
    mask_b = np.zeros((100, 100), dtype=np.uint8)

    mask_a[10:20, 10:20] = 1
    mask_b[10:20, 25:35] = 1

    result = calculate_mask_relationship_metrics(
        mask_a,
        mask_b,
        frame=0,
        object_a_id="object_a",
        object_b_id="object_b",
    )

    assert result.frame == 0

    # Distance between the closest pixel centers.
    assert result.min_distance == pytest.approx(6.0)

    assert result.intersection_area == 0
    assert result.iou == pytest.approx(0.0)

    assert result.overlap_a == pytest.approx(0.0)
    assert result.overlap_b == pytest.approx(0.0)

    assert result.normalized_delta_x == pytest.approx(-1.5)
    assert result.normalized_delta_y == pytest.approx(0.0)


def test_mask_relationship_with_intersection():
    mask_a = np.zeros((50, 50), dtype=np.uint8)
    mask_b = np.zeros((50, 50), dtype=np.uint8)

    mask_a[10:20, 10:20] = 1
    mask_b[15:25, 15:25] = 1

    result = calculate_mask_relationship_metrics(
        mask_a,
        mask_b,
        frame=1,
        object_a_id="a",
        object_b_id="b",
    )

    assert result.min_distance == pytest.approx(0.0)

    assert result.intersection_area == 25

    assert result.iou == pytest.approx(25 / 175)

    assert result.overlap_a == pytest.approx(0.25)
    assert result.overlap_b == pytest.approx(0.25)


def test_mask_relationship_rejects_different_dimensions():
    mask_a = np.ones((20, 20), dtype=np.uint8)
    mask_b = np.ones((30, 30), dtype=np.uint8)

    with pytest.raises(ValueError):
        calculate_mask_relationship_metrics(
            mask_a,
            mask_b,
            frame=0,
            object_a_id="a",
            object_b_id="b",
        )


def test_mask_relationship_rejects_empty_mask():
    mask_a = np.zeros((20, 20), dtype=np.uint8)
    mask_b = np.ones((20, 20), dtype=np.uint8)

    with pytest.raises(ValueError):
        calculate_mask_relationship_metrics(
            mask_a,
            mask_b,
            frame=0,
            object_a_id="a",
            object_b_id="b",
        )

