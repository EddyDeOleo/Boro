
import numpy as np
import pytest

from src.adapters.sam_observation_adapter import (
    adapt_sam_observations,
    build_entity_observations,
    build_region_observations,
    build_spatial_relationships,
)
from src.models.region import (
    RegionDefinition,
    RegionRelation,
    RegionType,
)
from src.perception.mask_analyzer import (
    MaskMetrics,
    MaskRelationshipMetrics,
    RelativeMetrics,
)


OBJECT_ID_MAP = {
    "1": "red_cube",
    "2": "gripper",
}


def make_region():
    return RegionDefinition(
        region_id="target_zone_01",
        name="Target Zone",
        region_type=RegionType.TARGET_ZONE,
        coordinate_system="pixel",
        frame_width=100,
        frame_height=100,
        reference_frame=0,
        polygon=[
            (20, 20),
            (80, 20),
            (80, 80),
            (20, 80),
        ],
    )


def make_object_metrics(
    frame=0,
    object_id="1",
    area=4,
    centroid_x=10.5,
    centroid_y=11.5,
    bbox=(10, 11, 12, 13),
    iou_previous=None,
    area_ratio=None,
):
    return MaskMetrics(
        frame=frame,
        object_id=object_id,
        area=area,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        bbox=bbox,
        iou_previous=iou_previous,
        area_ratio=area_ratio,
    )


def make_relationship_metrics(
    frame=0,
    object_a_id="1",
    object_b_id="2",
):
    return MaskRelationshipMetrics(
        frame=frame,
        object_a_id=object_a_id,
        object_b_id=object_b_id,
        min_distance=1.0,
        intersection_area=0,
        iou=0.0,
        overlap_a=0.0,
        overlap_b=0.0,
        normalized_delta_x=0.2,
        normalized_delta_y=0.3,
    )


def make_relative_metrics(frame=0):
    return RelativeMetrics(
        frame=frame,
        distance=53.01,
        delta_x=20.0,
        delta_y=-49.0,
    )


def test_build_entity_observations_maps_sam_ids_and_timestamps():
    metrics = [
        make_object_metrics()
    ]

    result = build_entity_observations(
        object_metrics=metrics,
        object_id_map=OBJECT_ID_MAP,
        confidences={(0, "1"): 0.95},
        fps=30,
    )

    assert len(result) == 1

    observation = result[0]

    assert observation.observation_id == "sam:red_cube:0"
    assert observation.entity_id == "red_cube"
    assert observation.frame_index == 0
    assert observation.timestamp_seconds == 0
    assert observation.centroid == (10.5, 11.5)
    assert observation.bbox == (10, 11, 12, 13)
    assert observation.area_pixels == 4
    assert observation.confidence == 0.95
    assert observation.attributes["sam_object_id"] == "1"
    assert observation.source == "sam-3.1"


def test_build_entity_observations_uses_clip_relative_time():
    metrics = [
        make_object_metrics(
            frame=30,
            centroid_x=10,
            centroid_y=10,
            bbox=(9, 9, 11, 11),
            iou_previous=0.8,
            area_ratio=1.0,
        )
    ]

    result = build_entity_observations(
        object_metrics=metrics,
        object_id_map=OBJECT_ID_MAP,
        confidences={},
        fps=30,
        time_offset_seconds=5.8,
    )

    assert result[0].timestamp_seconds == pytest.approx(6.8)


def test_build_entity_observations_accepts_custom_source():
    result = build_entity_observations(
        object_metrics=[make_object_metrics()],
        object_id_map=OBJECT_ID_MAP,
        confidences={},
        fps=30,
        source="custom-perception-source",
    )

    assert result[0].source == "custom-perception-source"


def test_build_entity_observations_rejects_unknown_sam_id():
    metrics = [
        make_object_metrics(object_id="99")
    ]

    with pytest.raises(ValueError, match="no semantic mapping"):
        build_entity_observations(
            object_metrics=metrics,
            object_id_map=OBJECT_ID_MAP,
            confidences={},
            fps=30,
        )


def test_build_entity_observations_rejects_invalid_fps():
    with pytest.raises(ValueError, match="fps"):
        build_entity_observations(
            object_metrics=[],
            object_id_map=OBJECT_ID_MAP,
            confidences={},
            fps=0,
        )


def test_build_spatial_relationships_combines_geometry_metrics():
    relationship = make_relationship_metrics()
    relative = make_relative_metrics()

    result = build_spatial_relationships(
        relationship_metrics=[relationship],
        relative_metrics=[relative],
        object_id_map=OBJECT_ID_MAP,
        relative_entity_ids=("1", "2"),
    )

    assert len(result) == 1

    observation = result[0]

    assert observation.entity_a_id == "red_cube"
    assert observation.entity_b_id == "gripper"
    assert observation.distance_pixels == 1.0
    assert observation.delta_x_pixels == 20.0
    assert observation.delta_y_pixels == -49.0
    assert observation.intersection_area_pixels == 0
    assert observation.iou == 0.0
    assert observation.attributes["centroid_distance_pixels"] == 53.01
    assert observation.attributes["physical_contact_established"] is False
    assert observation.attributes["source"] == "sam-3.1"


def test_build_spatial_relationships_accepts_custom_source():
    result = build_spatial_relationships(
        relationship_metrics=[make_relationship_metrics()],
        relative_metrics=[make_relative_metrics()],
        object_id_map=OBJECT_ID_MAP,
        relative_entity_ids=("1", "2"),
        source="custom-perception-source",
    )

    assert (
        result[0].attributes["source"]
        == "custom-perception-source"
    )


def test_build_spatial_relationships_rejects_wrong_pair_orientation():
    relationship = make_relationship_metrics()
    relative = make_relative_metrics()

    with pytest.raises(ValueError, match="orientation"):
        build_spatial_relationships(
            relationship_metrics=[relationship],
            relative_metrics=[relative],
            object_id_map=OBJECT_ID_MAP,
            relative_entity_ids=("2", "1"),
        )


def test_build_spatial_relationships_rejects_missing_relative_metrics():
    with pytest.raises(ValueError, match="Missing relative metrics"):
        build_spatial_relationships(
            relationship_metrics=[make_relationship_metrics()],
            relative_metrics=[],
            object_id_map=OBJECT_ID_MAP,
            relative_entity_ids=("1", "2"),
        )


def test_build_region_observations_detects_inside_mask():
    region = make_region()

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:45, 40:45] = 1

    result = build_region_observations(
        masks_by_frame={0: {"1": mask}},
        object_id_map=OBJECT_ID_MAP,
        region=region,
        sam_object_id="1",
    )

    assert len(result) == 1

    observation = result[0]

    assert observation.entity_id == "red_cube"
    assert observation.region_id == "target_zone_01"
    assert observation.relation == RegionRelation.INSIDE
    assert observation.centroid_inside is True
    assert observation.intersection_area_pixels == 25
    assert observation.overlap_ratio == 1.0


def test_build_region_observations_detects_partial_intersection():
    region = make_region()

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:45, 78:83] = 1

    result = build_region_observations(
        masks_by_frame={0: {"1": mask}},
        object_id_map=OBJECT_ID_MAP,
        region=region,
        sam_object_id="1",
    )

    observation = result[0]

    assert observation.relation == RegionRelation.PARTIALLY_INTERSECTING
    assert observation.intersection_area_pixels == 15
    assert observation.overlap_ratio == pytest.approx(0.6)


def test_build_region_observations_detects_outside_mask():
    region = make_region()

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[5:10, 5:10] = 1

    result = build_region_observations(
        masks_by_frame={0: {"1": mask}},
        object_id_map=OBJECT_ID_MAP,
        region=region,
        sam_object_id="1",
    )

    observation = result[0]

    assert observation.relation == RegionRelation.OUTSIDE
    assert observation.centroid_inside is False
    assert observation.intersection_area_pixels == 0
    assert observation.overlap_ratio == 0.0


def test_build_region_observations_rejects_wrong_mask_dimensions():
    region = make_region()
    mask = np.ones((50, 50), dtype=np.uint8)

    with pytest.raises(ValueError, match="does not match"):
        build_region_observations(
            masks_by_frame={0: {"1": mask}},
            object_id_map=OBJECT_ID_MAP,
            region=region,
            sam_object_id="1",
        )


def test_build_region_observations_rejects_empty_mask():
    region = make_region()
    mask = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(ValueError, match="Empty mask"):
        build_region_observations(
            masks_by_frame={0: {"1": mask}},
            object_id_map=OBJECT_ID_MAP,
            region=region,
            sam_object_id="1",
        )


def test_adapt_sam_observations_propagates_source():
    result = adapt_sam_observations(
        object_metrics=[
            make_object_metrics(object_id="1"),
            make_object_metrics(
                object_id="2",
                centroid_x=30,
                centroid_y=30,
                bbox=(29, 29, 31, 31),
            ),
        ],
        relative_metrics=[make_relative_metrics()],
        relationship_metrics=[make_relationship_metrics()],
        masks_by_frame={},
        object_id_map=OBJECT_ID_MAP,
        confidences={},
        fps=30,
        relative_entity_ids=("1", "2"),
        source="custom-perception-source",
    )

    assert len(result.entity_observations) == 2
    assert len(result.spatial_relationships) == 1
    assert result.region_observations == ()

    for observation in result.entity_observations:
        assert observation.source == "custom-perception-source"

    assert (
        result.spatial_relationships[0].attributes["source"]
        == "custom-perception-source"
    )
