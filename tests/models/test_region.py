
import pytest

from src.models.region import (
    RegionDefinition,
    RegionObservation,
    RegionRelation,
    RegionType,
)


@pytest.fixture
def sample_region():
    return RegionDefinition(
        region_id="target_zone_01",
        name="Target Zone",
        region_type=RegionType.TARGET_ZONE,
        polygon=(
            (100.0, 100.0),
            (200.0, 100.0),
            (200.0, 200.0),
            (100.0, 200.0),
        ),
        frame_width=1280,
        frame_height=960,
    )


def test_region_definition_creation(sample_region):
    assert sample_region.region_id == "target_zone_01"
    assert sample_region.region_type == RegionType.TARGET_ZONE


def test_region_contains_point_inside(sample_region):
    assert sample_region.contains_point((150.0, 150.0))


def test_region_does_not_contain_point_outside(sample_region):
    assert not sample_region.contains_point((250.0, 150.0))


def test_region_contains_boundary_point(sample_region):
    assert sample_region.contains_point((100.0, 150.0))


def test_region_rejects_polygon_with_less_than_three_points():
    with pytest.raises(ValueError):
        RegionDefinition(
            region_id="invalid_region",
            name="Invalid Region",
            region_type=RegionType.TARGET_ZONE,
            polygon=((100.0, 100.0), (200.0, 200.0)),
            frame_width=1280,
            frame_height=960,
        )


def test_region_rejects_out_of_frame_coordinates():
    with pytest.raises(ValueError):
        RegionDefinition(
            region_id="invalid_region",
            name="Invalid Region",
            region_type=RegionType.TARGET_ZONE,
            polygon=(
                (100.0, 100.0),
                (1400.0, 100.0),
                (200.0, 200.0),
            ),
            frame_width=1280,
            frame_height=960,
        )


def test_region_observation_creation():
    observation = RegionObservation(
        entity_id="cube",
        region_id="target_zone_01",
        frame_index=25,
        relation=RegionRelation.INSIDE,
        centroid_inside=True,
        intersection_area_pixels=1200.0,
        overlap_ratio=0.95,
    )

    assert observation.relation == RegionRelation.INSIDE
    assert observation.overlap_ratio == 0.95


def test_region_observation_rejects_invalid_overlap_ratio():
    with pytest.raises(ValueError):
        RegionObservation(
            entity_id="cube",
            region_id="target_zone_01",
            frame_index=25,
            relation=RegionRelation.INSIDE,
            overlap_ratio=1.5,
        )


def test_region_rejects_zero_area_polygon():
    with pytest.raises(ValueError):
        RegionDefinition(
            region_id="invalid_region",
            name="Invalid Region",
            region_type=RegionType.TARGET_ZONE,
            polygon=(
                (100.0, 100.0),
                (150.0, 150.0),
                (200.0, 200.0),
            ),
            frame_width=1280,
            frame_height=960,
        )


def test_region_rejects_self_intersecting_polygon():
    with pytest.raises(ValueError):
        RegionDefinition(
            region_id="invalid_region",
            name="Invalid Region",
            region_type=RegionType.TARGET_ZONE,
            polygon=(
                (100.0, 100.0),
                (200.0, 200.0),
                (100.0, 200.0),
                (200.0, 100.0),
            ),
            frame_width=1280,
            frame_height=960,
        )


def test_region_rejects_non_finite_coordinates():
    with pytest.raises(ValueError):
        RegionDefinition(
            region_id="invalid_region",
            name="Invalid Region",
            region_type=RegionType.TARGET_ZONE,
            polygon=(
                (100.0, 100.0),
                (float("nan"), 100.0),
                (200.0, 200.0),
            ),
            frame_width=1280,
            frame_height=960,
        )


def test_region_accepts_json_style_polygon():
    region = RegionDefinition(
        region_id="target_zone_01",
        name="Target Zone",
        region_type=RegionType.TARGET_ZONE,
        polygon=[
            [100.0, 100.0],
            [200.0, 100.0],
            [200.0, 200.0],
            [100.0, 200.0],
        ],
        frame_width=1280,
        frame_height=960,
    )

    assert isinstance(region.polygon, tuple)
    assert isinstance(region.polygon[0], tuple)


def test_region_rejects_non_finite_query_point(sample_region):
    with pytest.raises(ValueError):
        sample_region.contains_point((float("nan"), 150.0))

