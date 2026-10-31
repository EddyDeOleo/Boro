

"""Tests for event detection configuration."""

from dataclasses import FrozenInstanceError

import pytest

from src.models.event_detection_config import EventDetectionConfig


class TestEventDetectionConfig:

    def test_create_default_config(self):
        config = EventDetectionConfig()

        assert config.min_event_duration_seconds == 0.20
        assert config.min_observations == 3
        assert config.max_frame_gap == 1
        assert config.min_object_displacement_pixels == 5.0
        assert config.max_relative_distance_pixels == 30.0
        assert config.min_relative_separation_increase_pixels == 5.0
        assert config.min_motion_similarity == 0.75
        assert config.max_stationary_speed_pixels_per_second == 3.0
        assert config.min_region_overlap_ratio == 0.80
        assert config.min_stable_duration_seconds == 0.50

    def test_create_custom_config(self):
        config = EventDetectionConfig(
            min_event_duration_seconds=0.5,
            min_observations=5,
            max_frame_gap=2,
            min_object_displacement_pixels=10.0,
            max_relative_distance_pixels=50.0,
            min_relative_separation_increase_pixels=12.0,
            min_motion_similarity=0.9,
            max_stationary_speed_pixels_per_second=2.0,
            min_region_overlap_ratio=0.95,
            min_stable_duration_seconds=1.0,
        )

        assert config.min_event_duration_seconds == 0.5
        assert config.min_observations == 5
        assert config.max_frame_gap == 2
        assert config.min_object_displacement_pixels == 10.0
        assert config.max_relative_distance_pixels == 50.0
        assert config.min_relative_separation_increase_pixels == 12.0
        assert config.min_motion_similarity == 0.9
        assert config.max_stationary_speed_pixels_per_second == 2.0
        assert config.min_region_overlap_ratio == 0.95
        assert config.min_stable_duration_seconds == 1.0

    @pytest.mark.parametrize(
        "field_name",
        [
            "min_event_duration_seconds",
            "min_object_displacement_pixels",
            "max_relative_distance_pixels",
            "min_relative_separation_increase_pixels",
            "max_stationary_speed_pixels_per_second",
            "min_stable_duration_seconds",
        ],
    )
    @pytest.mark.parametrize(
        "value",
        [
            0,
            -1,
            -0.5,
            float("inf"),
            float("-inf"),
            float("nan"),
            True,
            "1.0",
            None,
        ],
    )
    def test_positive_numeric_fields_reject_invalid_values(
        self,
        field_name,
        value,
    ):
        with pytest.raises(ValueError, match=field_name):
            EventDetectionConfig(**{field_name: value})

    @pytest.mark.parametrize(
        "value",
        [2, 3, 5, 10],
    )
    def test_min_observations_accepts_valid_values(self, value):
        config = EventDetectionConfig(min_observations=value)

        assert config.min_observations == value

    @pytest.mark.parametrize(
        "value",
        [-1, 0, 1, 1.5, True, "3", None],
    )
    def test_invalid_min_observations_raises_value_error(self, value):
        with pytest.raises(
            ValueError,
            match="min_observations must be an integer >= 2",
        ):
            EventDetectionConfig(min_observations=value)

    @pytest.mark.parametrize(
        "value",
        [0, 1, 2, 10],
    )
    def test_max_frame_gap_accepts_valid_values(self, value):
        config = EventDetectionConfig(max_frame_gap=value)

        assert config.max_frame_gap == value

    @pytest.mark.parametrize(
        "value",
        [-1, 1.5, True, "1", None],
    )
    def test_invalid_max_frame_gap_raises_value_error(self, value):
        with pytest.raises(
            ValueError,
            match="max_frame_gap must be a non-negative integer",
        ):
            EventDetectionConfig(max_frame_gap=value)

    @pytest.mark.parametrize(
        "value",
        [0.0, 0.5, 0.75, 1.0],
    )
    def test_min_motion_similarity_accepts_valid_values(self, value):
        config = EventDetectionConfig(min_motion_similarity=value)

        assert config.min_motion_similarity == value

    @pytest.mark.parametrize(
        "value",
        [-0.1, 1.1, float("inf"), float("-inf"), float("nan"), True, "0.75"],
    )
    def test_invalid_min_motion_similarity_raises_value_error(self, value):
        with pytest.raises(
            ValueError,
            match="min_motion_similarity must be between 0 and 1",
        ):
            EventDetectionConfig(min_motion_similarity=value)

    @pytest.mark.parametrize(
        "value",
        [0.0, 0.5, 1.0],
    )
    def test_min_region_overlap_ratio_accepts_valid_values(self, value):
        config = EventDetectionConfig(min_region_overlap_ratio=value)

        assert config.min_region_overlap_ratio == value

    @pytest.mark.parametrize(
        "value",
        [-0.1, 1.1, float("inf"), float("-inf"), float("nan"), True, "0.8"],
    )
    def test_invalid_min_region_overlap_ratio_raises_value_error(self, value):
        with pytest.raises(ValueError, match="min_region_overlap_ratio"):
            EventDetectionConfig(min_region_overlap_ratio=value)

    def test_config_is_immutable(self):
        config = EventDetectionConfig()

        with pytest.raises(FrozenInstanceError):
            config.min_observations = 10
