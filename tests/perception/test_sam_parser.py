

from pathlib import Path

import pytest

from src.perception.sam_parser import (
    parse_sam_line,
    parse_sam_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAM_OUTPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "sam_episode_000025_raw.txt"
)


@pytest.fixture
def real_sam_detections():
    if not SAM_OUTPUT.is_file():
        pytest.skip(
            "Real SAM evaluation output is not available."
        )

    return parse_sam_file(SAM_OUTPUT)


def test_parse_real_sam_output(real_sam_detections):
    detections = real_sam_detections

    assert len(detections) == 36

    assert detections[0].frame == 15
    assert detections[-1].frame == 50

    assert all(item.object_id == 1 for item in detections)


def test_parse_first_detection(real_sam_detections):
    first = real_sam_detections[0]

    assert first.x1 == 966
    assert first.y1 == 798
    assert first.x2 == 996
    assert first.y2 == 830

    assert first.center_x == 981
    assert first.center_y == 814

    assert first.image_width == 1280
    assert first.image_height == 960

    assert first.confidence == pytest.approx(0.58203125)


def test_mask_data_is_preserved(real_sam_detections):
    first = real_sam_detections[0]

    assert first.mask_data is not None
    assert len(first.mask_data) > 0
    assert first.mask_x == 0
    assert first.mask_y == 0


def test_invalid_line_raises_error():
    with pytest.raises(ValueError):
        parse_sam_line("invalid SAM output")


def test_invalid_box_raises_error():
    line = (
        "<15f>1"
        "<|box;x1=100;y1=100;x2=50;y2=150;w=1280;h=960;c=0.9|>"
    )

    with pytest.raises(ValueError):
        parse_sam_line(line)
