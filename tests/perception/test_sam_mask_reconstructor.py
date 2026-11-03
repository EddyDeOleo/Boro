"""Tests for reconstructing SAM masks in full-frame coordinates."""

import numpy as np
import pytest

from meta_sam_parser import (
    FrameReference,
    SegmentationMask,
    SegmentationMaskBounds,
    SegmentationMaskIdentity,
    SegmentationMaskRecord,
)

import src.perception.sam_mask_reconstructor as reconstructor

from src.perception.sam_mask_reconstructor import (
    SAMMaskReconstructionError,
    reconstruct_frame_mask,
)


@pytest.fixture
def mask_record_factory():
    """Create SAM mask records with configurable dimensions and bounds."""

    def create(
        *,
        mask_width=2,
        mask_height=2,
        bounds=(2, 1, 4, 3),
        frame_index=0,
        object_id="1",
        confidence=0.95,
    ):
        return SegmentationMaskRecord(
            order=0,
            object_id=object_id,
            identity=SegmentationMaskIdentity(
                media="video",
                frame_index=frame_index,
                object_id=object_id,
            ),
            revision=0,
            mask=SegmentationMask(
                encoding="one_bit",
                payload="!",
                width=mask_width,
                height=mask_height,
            ),
            bounds=SegmentationMaskBounds(
                left=bounds[0],
                top=bounds[1],
                right=bounds[2],
                bottom=bounds[3],
            ),
            frame=FrameReference(frame_index=frame_index),
            confidence=confidence,
        )

    return create


@pytest.fixture
def install_decoder(monkeypatch):
    """Replace the decoder with a deterministic local-mask fixture."""

    def install(mask):
        def fake_decoder(_):
            return mask.copy()

        monkeypatch.setattr(
            reconstructor,
            "decode_segmentation_mask",
            fake_decoder,
        )

    return install


def test_reconstructs_mask_at_correct_frame_coordinates(
    mask_record_factory,
    install_decoder,
):
    local_mask = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    install_decoder(local_mask)

    record = mask_record_factory(
        bounds=(2, 1, 4, 3),
    )

    result = reconstruct_frame_mask(
        record,
        frame_width=6,
        frame_height=5,
    )

    expected = np.zeros((5, 6), dtype=np.uint8)
    expected[1:3, 2:4] = local_mask

    np.testing.assert_array_equal(result, expected)


def test_returns_full_frame_dimensions(
    mask_record_factory,
    install_decoder,
):
    install_decoder(
        np.ones((2, 2), dtype=np.uint8)
    )

    record = mask_record_factory()

    result = reconstruct_frame_mask(
        record,
        frame_width=1280,
        frame_height=960,
    )

    assert result.shape == (960, 1280)


def test_returns_uint8_binary_mask(
    mask_record_factory,
    install_decoder,
):
    install_decoder(
        np.array(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=np.uint8,
        )
    )

    record = mask_record_factory()

    result = reconstruct_frame_mask(
        record,
        frame_width=6,
        frame_height=5,
    )

    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 1})


def test_returns_writable_array(
    mask_record_factory,
    install_decoder,
):
    install_decoder(
        np.ones((2, 2), dtype=np.uint8)
    )

    record = mask_record_factory()

    result = reconstruct_frame_mask(
        record,
        frame_width=6,
        frame_height=5,
    )

    assert result.flags.writeable


def test_does_not_modify_decoded_local_mask(
    mask_record_factory,
    install_decoder,
):
    local_mask = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    original = local_mask.copy()

    install_decoder(local_mask)

    record = mask_record_factory()

    reconstruct_frame_mask(
        record,
        frame_width=6,
        frame_height=5,
    )

    np.testing.assert_array_equal(local_mask, original)


def test_resizes_local_mask_using_nearest_neighbor(
    mask_record_factory,
    install_decoder,
):
    local_mask = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    install_decoder(local_mask)

    record = mask_record_factory(
        bounds=(1, 1, 5, 5),
    )

    result = reconstruct_frame_mask(
        record,
        frame_width=6,
        frame_height=6,
    )

    expected_local = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )

    expected = np.zeros((6, 6), dtype=np.uint8)
    expected[1:5, 1:5] = expected_local

    np.testing.assert_array_equal(result, expected)


def test_resizes_mask_when_local_dimensions_differ_from_bounds(
    mask_record_factory,
    install_decoder,
):
    local_mask = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    install_decoder(local_mask)

    record = mask_record_factory(
        mask_width=2,
        mask_height=2,
        bounds=(1, 1, 4, 3),
    )

    result = reconstruct_frame_mask(
        record,
        frame_width=6,
        frame_height=5,
    )

    expected = np.zeros((5, 6), dtype=np.uint8)

    expected[1:3, 1:4] = np.array(
        [
            [1, 0, 0],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("frame_width", "frame_height"),
    [
        (0, 10),
        (10, 0),
        (-1, 10),
        (10, -1),
        (1.5, 10),
        (10, 1.5),
        (True, 10),
        (10, False),
    ],
)
def test_rejects_invalid_frame_dimensions(
    mask_record_factory,
    install_decoder,
    frame_width,
    frame_height,
):
    install_decoder(
        np.ones((2, 2), dtype=np.uint8)
    )

    record = mask_record_factory()

    with pytest.raises(SAMMaskReconstructionError):
        reconstruct_frame_mask(
            record,
            frame_width=frame_width,
            frame_height=frame_height,
        )


@pytest.mark.parametrize(
    "bounds",
    [
        (-1, 0, 2, 2),
        (0, -1, 2, 2),
        (3, 1, 2, 3),
        (1, 3, 3, 2),
        (0, 0, 0, 2),
        (0, 0, 2, 0),
        (0, 0, 7, 2),
        (0, 0, 2, 7),
        (0.5, 0, 2, 2),
        (0, 0.5, 2, 2),
    ],
)
def test_rejects_invalid_bounds(
    mask_record_factory,
    install_decoder,
    bounds,
):
    install_decoder(
        np.ones((2, 2), dtype=np.uint8)
    )

    record = mask_record_factory(bounds=bounds)

    with pytest.raises(SAMMaskReconstructionError):
        reconstruct_frame_mask(
            record,
            frame_width=6,
            frame_height=6,
        )


def test_rejects_non_finite_bounds(
    mask_record_factory,
    install_decoder,
):
    install_decoder(
        np.ones((2, 2), dtype=np.uint8)
    )

    record = mask_record_factory(
        bounds=(0, 0, float("inf"), 2),
    )

    with pytest.raises(
        SAMMaskReconstructionError,
        match="finite numeric coordinates",
    ):
        reconstruct_frame_mask(
            record,
            frame_width=6,
            frame_height=6,
        )


def test_rejects_non_record_input():
    with pytest.raises(TypeError):
        reconstruct_frame_mask(
            object(),
            frame_width=10,
            frame_height=10,
        )


@pytest.mark.parametrize(
    "invalid_mask",
    [
        np.array([1, 0, 1, 0], dtype=np.uint8),
        np.array(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=np.uint8,
        ),
        np.empty((0, 0), dtype=np.uint8),
    ],
)
def test_rejects_invalid_decoded_mask_structure(
    mask_record_factory,
    install_decoder,
    invalid_mask,
):
    install_decoder(invalid_mask)

    record = mask_record_factory()

    with pytest.raises(SAMMaskReconstructionError):
        reconstruct_frame_mask(
            record,
            frame_width=6,
            frame_height=5,
        )


def test_rejects_decoded_mask_with_wrong_declared_dimensions(
    mask_record_factory,
    install_decoder,
):
    install_decoder(
        np.ones((3, 2), dtype=np.uint8)
    )

    record = mask_record_factory(
        mask_width=2,
        mask_height=2,
    )

    with pytest.raises(
        SAMMaskReconstructionError,
        match="declared mask dimensions",
    ):
        reconstruct_frame_mask(
            record,
            frame_width=6,
            frame_height=5,
        )


def test_preserves_empty_background_outside_bounds(
    mask_record_factory,
    install_decoder,
):
    install_decoder(
        np.ones((2, 2), dtype=np.uint8)
    )

    record = mask_record_factory(
        bounds=(2, 2, 4, 4),
    )

    result = reconstruct_frame_mask(
        record,
        frame_width=6,
        frame_height=6,
    )

    assert np.count_nonzero(result[:2, :]) == 0
    assert np.count_nonzero(result[4:, :]) == 0
    assert np.count_nonzero(result[:, :2]) == 0
    assert np.count_nonzero(result[:, 4:]) == 0


def test_preserves_mask_when_dimensions_already_match_bounds(
    mask_record_factory,
    install_decoder,
):
    local_mask = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    install_decoder(local_mask)

    record = mask_record_factory(
        bounds=(1, 1, 3, 3),
    )

    result = reconstruct_frame_mask(
        record,
        frame_width=5,
        frame_height=5,
    )

    np.testing.assert_array_equal(
        result[1:3, 1:3],
        local_mask,
    )


def test_supports_mask_touching_right_and_bottom_frame_edges(
    mask_record_factory,
    install_decoder,
):
    install_decoder(
        np.ones((2, 2), dtype=np.uint8)
    )

    record = mask_record_factory(
        bounds=(3, 2, 6, 5),
    )

    result = reconstruct_frame_mask(
        record,
        frame_width=6,
        frame_height=5,
    )

    assert np.all(result[2:5, 3:6] == 1)