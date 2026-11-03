
import numpy as np
import pytest

from meta_sam_parser import SegmentationMask

from src.perception import sam_mask_decoder
from src.perception.sam_mask_decoder import (
    SAMMaskDecodingError,
    decode_segmentation_mask,
)


def create_mask(
    width=2,
    height=2,
    encoding="one_bit",
):
    return SegmentationMask(
        encoding=encoding,
        payload="!test",
        width=width,
        height=height,
    )


def test_decodes_binary_raster(monkeypatch):
    mask = create_mask()

    monkeypatch.setattr(
        sam_mask_decoder,
        "decode_mask_to_raster",
        lambda _: bytes([0, 1, 1, 0]),
    )

    result = decode_segmentation_mask(mask)

    expected = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(result, expected)


def test_preserves_original_mask_dimensions(monkeypatch):
    mask = create_mask(width=3, height=2)

    monkeypatch.setattr(
        sam_mask_decoder,
        "decode_mask_to_raster",
        lambda _: bytes([0, 1, 0, 1, 1, 0]),
    )

    result = decode_segmentation_mask(mask)

    assert result.shape == (2, 3)


def test_returns_uint8_array(monkeypatch):
    mask = create_mask()

    monkeypatch.setattr(
        sam_mask_decoder,
        "decode_mask_to_raster",
        lambda _: bytes([0, 1, 0, 1]),
    )

    result = decode_segmentation_mask(mask)

    assert result.dtype == np.uint8


def test_returns_writable_array(monkeypatch):
    mask = create_mask()

    monkeypatch.setattr(
        sam_mask_decoder,
        "decode_mask_to_raster",
        lambda _: bytes([0, 1, 0, 1]),
    )

    result = decode_segmentation_mask(mask)

    result[0, 0] = 1

    assert result[0, 0] == 1


@pytest.mark.parametrize(
    "encoding",
    ["one_bit", "lossless"],
)
def test_supports_available_encodings(monkeypatch, encoding):
    mask = create_mask(encoding=encoding)
    received_masks = []

    def fake_decoder(received_mask):
        received_masks.append(received_mask)
        return bytes([0, 1, 0, 1])

    monkeypatch.setattr(
        sam_mask_decoder,
        "decode_mask_to_raster",
        fake_decoder,
    )

    result = decode_segmentation_mask(mask)

    assert received_masks == [mask]
    assert result.shape == (2, 2)


def test_rejects_invalid_input_type():
    with pytest.raises(TypeError):
        decode_segmentation_mask("invalid")


def test_rejects_incorrect_raster_size(monkeypatch):
    mask = create_mask()

    monkeypatch.setattr(
        sam_mask_decoder,
        "decode_mask_to_raster",
        lambda _: bytes([0, 1]),
    )

    with pytest.raises(SAMMaskDecodingError):
        decode_segmentation_mask(mask)


def test_rejects_non_binary_raster(monkeypatch):
    mask = create_mask()

    monkeypatch.setattr(
        sam_mask_decoder,
        "decode_mask_to_raster",
        lambda _: bytes([0, 1, 2, 0]),
    )

    with pytest.raises(SAMMaskDecodingError):
        decode_segmentation_mask(mask)


def test_rejects_non_bytes_decoder_result(monkeypatch):
    mask = create_mask()

    monkeypatch.setattr(
        sam_mask_decoder,
        "decode_mask_to_raster",
        lambda _: bytearray([0, 1, 0, 1]),
    )

    with pytest.raises(SAMMaskDecodingError):
        decode_segmentation_mask(mask)
