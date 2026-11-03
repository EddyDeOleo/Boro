"""Reconstruct SAM local masks in full-frame coordinates."""

from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from meta_sam_parser import SegmentationMaskRecord

from src.perception.sam_mask_decoder import decode_segmentation_mask


class SAMMaskReconstructionError(ValueError):
    """Raised when a SAM mask cannot be reconstructed in frame coordinates."""


def _validate_frame_dimensions(
    frame_width: int,
    frame_height: int,
) -> None:
    """Validate the dimensions of the source video frame."""
    if (
        isinstance(frame_width, bool)
        or not isinstance(frame_width, Integral)
        or frame_width <= 0
    ):
        raise SAMMaskReconstructionError(
            "Frame width must be a positive integer."
        )

    if (
        isinstance(frame_height, bool)
        or not isinstance(frame_height, Integral)
        or frame_height <= 0
    ):
        raise SAMMaskReconstructionError(
            "Frame height must be a positive integer."
        )


def _validate_bounds(
    record: SegmentationMaskRecord,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Validate and normalize half-open source-frame bounds."""
    bounds = record.bounds

    coordinates = (
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
    )

    normalized: list[int] = []

    for coordinate in coordinates:
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, Real)
            or not np.isfinite(coordinate)
        ):
            raise SAMMaskReconstructionError(
                "Mask bounds must contain finite numeric coordinates."
            )

        if not float(coordinate).is_integer():
            raise SAMMaskReconstructionError(
                "Mask bounds must use integer pixel coordinates."
            )

        normalized.append(int(coordinate))

    left, top, right, bottom = normalized

    if left < 0 or top < 0:
        raise SAMMaskReconstructionError(
            "Mask bounds cannot contain negative coordinates."
        )

    if right <= left or bottom <= top:
        raise SAMMaskReconstructionError(
            "Mask bounds must define a non-empty area."
        )

    if right > frame_width or bottom > frame_height:
        raise SAMMaskReconstructionError(
            "Mask bounds exceed the source frame dimensions."
        )

    return left, top, right, bottom


def _resize_binary_mask(
    mask: NDArray[np.uint8],
    target_width: int,
    target_height: int,
) -> NDArray[np.uint8]:
    """Resize a binary mask using deterministic nearest-neighbor sampling."""
    source_height, source_width = mask.shape

    if source_width == target_width and source_height == target_height:
        return mask.copy()

    source_x = np.floor(
        (np.arange(target_width, dtype=np.float64) + 0.5)
        * source_width
        / target_width
    ).astype(np.intp)

    source_y = np.floor(
        (np.arange(target_height, dtype=np.float64) + 0.5)
        * source_height
        / target_height
    ).astype(np.intp)

    source_x = np.clip(source_x, 0, source_width - 1)
    source_y = np.clip(source_y, 0, source_height - 1)

    return mask[np.ix_(source_y, source_x)].copy()


def reconstruct_frame_mask(
    record: SegmentationMaskRecord,
    frame_width: int,
    frame_height: int,
) -> NDArray[np.uint8]:
    """Reconstruct a SAM local mask as a full-frame binary raster.

    The record bounds use half-open source-frame coordinates:
    left and top are inclusive, while right and bottom are exclusive.

    If the decoded local mask dimensions differ from the bounds dimensions,
    nearest-neighbor sampling is used to resize the local mask.

    Args:
        record: A complete SAM segmentation mask record.
        frame_width: Width of the original video frame.
        frame_height: Height of the original video frame.

    Returns:
        A writable two-dimensional uint8 array with the dimensions
        (frame_height, frame_width). Pixel values are 0 or 1.

    Raises:
        TypeError: If record is not a SegmentationMaskRecord.
        SAMMaskReconstructionError: If frame dimensions, bounds, or decoded
            mask structure are invalid.
        InvalidSegmentationMaskError: If the underlying decoder rejects
            the encoded mask.
    """
    if not isinstance(record, SegmentationMaskRecord):
        raise TypeError(
            "record must be a SegmentationMaskRecord instance."
        )

    _validate_frame_dimensions(frame_width, frame_height)

    left, top, right, bottom = _validate_bounds(
        record,
        int(frame_width),
        int(frame_height),
    )

    local_mask = decode_segmentation_mask(record.mask)

    if not isinstance(local_mask, np.ndarray):
        raise SAMMaskReconstructionError(
            "Decoded mask must be a NumPy array."
        )

    if local_mask.ndim != 2:
        raise SAMMaskReconstructionError(
            "Decoded mask must be a two-dimensional array."
        )

    expected_shape = (record.mask.height, record.mask.width)

    if local_mask.shape != expected_shape:
        raise SAMMaskReconstructionError(
            "Decoded mask dimensions do not match the declared mask dimensions."
        )

    if local_mask.size == 0:
        raise SAMMaskReconstructionError(
            "Decoded mask cannot be empty."
        )

    if not np.isin(local_mask, (0, 1)).all():
        raise SAMMaskReconstructionError(
            "Decoded mask must contain only binary values 0 and 1."
        )

    target_width = right - left
    target_height = bottom - top

    resized_mask = _resize_binary_mask(
        local_mask.astype(np.uint8, copy=False),
        target_width,
        target_height,
    )

    frame_mask = np.zeros(
        (int(frame_height), int(frame_width)),
        dtype=np.uint8,
    )

    frame_mask[top:bottom, left:right] = resized_mask

    return frame_mask