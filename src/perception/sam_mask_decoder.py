
"""Decode SAM segmentation masks into binary NumPy arrays."""

import numpy as np
from numpy.typing import NDArray

from meta_sam_parser import SegmentationMask, decode_mask_to_raster


class SAMMaskDecodingError(ValueError):
    """Raised when a decoded SAM mask does not match its expected structure."""


def decode_segmentation_mask(
    mask: SegmentationMask,
) -> NDArray[np.uint8]:
    """Decode a SAM segmentation mask into a binary raster.

    The returned array preserves the original mask dimensions.
    Pixel values are normalized to 0 and 1.

    Args:
        mask: Complete encoded SAM segmentation mask.

    Returns:
        A writable two-dimensional NumPy array with dtype uint8.

    Raises:
        TypeError: If the input is not a SegmentationMask.
        SAMMaskDecodingError: If the decoded raster is structurally invalid.
        InvalidSegmentationMaskError: If the underlying decoder rejects
            the encoded mask.
    """
    if not isinstance(mask, SegmentationMask):
        raise TypeError("mask must be a SegmentationMask instance.")

    raster = decode_mask_to_raster(mask)

    if not isinstance(raster, bytes):
        raise SAMMaskDecodingError(
            "The decoded raster must be bytes."
        )

    expected_size = mask.width * mask.height

    if len(raster) != expected_size:
        raise SAMMaskDecodingError(
            "Decoded raster size does not match mask dimensions."
        )

    array = np.frombuffer(raster, dtype=np.uint8)

    if not np.isin(array, (0, 1)).all():
        raise SAMMaskDecodingError(
            "Decoded raster contains values outside the binary range."
        )

    return array.reshape(mask.height, mask.width).copy()
