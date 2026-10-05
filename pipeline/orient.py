"""
Stage 2 — Orientation normalisation.

Reads the EXIF Orientation tag and rotates/flips the image so it is
visually upright before any CV operations run.  OpenCV contour and
perspective transforms assume a consistently oriented image; skipping
this step causes page-detection to find the wrong quadrilateral on
photos taken in portrait mode.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# EXIF tag 274 values → Pillow transpose operations
# https://exiftool.org/TagNames/EXIF.html
_EXIF_ORIENTATION_TAG = 274
_TRANSPOSE_MAP: dict[int, int | None] = {
    1: None,                          # normal
    2: Image.FLIP_LEFT_RIGHT,
    3: Image.ROTATE_180,
    4: Image.FLIP_TOP_BOTTOM,
    5: Image.TRANSPOSE,               # flip + rotate 90 CW
    6: Image.ROTATE_270,              # 90 CW
    7: Image.TRANSVERSE,              # flip + rotate 90 CCW
    8: Image.ROTATE_90,               # 90 CCW
}


def normalize_orientation(image: np.ndarray) -> np.ndarray:
    """
    Apply EXIF orientation correction to a float32 RGB array.

    The array must have been produced by ``decode_image``; orientation
    information is re-read from the Pillow EXIF data embedded during decode.

    Since the ndarray itself carries no EXIF data, this function accepts a
    float32 array and returns it unmodified — the actual correction is
    performed inside ``decode_image`` via ``ImageOps.exif_transpose``.

    In practice the correct approach is to apply the transpose *before*
    converting to ndarray.  ``decode_image`` delegates here, but the
    function is kept as a separate stage so the pipeline is auditable
    and the stage can be extended (e.g. heuristic upright detection).
    """
    # The array arrives already oriented because decode_image calls
    # _apply_exif_transpose before converting to ndarray.
    # This stage is a no-op in the current implementation but is preserved
    # as an explicit pipeline step for future extension.
    return image


def apply_exif_transpose(pil_img: Image.Image) -> Image.Image:
    """
    Apply EXIF orientation to a Pillow image and strip the tag.
    Called by decode_image before ndarray conversion.
    """
    try:
        exif = pil_img.getexif()
        orientation = exif.get(_EXIF_ORIENTATION_TAG)
    except Exception:
        orientation = None

    if orientation is None or orientation == 1:
        return pil_img

    op = _TRANSPOSE_MAP.get(orientation)
    if op is not None:
        pil_img = pil_img.transpose(op)

    # Remove the orientation tag so downstream readers see a normal image
    try:
        exif[_EXIF_ORIENTATION_TAG] = 1
        pil_img.info["exif"] = exif.tobytes()
    except Exception:
        pass

    return pil_img
