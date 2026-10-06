"""
Optional geometric transform — manual reframing of the user's own pixels.

Applied to the ORIGINAL (oriented) image *before* the extraction pipeline
runs, so the user can rotate and crop their photo manually before automatic
page detection / deskew. This is purely geometric: it rotates and crops the
user's existing pixels, never synthesising new content.

Semantics (matches the caller contract):
    rotate — degrees, clockwise. Rotating a non-square image by a non-multiple
             of 90 enlarges the canvas; exposed corners are filled with WHITE
             (paper) so illumination normalisation still behaves.
    crop   — normalised fractions (0..1) of the ROTATED image, applied after
             rotation. Clamped defensively to the image bounds.

The transform is a no-op when `rotate` is 0/absent and `crop` is absent.

This stage is pure: it reads no globals, touches no filesystem, and never
mutates its input array.
"""

from __future__ import annotations

import cv2
import numpy as np

# White fill for exposed corners. The pipeline carries float32 RGB in [0, 1],
# so paper white is 1.0 on every channel.
_WHITE = 1.0


def apply_transform(image: np.ndarray, transform: dict | None) -> np.ndarray:
    """Rotate (deg, clockwise, white fill) then crop (normalized 0..1).

    `image` is the decoded array in the same form the pipeline's first
    post-decode stage expects (float32 RGB, (H, W, 3), values in [0, 1]).
    Returns a new array; never mutates input. No-op (returns image unchanged)
    when transform is falsy / empty.
    """
    if not transform:
        return image

    rotate = float(transform.get("rotate") or 0.0)
    crop = transform.get("crop")

    # Genuine no-op: no rotation and no crop requested.
    if rotate == 0.0 and not crop:
        return image

    result = image

    if rotate != 0.0:
        result = _rotate_expand_white(result, rotate)

    if crop:
        result = _crop_normalized(result, crop)

    return result


def _rotate_expand_white(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate clockwise by `degrees`, expanding the canvas and filling
    the newly exposed corners with white.

    OpenCV's warpAffine rotates counter-clockwise for positive angles, so we
    negate the angle to make positive `degrees` rotate clockwise, matching the
    contract.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Negative angle → clockwise rotation (OpenCV uses CCW-positive).
    M = cv2.getRotationMatrix2D(center, -degrees, 1.0)

    # Compute the bounding box of the rotated image so nothing is clipped.
    # Round (not ceil) so axis-aligned rotations (0/90/180/270) land on exact
    # dimensions: at 90 degrees cos is a tiny float residual, not a true 0, and
    # a bare ceil would inflate the canvas by one pixel. A small epsilon biases
    # genuine fractional spans upward so no real content is clipped.
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(np.floor(h * sin + w * cos + 0.5 + 1e-6))
    new_h = int(np.floor(h * cos + w * sin + 0.5 + 1e-6))

    # Shift the transform so the rotated image is centred in the larger canvas.
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(_WHITE, _WHITE, _WHITE),
    )

    # LANCZOS can overshoot slightly outside [0, 1]; keep the pipeline's
    # contract intact.
    return np.clip(rotated, 0.0, 1.0).astype(np.float32)


def _crop_normalized(image: np.ndarray, crop: dict) -> np.ndarray:
    """Crop using normalised fractions (0..1) of the current image size.

    Values are clamped so the crop always stays within bounds and yields a
    non-empty region.
    """
    h, w = image.shape[:2]

    left = _clamp01(crop.get("left"))
    top = _clamp01(crop.get("top"))
    width = _clamp01(crop.get("width"), default=1.0)
    height = _clamp01(crop.get("height"), default=1.0)

    # Convert fractions to pixel bounds.
    x0 = int(round(left * w))
    y0 = int(round(top * h))
    x1 = int(round((left + width) * w))
    y1 = int(round((top + height) * h))

    # Clamp to image bounds and guarantee at least one pixel in each axis.
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))

    # Slicing returns a view; copy so the result never aliases the input.
    return image[y0:y1, x0:x1].copy()


def _clamp01(value, default: float = 0.0) -> float:
    """Coerce `value` to a float in [0, 1]; fall back to `default` when None."""
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, v))
