"""
Stage 4 — Perspective correction (deskew).

Maps the detected paper quadrilateral to a flat rectangle using a
homography computed with cv2.getPerspectiveTransform.

IMPORTANT: detection is done on a downscaled image, but the warp is
always applied to the full-resolution original to preserve image quality.
"""

from __future__ import annotations

import cv2
import numpy as np


def correct_perspective(
    image: np.ndarray,
    quad: np.ndarray,
) -> np.ndarray:
    """
    Warp the paper quad to a flat rectangle.

    Parameters
    ----------
    image:
        float32 RGB array (H, W, 3) — full-resolution oriented image.
    quad:
        float32 array (4, 2) of corner points in (x, y) image coordinates,
        ordered: top-left, top-right, bottom-right, bottom-left.
        Produced by detect_page_quad().

    Returns
    -------
    float32 RGB array of the warped (cropped/rectified) image.
    """
    tl, tr, br, bl = quad

    # Compute the output rectangle dimensions from the quad edge lengths
    # so the aspect ratio is preserved.
    #
    # Edge length is the distance between corner *pixel coordinates*, e.g. a
    # full-width edge spans x=0..w-1, giving length w-1. The output needs w
    # pixels to cover that span, so we add 1 (and round) to avoid an
    # off-by-one that would shrink the image by a pixel on each axis.
    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    dst_w = int(round(max(width_top, width_bottom))) + 1

    height_left  = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    dst_h = int(round(max(height_left, height_right))) + 1

    # Destination rectangle corners (top-left origin)
    dst = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(quad, dst)

    # Convert to uint8 for warpPerspective (avoids float precision issues)
    img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    warped = cv2.warpPerspective(
        img_uint8,
        M,
        (dst_w, dst_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return warped.astype(np.float32) / 255.0
