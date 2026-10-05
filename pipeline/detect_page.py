"""
Stage 3 — Paper / page detection.

Finds the quadrilateral bounding the physical paper in the photograph
using Canny edge detection + contour approximation.

Returns the four corner points in the original (full-resolution) image
coordinate space, plus a boolean indicating whether detection succeeded.
Falls back gracefully to the full image bounds if no quad is found.
"""

from __future__ import annotations

import cv2
import numpy as np


# Working resolution for detection (longer edge).  Detection is done on a
# downscaled copy; the warp is applied to the full-resolution original.
_DETECT_SIZE = 1024


def detect_page_quad(
    image: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """
    Detect the paper quadrilateral in a float32 RGB image.

    Parameters
    ----------
    image:
        float32 RGB array (H, W, 3), values in [0, 1].

    Returns
    -------
    (quad, detected) where:
        quad     — float32 array (4, 2) of corner points in ORIGINAL pixel space,
                   ordered: top-left, top-right, bottom-right, bottom-left.
        detected — True if a quadrilateral was found; False if using fallback.
    """
    h, w = image.shape[:2]
    scale = _DETECT_SIZE / max(h, w)
    small_h, small_w = int(h * scale), int(w * scale)

    # Work on an 8-bit grayscale downscale
    small = cv2.resize(
        (image * 255).astype(np.uint8),
        (small_w, small_h),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    quad_small = _find_quad(gray, small_h, small_w)

    if quad_small is None:
        # Fallback: use full image corners
        return _full_image_quad(h, w), False

    # Scale detected corners back to original image space
    quad_full = quad_small / scale
    return quad_full.astype(np.float32), True


def _find_quad(
    gray: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray | None:
    """
    Try several Canny threshold pairs and return the first valid quad found.
    Returns None if detection fails at all thresholds.
    """
    # Try progressively relaxed thresholds to handle varying image contrast
    threshold_pairs = [
        (50, 150),
        (30, 100),
        (10, 50),
    ]

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    for low, high in threshold_pairs:
        edges = cv2.Canny(blurred, low, high)
        # Dilate to close small gaps in paper edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        # Sort contours by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        image_area = h * w
        for cnt in contours[:5]:  # inspect only the 5 largest
            area = cv2.contourArea(cnt)
            if area < 0.10 * image_area:
                # Too small to be the paper
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                return _order_points(pts)

    return None


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four corner points as: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left:     smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right: largest sum
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]  # top-right:    smallest diff
    rect[3] = pts[np.argmax(diff)]  # bottom-left:  largest diff
    return rect


def _full_image_quad(h: int, w: int) -> np.ndarray:
    """Return the four corners of the full image as a fallback quad."""
    return np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )


def draw_debug_quad(
    image: np.ndarray,
    quad: np.ndarray,
    detected: bool,
) -> np.ndarray:
    """
    Return a copy of the image (uint8 RGB) with the detected quad drawn on it.
    Green = detected quad; red = fallback full-image bounds.
    Useful for visual debugging via the CLI --debug flag.
    """
    vis = (image * 255).clip(0, 255).astype(np.uint8).copy()
    color = (0, 200, 0) if detected else (200, 0, 0)
    pts = quad.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=3)
    for pt in quad:
        cv2.circle(vis, tuple(pt.astype(int)), 8, color, -1)
    return vis
