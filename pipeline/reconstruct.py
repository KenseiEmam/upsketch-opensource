"""
Stage 9 — RGBA reconstruction.

Composes the float32 RGB colour image and the float32 alpha mask into
a final uint8 RGBA array ready for export.

Stage 8 (optional ML refinement) is intentionally skipped in MVP.
"""

from __future__ import annotations

import numpy as np


def reconstruct_rgba(
    rgb: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Combine RGB colour and alpha mask into a uint8 RGBA array.

    Parameters
    ----------
    rgb:
        float32 (H, W, 3), values in [0, 1].
    alpha:
        float32 (H, W), values in [0, 1].

    Returns
    -------
    uint8 (H, W, 4) RGBA array with straight (unassociated) alpha.
    """
    assert rgb.shape[:2] == alpha.shape, (
        f"RGB shape {rgb.shape[:2]} does not match alpha shape {alpha.shape}"
    )

    h, w = alpha.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = (rgb * 255).clip(0, 255).astype(np.uint8)
    rgba[:, :, 3]  = (alpha * 255).clip(0, 255).astype(np.uint8)
    return rgba
