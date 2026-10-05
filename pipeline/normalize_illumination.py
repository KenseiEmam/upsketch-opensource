"""
Stage 5 — Illumination normalisation.

Removes uneven lighting (desk lamp hotspot, window glare, gradients) while
preserving pencil darkness variation.

This is the most technically critical stage:
  - Too aggressive → faint strokes erased (mistaken for paper variation)
  - Too conservative → paper shadow regions appear opaque in alpha

Baseline algorithm: morphological closing (background estimation) + divide.

The kernel_size parameter must be larger than any contiguous pencil region
to avoid erasing shading fills.  Default 80 px; increase to 120+ for
images with large dark shaded areas.

Three candidate algorithms are exposed for benchmarking:
  normalize_morphological  ← selected baseline
  normalize_gaussian
  normalize_clahe          ← does not remove gradient; useful as comparison
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_illumination(
    image: np.ndarray,
    kernel_size: int = 80,
    method: str = "morphological",
) -> np.ndarray:
    """
    Estimate and remove the paper background illumination.

    Parameters
    ----------
    image:
        float32 RGB array (H, W, 3), values in [0, 1].
    kernel_size:
        Morphological kernel diameter in pixels.  Must be larger than the
        largest connected pencil region to avoid destroying shading.
    method:
        One of 'morphological' (default), 'gaussian', 'clahe'.

    Returns
    -------
    float32 grayscale array (H, W), values in [0, 1].
    1.0 = paper (bright), 0.0 = dark stroke.
    """
    gray = _to_gray_uint8(image)

    if method == "morphological":
        return normalize_morphological(gray, kernel_size)
    elif method == "gaussian":
        return normalize_gaussian(gray, kernel_size)
    elif method == "clahe":
        return normalize_clahe(gray)
    else:
        raise ValueError(f"Unknown normalisation method: {method!r}")


# ---------------------------------------------------------------------------
# Candidate 1 — morphological closing background estimation (selected)
# ---------------------------------------------------------------------------

def normalize_morphological(gray: np.ndarray, kernel_size: int = 80) -> np.ndarray:
    """
    Estimate background via morphological closing, then divide.

    Morphological closing (dilate then erode) fills in dark pencil marks,
    leaving an estimate of what the paper would look like without any strokes.
    Dividing the original by this estimate flattens the illumination gradient.

    Chosen over the Gaussian approach because closing is more robust when
    the paper has a non-uniform reflectance (e.g., near a shadow boundary).
    """
    # kernel_size is a diameter in pixels; must be at least 3
    ksize = max(3, kernel_size)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ksize, ksize)
    )
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Divide: bright paper / bright paper ≈ 1.0; dark stroke / bright paper < 1.0
    normalized = gray.astype(np.float32) / (background.astype(np.float32) + 1e-6)
    return np.clip(normalized, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Candidate 2 — divide by Gaussian blur
# ---------------------------------------------------------------------------

def normalize_gaussian(gray: np.ndarray, blur_radius: int = 80) -> np.ndarray:
    """
    Estimate background by heavily blurring, then divide.

    Faster than morphological, but less accurate when the illumination
    gradient changes sharply (e.g., a harsh shadow line).
    """
    # GaussianBlur kernel must be odd and positive
    ksize = max(3, (blur_radius * 2 + 1) | 1)
    background = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    normalized = gray.astype(np.float32) / (background.astype(np.float32) + 1e-6)
    return np.clip(normalized, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Candidate 3 — CLAHE (contrast limited adaptive histogram equalisation)
# ---------------------------------------------------------------------------

def normalize_clahe(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Apply CLAHE.  Enhances local contrast but does NOT remove large-scale
    illumination gradients.  Include in benchmarks as a baseline comparison.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    return eq.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [0,1] to uint8 grayscale."""
    rgb_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
