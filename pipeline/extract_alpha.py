"""
Stage 6 — Alpha mask extraction.

Produces a continuous float32 [0, 1] alpha mask from the normalised
luminance image.

  1.0 = definite pencil stroke
  0.0 = clean paper
  In-between = faint strokes, graphite texture, light shading

BINARY THRESHOLDING IS EXPLICITLY PROHIBITED.
The output must preserve faint strokes and natural line-weight variation.

Four candidate approaches are implemented for benchmarking:
  extract_soft_threshold    ← selected baseline
  extract_bilateral
  extract_dog               (Difference of Gaussians)
  extract_clahe_inverted

The sensitivity parameter (0–1) maps directly to the frontend slider.
  higher sensitivity → more strokes retained (may include paper noise)
  lower  sensitivity → cleaner background (risk of losing faint lines)
"""

from __future__ import annotations

import cv2
import numpy as np


def extract_alpha_mask(
    normalized: np.ndarray,
    sensitivity: float = 0.85,
    method: str = "soft_threshold",
) -> np.ndarray:
    """
    Convert the illumination-normalised image into a float32 alpha mask.

    Parameters
    ----------
    normalized:
        float32 grayscale (H, W), values in [0, 1].
        1.0 = paper bright, 0.0 = dark stroke (output of normalize_illumination).
    sensitivity:
        0–1 float.  Higher = more strokes kept.  Default 0.85.
    method:
        One of 'soft_threshold' (default), 'bilateral', 'dog', 'clahe_inverted'.

    Returns
    -------
    float32 array (H, W), values in [0, 1].
    """
    if method == "soft_threshold":
        return extract_soft_threshold(normalized, sensitivity)
    elif method == "bilateral":
        return extract_bilateral(normalized, sensitivity)
    elif method == "dog":
        return extract_dog(normalized, sensitivity)
    elif method == "clahe_inverted":
        return extract_clahe_inverted(normalized, sensitivity)
    else:
        raise ValueError(f"Unknown alpha extraction method: {method!r}")


# ---------------------------------------------------------------------------
# Candidate 1 — gamma-boosted soft threshold (selected baseline)
# ---------------------------------------------------------------------------

def extract_soft_threshold(
    normalized: np.ndarray,
    sensitivity: float = 0.85,
    gamma: float = 1.5,
) -> np.ndarray:
    """
    Invert the normalised image, boost faint values with a gamma curve,
    then ramp down values below the sensitivity threshold.

    The gamma curve (> 1.0) lifts faint strokes closer to 1.0 before the
    threshold ramp is applied, preserving them against suppression.

    The ramp is a linear rescale from [threshold, 1.0] → [0, 1.0], not a
    hard binary cutoff.  Values just below the threshold fade out smoothly.
    """
    # Invert: paper → 0, strokes → high values
    inverted = 1.0 - normalized

    # Gamma boost: lifts faint strokes above the noise floor
    boosted = np.power(np.clip(inverted, 0.0, 1.0), 1.0 / gamma)

    # Threshold: noise below (1 - sensitivity) is suppressed
    threshold = 1.0 - sensitivity
    alpha = (boosted - threshold) / (1.0 - threshold + 1e-6)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Candidate 2 — bilateral filter pre-smoothing
# ---------------------------------------------------------------------------

def extract_bilateral(
    normalized: np.ndarray,
    sensitivity: float = 0.85,
) -> np.ndarray:
    """
    Apply bilateral filtering before soft-thresholding.

    Bilateral filtering is edge-preserving: it smooths paper grain in
    uniform regions while keeping sharp transitions at stroke edges.
    This reduces false positives from paper texture at the cost of ~3×
    extra processing time.
    """
    # Bilateral filter operates on uint8
    gray_uint8 = (normalized * 255).clip(0, 255).astype(np.uint8)
    smoothed = cv2.bilateralFilter(gray_uint8, d=9, sigmaColor=75, sigmaSpace=75)
    smoothed_f = smoothed.astype(np.float32) / 255.0
    return extract_soft_threshold(smoothed_f, sensitivity)


# ---------------------------------------------------------------------------
# Candidate 3 — Difference of Gaussians (DoG)
# ---------------------------------------------------------------------------

def extract_dog(
    normalized: np.ndarray,
    sensitivity: float = 0.85,
    sigma1: float = 1.0,
    sigma2: float = 2.0,
) -> np.ndarray:
    """
    Use Difference of Gaussians to isolate stroke edges.

    DoG highlights the high-frequency content (edges and fine texture) of
    the inverted image.  Works well for sharp ink lines; may miss soft
    graphite shading fills.  Combine with the soft-threshold result for
    best coverage.
    """
    inverted = 1.0 - normalized

    k1 = _odd_kernel(sigma1 * 3)
    k2 = _odd_kernel(sigma2 * 3)
    blurred1 = cv2.GaussianBlur(inverted, (k1, k1), sigma1)
    blurred2 = cv2.GaussianBlur(inverted, (k2, k2), sigma2)

    dog = blurred1 - blurred2  # may contain negative values

    # Normalise DoG to [0, 1]
    dog_min, dog_max = dog.min(), dog.max()
    if dog_max - dog_min > 1e-6:
        dog = (dog - dog_min) / (dog_max - dog_min)
    else:
        dog = np.zeros_like(dog)

    # Blend DoG edges with the direct inversion to preserve filled areas
    blended = np.maximum(inverted * 0.6, dog * 0.4)
    return extract_soft_threshold(1.0 - blended, sensitivity)


# ---------------------------------------------------------------------------
# Candidate 4 — CLAHE + invert
# ---------------------------------------------------------------------------

def extract_clahe_inverted(
    normalized: np.ndarray,
    sensitivity: float = 0.85,
) -> np.ndarray:
    """
    Apply CLAHE to boost local contrast before soft-thresholding.

    Useful when the illumination normalisation has left some regions with
    low contrast (e.g., a sheet photographed under very flat lighting).
    """
    gray_uint8 = (normalized * 255).clip(0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray_uint8).astype(np.float32) / 255.0
    return extract_soft_threshold(eq, sensitivity)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _odd_kernel(size: float) -> int:
    """Return the nearest odd integer >= size (minimum 3)."""
    k = max(3, int(size))
    return k if k % 2 == 1 else k + 1
