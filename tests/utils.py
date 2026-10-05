"""
Shared utilities for the regression test suite.

Used by test_pipeline_regression.py to load ground-truth alpha masks
and compute quality metrics against pipeline output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_alpha_ground_truth(path: str | Path) -> np.ndarray:
    """
    Load a ground-truth alpha mask from a 16-bit grayscale PNG.

    The ground-truth PNGs are 16-bit grayscale images where:
      65535 (white)  = definite pencil stroke  → 1.0
      0     (black)  = clean paper             → 0.0

    Returns a float32 array in [0, 1].

    If the file is 8-bit, it is accepted and normalised to [0, 1].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth mask not found: {path}")

    img = Image.open(str(path))

    # Accept L (8-bit) or I;16 / I (16-bit) grayscale
    if img.mode in ("I", "I;16"):
        arr = np.array(img, dtype=np.float32)
        arr /= 65535.0
    elif img.mode == "L":
        arr = np.array(img, dtype=np.float32) / 255.0
    else:
        # Convert to grayscale then normalise
        arr = np.array(img.convert("L"), dtype=np.float32) / 255.0

    return np.clip(arr, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_alpha(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    stroke_threshold: float = 0.30,
    prediction_threshold: float = 0.20,
    paper_threshold: float = 0.05,
    false_positive_threshold: float = 0.10,
) -> dict[str, float]:
    """
    Compute quality metrics comparing a predicted alpha mask to ground truth.

    Parameters
    ----------
    predicted:
        float32 [0, 1] alpha mask from the pipeline.
    ground_truth:
        float32 [0, 1] hand-labelled alpha mask.
    stroke_threshold:
        GT value above which a pixel is considered "definitely a stroke".
    prediction_threshold:
        Predicted value above which we count a pixel as "detected".
    paper_threshold:
        GT value below which a pixel is considered "definitely paper".
    false_positive_threshold:
        Predicted value above which a paper pixel is a false positive.

    Returns
    -------
    Dict with keys: mae, stroke_recall, false_positive_rate.

    Thresholds used in regression tests (from doc 07):
        stroke_recall        >= 0.90
        false_positive_rate  <= 0.10
        mae                  <= 0.15
    """
    assert predicted.shape == ground_truth.shape, (
        f"Shape mismatch: predicted {predicted.shape} vs gt {ground_truth.shape}"
    )

    # Mean absolute error
    mae = float(np.mean(np.abs(predicted - ground_truth)))

    # Stroke recall: fraction of definite-stroke GT pixels that the
    # pipeline also marks as strokes
    stroke_mask = ground_truth > stroke_threshold
    if stroke_mask.sum() == 0:
        recall = 1.0  # no strokes in GT → trivially perfect recall
    else:
        recall = float(np.mean(predicted[stroke_mask] > prediction_threshold))

    # False positive rate: fraction of definite-paper GT pixels that the
    # pipeline incorrectly marks as strokes
    paper_mask = ground_truth < paper_threshold
    if paper_mask.sum() == 0:
        fpr = 0.0  # no paper in GT → trivially zero FPR
    else:
        fpr = float(np.mean(predicted[paper_mask] > false_positive_threshold))

    return {
        "mae":                mae,
        "stroke_recall":      recall,
        "false_positive_rate": fpr,
    }


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    """Pretty-print a metrics dict to stdout."""
    status_recall = "OK" if metrics["stroke_recall"]      >= 0.90 else "FAIL"
    status_fpr    = "OK" if metrics["false_positive_rate"] <= 0.10 else "FAIL"
    status_mae    = "OK" if metrics["mae"]                 <= 0.15 else "FAIL"

    print(
        f"  {name:<40}  "
        f"recall={metrics['stroke_recall']:.3f} [{status_recall}]  "
        f"fpr={metrics['false_positive_rate']:.3f} [{status_fpr}]  "
        f"mae={metrics['mae']:.3f} [{status_mae}]"
    )
