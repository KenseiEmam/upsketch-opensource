"""
Image-processing regression tests.

These tests run the full pipeline against the fixture corpus and assert
quality thresholds (doc 07 §5.3).  They are skipped automatically when
no fixture images are present — add real sketches to run them.

Directory layout expected:
    tests/fixtures/images/          *.jpg / *.png  — source images
    tests/fixtures/ground-truth/    <stem>.alpha.png — hand-labelled masks

To add a fixture:
1. Copy a representative sketch photograph to tests/fixtures/images/
2. Create a hand-labelled grayscale alpha mask in tests/fixtures/ground-truth/
   named <image_stem>.alpha.png  (16-bit or 8-bit grayscale PNG)
3. Re-run: pytest tests/test_pipeline_regression.py -v

Thresholds (from doc 07 §5.3):
    stroke_recall        >= 0.90  (hard failure — missing strokes unacceptable)
    false_positive_rate  <= 0.10
    mae                  <= 0.15
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.utils import load_alpha_ground_truth, evaluate_alpha, print_metrics

FIXTURES_DIR    = Path(__file__).parent / "fixtures"
IMAGES_DIR      = FIXTURES_DIR / "images"
GROUND_TRUTH_DIR = FIXTURES_DIR / "ground-truth"

EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Collect fixture image paths
_fixture_images = sorted(
    p for p in IMAGES_DIR.glob("*") if p.suffix.lower() in EXTENSIONS
) if IMAGES_DIR.exists() else []


# ---------------------------------------------------------------------------
# Regression test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    len(_fixture_images) == 0,
    reason=(
        "No fixture images found in tests/fixtures/images/. "
        "Add representative sketch photographs to run regression tests."
    ),
)
@pytest.mark.parametrize("image_path", _fixture_images, ids=lambda p: p.stem)
def test_pipeline_regression(image_path: Path):
    """
    Full pipeline must meet quality thresholds on every fixture image
    that has a corresponding ground-truth alpha mask.

    If no ground-truth mask exists for an image, the test is skipped
    (the image can still be used for visual inspection via the CLI/UI).
    """
    from pipeline import run_pipeline

    gt_path = GROUND_TRUTH_DIR / f"{image_path.stem}.alpha.png"

    if not gt_path.exists():
        pytest.skip(f"No ground-truth mask for {image_path.name} — visual inspection only")

    result = run_pipeline(str(image_path), sensitivity=0.85)
    gt     = load_alpha_ground_truth(gt_path)

    # Resize prediction to match ground-truth dimensions if needed
    # (ground truth may have been created at a different resolution)
    pred = result.alpha
    if pred.shape != gt.shape:
        from PIL import Image as PilImage
        pred_img = PilImage.fromarray((pred * 255).astype(np.uint8), mode="L")
        pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), PilImage.LANCZOS)
        pred = np.array(pred_img, dtype=np.float32) / 255.0

    metrics = evaluate_alpha(pred, gt)
    print_metrics(image_path.name, metrics)

    assert metrics["stroke_recall"] >= 0.90, (
        f"{image_path.name}: stroke_recall={metrics['stroke_recall']:.3f} < 0.90 "
        f"(missing strokes are a hard failure)"
    )
    assert metrics["false_positive_rate"] <= 0.10, (
        f"{image_path.name}: false_positive_rate={metrics['false_positive_rate']:.3f} > 0.10"
    )
    assert metrics["mae"] <= 0.15, (
        f"{image_path.name}: mae={metrics['mae']:.3f} > 0.15"
    )


# ---------------------------------------------------------------------------
# Smoke test: pipeline runs on every fixture image (no ground truth needed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    len(_fixture_images) == 0,
    reason="No fixture images found in tests/fixtures/images/.",
)
@pytest.mark.parametrize("image_path", _fixture_images, ids=lambda p: p.stem)
def test_pipeline_does_not_crash(image_path: Path):
    """
    The pipeline must complete without raising an exception on every
    fixture image, even if no ground-truth mask exists.
    This catches hard failures (corrupt output, shape mismatches, etc.).
    """
    from pipeline import run_pipeline
    result = run_pipeline(str(image_path), sensitivity=0.85)
    assert result.rgba.shape[2] == 4
    assert result.rgba.dtype.kind == "u"
    assert 0.0 <= result.alpha.min() and result.alpha.max() <= 1.0
    assert result.metadata["total_seconds"] >= 0


# ---------------------------------------------------------------------------
# Benchmark runner (not a pytest test — invoke directly)
# ---------------------------------------------------------------------------

def run_benchmark(sensitivity: float = 0.85) -> None:
    """
    Compare all four alpha-extraction methods across the fixture corpus.
    Run directly: python -c "from tests.test_pipeline_regression import run_benchmark; run_benchmark()"
    """
    from pipeline.decode import decode_image
    from pipeline.orient import normalize_orientation
    from pipeline.detect_page import detect_page_quad
    from pipeline.deskew import correct_perspective
    from pipeline.normalize_illumination import normalize_illumination
    from pipeline.extract_alpha import (
        extract_soft_threshold, extract_bilateral,
        extract_dog, extract_clahe_inverted,
    )

    methods = {
        "soft_threshold":  extract_soft_threshold,
        "bilateral":       extract_bilateral,
        "dog":             extract_dog,
        "clahe_inverted":  extract_clahe_inverted,
    }

    images = sorted(
        p for p in IMAGES_DIR.glob("*") if p.suffix.lower() in EXTENSIONS
    ) if IMAGES_DIR.exists() else []

    if not images:
        print("No fixture images found. Add images to tests/fixtures/images/")
        return

    for img_path in images:
        gt_path = GROUND_TRUTH_DIR / f"{img_path.stem}.alpha.png"
        if not gt_path.exists():
            print(f"\n{img_path.name}  (no ground truth — skipping metrics)")
            continue

        raw, _ = decode_image(img_path)
        oriented  = normalize_orientation(raw)
        quad, _   = detect_page_quad(oriented)
        corrected = correct_perspective(oriented, quad)
        norm      = normalize_illumination(corrected)
        gt        = load_alpha_ground_truth(gt_path)

        print(f"\n{img_path.name}")
        for name, fn in methods.items():
            pred    = fn(norm, sensitivity=sensitivity)
            metrics = evaluate_alpha(pred, gt)
            print_metrics(f"  {name}", metrics)
