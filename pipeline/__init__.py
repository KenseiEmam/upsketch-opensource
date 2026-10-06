"""
SketchLift image-processing pipeline.

Entry point: run_pipeline(source_path, sensitivity=0.85) -> PipelineResult
Each stage is independently importable and testable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .decode import decode_image
from .orient import normalize_orientation
from .transform import apply_transform
from .detect_page import detect_page_quad
from .deskew import correct_perspective
from .normalize_illumination import normalize_illumination
from .extract_alpha import extract_alpha_mask
from .recover_color import recover_color
from .reconstruct import reconstruct_rgba


@dataclass
class PipelineResult:
    """All outputs produced by run_pipeline()."""

    rgba: np.ndarray             # uint8 (H, W, 4) — final transparent sketch
    alpha: np.ndarray            # float32 (H, W)  — continuous alpha mask [0,1]
    corrected_rgb: np.ndarray    # uint8 (H, W, 3) — perspective-corrected source
    original_rgb: np.ndarray     # uint8 (H, W, 3) — oriented but not warped
    metadata: dict = field(default_factory=dict)
    # metadata keys:
    #   source_path: str
    #   original_size: (h, w)
    #   output_size: (h, w)
    #   dpi: (x, y)
    #   perspective_corrected: bool
    #   transform_applied: bool
    #   transform_rotate: float   — effective clockwise degrees (0 if none)
    #   transform_crop: dict|None — effective normalized crop (None if none)
    #   sensitivity: float
    #   timings: dict[stage_name, seconds]


def run_pipeline(
    source_path: str | Path,
    sensitivity: float = 0.85,
    illumination_kernel: int = 80,
    *,
    perspective: bool = True,
    illumination_method: str = "morphological",
    alpha_method: str = "soft_threshold",
    transform: dict | None = None,
) -> PipelineResult:
    """
    Run the full extraction pipeline on a single image file.

    Parameters
    ----------
    source_path:
        Path to a JPEG or PNG file.
    sensitivity:
        Alpha extraction sensitivity in [0, 1].  Higher = more strokes kept,
        more paper noise.  Lower = cleaner background, risk of losing faint lines.
    illumination_kernel:
        Morphological kernel size (px) used for background estimation.
        Increase for images with large shaded areas (>=120) to avoid erasing fills.
    perspective:
        When True (default) detect the paper quad and deskew. When False, skip
        page detection + perspective correction entirely and process the image
        as-is (useful when the page already fills the frame, or when detection
        would wrongly crop/warp the image).
    illumination_method:
        One of 'morphological', 'gaussian', 'clahe'. See normalize_illumination.
    alpha_method:
        One of 'soft_threshold', 'bilateral', 'dog', 'clahe_inverted'.
        See extract_alpha_mask.
    transform:
        Optional geometric reframing applied to the ORIGINAL (oriented) image
        before any detection/deskew runs. A dict of the form
        ``{"rotate": <deg clockwise>, "crop": {"left","top","width","height"}}``
        where crop values are fractions (0..1) of the rotated image. See
        pipeline.transform.apply_transform. A no-op when absent/empty.

    Returns
    -------
    PipelineResult with RGBA output, float32 alpha mask, and stage timings.
    """
    source_path = Path(source_path)
    timings: dict[str, float] = {}

    def _time(name: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[name] = round(time.perf_counter() - t0, 3)
        return result

    # Stage 1 — decode
    raw, dpi = _time("decode", decode_image, source_path)

    # Stage 2 — orientation
    oriented = _time("orient", normalize_orientation, raw)

    # Stage 2b — optional manual geometric transform (rotate + crop).
    # Earliest step that can change pixels: applied to the ORIGINAL (oriented)
    # image before automatic page detection / deskew, so manual framing comes
    # first and automatic deskew then operates on the reframed image.
    # apply_transform returns the input unchanged on a no-op, so identity of
    # the returned array tells us whether anything was actually applied.
    before_transform = oriented
    oriented = _time("transform", apply_transform, oriented, transform)
    transform_applied = oriented is not before_transform
    effective_rotate = (
        float((transform or {}).get("rotate") or 0.0) if transform_applied else 0.0
    )
    effective_crop = (transform or {}).get("crop") if transform_applied else None

    # The "before" preview must reflect what was actually processed, so it is
    # computed from the (possibly) transformed image.
    original_rgb = (oriented * 255).clip(0, 255).astype(np.uint8)

    # Stages 3 & 4 — page detection + perspective correction (optional)
    if perspective:
        quad, perspective_corrected = _time("detect_page", detect_page_quad, oriented)
        corrected = _time("deskew", correct_perspective, oriented, quad)
    else:
        # Skip detection/deskew; process the oriented image as-is.
        perspective_corrected = False
        corrected = oriented

    # Stage 5 — illumination normalisation
    normalized = _time(
        "normalize_illumination",
        normalize_illumination,
        corrected,
        kernel_size=illumination_kernel,
        method=illumination_method,
    )

    # Stage 6 — alpha mask extraction
    alpha = _time(
        "extract_alpha",
        extract_alpha_mask,
        normalized,
        sensitivity,
        method=alpha_method,
    )

    # Stage 7 — color recovery
    colored = _time("recover_color", recover_color, corrected, alpha)

    # Stage 9 — RGBA reconstruction  (Stage 8, ML refinement, is deferred)
    rgba = _time("reconstruct", reconstruct_rgba, colored, alpha)

    corrected_uint8 = (corrected * 255).clip(0, 255).astype(np.uint8)

    metadata = {
        "source_path": str(source_path),
        "original_size": oriented.shape[:2],
        "output_size": rgba.shape[:2],
        "dpi": dpi,
        "perspective_requested": perspective,
        "perspective_corrected": perspective_corrected,
        "transform_applied": transform_applied,
        "transform_rotate": effective_rotate,
        "transform_crop": effective_crop,
        "sensitivity": sensitivity,
        "illumination_kernel": illumination_kernel,
        "illumination_method": illumination_method,
        "alpha_method": alpha_method,
        "timings": timings,
        "total_seconds": round(sum(timings.values()), 3),
    }

    return PipelineResult(
        rgba=rgba,
        alpha=alpha,
        corrected_rgb=corrected_uint8,
        original_rgb=original_rgb,
        metadata=metadata,
    )
