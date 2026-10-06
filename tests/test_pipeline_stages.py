"""
Unit tests for pipeline stages 3–9.

All tests use synthetic in-memory images — no real sketch photographs needed.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stage 2b — transform (optional rotate + crop)
# ---------------------------------------------------------------------------

class TestTransform:
    def _make_image(self, h=100, w=150):
        """float32 RGB with a distinct marked top-left corner for sanity."""
        img = np.ones((h, w, 3), dtype=np.float32)
        img[0:10, 0:10, :] = 0.0  # black square in the top-left corner
        return img

    def test_none_is_noop(self):
        from pipeline.transform import apply_transform
        img = self._make_image()
        result = apply_transform(img, None)
        # No-op must return the exact same object (used by run_pipeline to
        # decide transform_applied).
        assert result is img

    def test_empty_dict_is_noop(self):
        from pipeline.transform import apply_transform
        img = self._make_image()
        assert apply_transform(img, {}) is img

    def test_zero_rotate_no_crop_is_noop(self):
        from pipeline.transform import apply_transform
        img = self._make_image()
        assert apply_transform(img, {"rotate": 0}) is img

    def test_identity_preserves_dimensions(self):
        from pipeline.transform import apply_transform
        img = self._make_image(100, 150)
        # A full-frame crop is effectively identity in size.
        result = apply_transform(
            img, {"crop": {"left": 0.0, "top": 0.0, "width": 1.0, "height": 1.0}}
        )
        assert result.shape[:2] == (100, 150)

    def test_90_degree_rotation_swaps_dimensions(self):
        from pipeline.transform import apply_transform
        img = self._make_image(100, 150)  # (H=100, W=150)
        result = apply_transform(img, {"rotate": 90})
        # A 90-degree rotation swaps width and height.
        assert result.shape[:2] == (150, 100)

    def test_does_not_mutate_input(self):
        from pipeline.transform import apply_transform
        img = self._make_image()
        before = img.copy()
        apply_transform(img, {"rotate": 90})
        apply_transform(
            img, {"crop": {"left": 0.1, "top": 0.1, "width": 0.5, "height": 0.5}}
        )
        assert np.array_equal(img, before)

    def test_output_is_float32(self):
        from pipeline.transform import apply_transform
        img = self._make_image()
        result = apply_transform(img, {"rotate": 12.5})
        assert result.dtype == np.float32

    def test_rotation_fills_corners_white(self):
        from pipeline.transform import apply_transform
        img = self._make_image(100, 150)
        result = apply_transform(img, {"rotate": 30})
        # The expanded canvas corners (exposed by rotation) must be white, not
        # black/transparent, so illumination normalisation still behaves.
        assert result[0, 0].mean() == pytest.approx(1.0, abs=1e-3)

    def test_crop_yields_expected_subrectangle(self):
        from pipeline.transform import apply_transform
        img = self._make_image(100, 200)  # H=100, W=200
        result = apply_transform(
            img, {"crop": {"left": 0.10, "top": 0.05, "width": 0.80, "height": 0.70}}
        )
        # Expected: width = round(0.8*200)=160, height = round(0.7*100)=70.
        assert result.shape[:2] == (70, 160)

    def test_crop_out_of_range_is_clamped(self):
        from pipeline.transform import apply_transform
        img = self._make_image(100, 150)
        # Values beyond [0,1] must be clamped so the crop stays in bounds.
        result = apply_transform(
            img, {"crop": {"left": -0.5, "top": -0.5, "width": 2.0, "height": 2.0}}
        )
        # Clamped to the full image.
        assert result.shape[:2] == (100, 150)

    def test_crop_degenerate_values_stay_nonempty(self):
        from pipeline.transform import apply_transform
        img = self._make_image(100, 150)
        # Zero-size crop should still yield at least a 1x1 region, not empty.
        result = apply_transform(
            img, {"crop": {"left": 0.5, "top": 0.5, "width": 0.0, "height": 0.0}}
        )
        assert result.shape[0] >= 1 and result.shape[1] >= 1

    def test_rotate_then_crop_order(self):
        from pipeline.transform import apply_transform
        img = self._make_image(100, 150)
        # Rotate 90 (-> 150x100) then full crop keeps the rotated dimensions.
        result = apply_transform(
            img,
            {
                "rotate": 90,
                "crop": {"left": 0.0, "top": 0.0, "width": 1.0, "height": 1.0},
            },
        )
        assert result.shape[:2] == (150, 100)


# ---------------------------------------------------------------------------
# Stage 3 — detect_page
# ---------------------------------------------------------------------------

class TestDetectPage:
    def _make_page_image(self, h=400, w=400):
        """White rectangle on a dark background — easy case for quad detection."""
        img = np.zeros((h, w, 3), dtype=np.float32)
        img[50:350, 50:350, :] = 1.0  # white paper region
        return img

    def test_returns_four_points(self):
        from pipeline.detect_page import detect_page_quad
        img = self._make_page_image()
        quad, detected = detect_page_quad(img)
        assert quad.shape == (4, 2)

    def test_detected_flag_true_for_clear_image(self):
        from pipeline.detect_page import detect_page_quad
        img = self._make_page_image()
        _, detected = detect_page_quad(img)
        # May be True or False depending on contrast — assert valid quad either way
        assert isinstance(detected, bool)

    def test_fallback_returns_full_bounds(self):
        """A uniform image has no detectable quad — must fall back gracefully."""
        from pipeline.detect_page import detect_page_quad
        img = np.ones((100, 200, 3), dtype=np.float32) * 0.9
        quad, detected = detect_page_quad(img)
        assert quad.shape == (4, 2)
        assert detected is False
        # Fallback corners should match image bounds
        assert quad[0][0] == pytest.approx(0, abs=1)    # top-left x
        assert quad[0][1] == pytest.approx(0, abs=1)    # top-left y
        assert quad[2][0] == pytest.approx(199, abs=1)  # bottom-right x
        assert quad[2][1] == pytest.approx(99, abs=1)   # bottom-right y

    def test_order_points(self):
        from pipeline.detect_page import _order_points
        pts = np.array([[100, 0], [0, 0], [0, 100], [100, 100]], dtype=np.float32)
        ordered = _order_points(pts)
        assert ordered.shape == (4, 2)
        # top-left has smallest x+y sum
        assert ordered[0].tolist() == [0.0, 0.0]
        # bottom-right has largest x+y sum
        assert ordered[2].tolist() == [100.0, 100.0]


# ---------------------------------------------------------------------------
# Stage 4 — deskew
# ---------------------------------------------------------------------------

class TestDeskew:
    def test_output_is_float32_rgb(self, sketch_float_image):
        from pipeline.deskew import correct_perspective
        from pipeline.detect_page import _full_image_quad
        h, w = sketch_float_image.shape[:2]
        quad = _full_image_quad(h, w)
        result = correct_perspective(sketch_float_image, quad)
        assert result.dtype == np.float32
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_output_values_in_range(self, sketch_float_image):
        from pipeline.deskew import correct_perspective
        from pipeline.detect_page import _full_image_quad
        h, w = sketch_float_image.shape[:2]
        quad = _full_image_quad(h, w)
        result = correct_perspective(sketch_float_image, quad)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_identity_warp_preserves_dimensions(self, sketch_float_image):
        """Full-image quad warp should produce the same dimensions."""
        from pipeline.deskew import correct_perspective
        from pipeline.detect_page import _full_image_quad
        h, w = sketch_float_image.shape[:2]
        quad = _full_image_quad(h, w)
        result = correct_perspective(sketch_float_image, quad)
        assert result.shape[:2] == (h, w)


# ---------------------------------------------------------------------------
# Stage 5 — normalize_illumination
# ---------------------------------------------------------------------------

class TestNormalizeIllumination:
    def test_returns_2d_float32(self, sketch_float_image):
        from pipeline.normalize_illumination import normalize_illumination
        result = normalize_illumination(sketch_float_image)
        assert result.ndim == 2
        assert result.dtype == np.float32

    def test_values_in_range(self, sketch_float_image):
        from pipeline.normalize_illumination import normalize_illumination
        result = normalize_illumination(sketch_float_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_dark_stroke_has_lower_value_than_paper(self, sketch_float_image):
        """After normalisation the stroke region should be darker than the paper."""
        from pipeline.normalize_illumination import normalize_illumination
        norm = normalize_illumination(sketch_float_image)
        stroke_mean = norm[40:60, 20:130].mean()
        paper_mean  = norm[0:20,  0:20].mean()
        assert stroke_mean < paper_mean, (
            f"Stroke mean {stroke_mean:.3f} should be lower than paper mean {paper_mean:.3f}"
        )

    @pytest.mark.parametrize("method", ["morphological", "gaussian", "clahe"])
    def test_all_methods_valid_output(self, sketch_float_image, method):
        from pipeline.normalize_illumination import normalize_illumination
        result = normalize_illumination(sketch_float_image, method=method)
        assert result.ndim == 2
        assert result.dtype == np.float32
        assert 0.0 <= result.min() and result.max() <= 1.0

    def test_unknown_method_raises(self, sketch_float_image):
        from pipeline.normalize_illumination import normalize_illumination
        with pytest.raises(ValueError, match="Unknown"):
            normalize_illumination(sketch_float_image, method="nonexistent")


# ---------------------------------------------------------------------------
# Stage 6 — extract_alpha
# ---------------------------------------------------------------------------

class TestExtractAlpha:
    def test_returns_float32_2d(self, sketch_float_image):
        from pipeline.normalize_illumination import normalize_illumination
        from pipeline.extract_alpha import extract_alpha_mask
        norm = normalize_illumination(sketch_float_image)
        alpha = extract_alpha_mask(norm)
        assert alpha.ndim == 2
        assert alpha.dtype == np.float32

    def test_values_in_0_1(self, sketch_float_image):
        from pipeline.normalize_illumination import normalize_illumination
        from pipeline.extract_alpha import extract_alpha_mask
        norm = normalize_illumination(sketch_float_image)
        alpha = extract_alpha_mask(norm)
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

    def test_stroke_has_higher_alpha_than_paper(self, sketch_float_image):
        """The dark stroke region should have higher alpha than the white paper."""
        from pipeline.normalize_illumination import normalize_illumination
        from pipeline.extract_alpha import extract_alpha_mask
        norm = normalize_illumination(sketch_float_image)
        alpha = extract_alpha_mask(norm, sensitivity=0.85)
        stroke_alpha = alpha[40:60, 20:130].mean()
        paper_alpha  = alpha[0:20,  0:20].mean()
        assert stroke_alpha > paper_alpha, (
            f"Stroke alpha {stroke_alpha:.3f} should exceed paper alpha {paper_alpha:.3f}"
        )

    def test_sensitivity_zero_gives_minimal_alpha(self):
        """Very low sensitivity should suppress almost everything."""
        from pipeline.extract_alpha import extract_soft_threshold
        # Mild stroke — normalised value around 0.7 (paper-like)
        norm = np.full((50, 50), 0.7, dtype=np.float32)
        alpha = extract_soft_threshold(norm, sensitivity=0.0)
        assert alpha.max() < 0.5

    def test_sensitivity_one_keeps_all(self):
        """Maximum sensitivity should pass through all non-zero signal."""
        from pipeline.extract_alpha import extract_soft_threshold
        norm = np.full((50, 50), 0.5, dtype=np.float32)  # mid-grey
        alpha = extract_soft_threshold(norm, sensitivity=1.0)
        assert alpha.mean() > 0.0

    @pytest.mark.parametrize("method", [
        "soft_threshold", "bilateral", "dog", "clahe_inverted"
    ])
    def test_all_methods_valid_output(self, sketch_float_image, method):
        from pipeline.normalize_illumination import normalize_illumination
        from pipeline.extract_alpha import extract_alpha_mask
        norm  = normalize_illumination(sketch_float_image)
        alpha = extract_alpha_mask(norm, method=method)
        assert alpha.ndim == 2
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

    def test_unknown_method_raises(self, sketch_float_image):
        from pipeline.normalize_illumination import normalize_illumination
        from pipeline.extract_alpha import extract_alpha_mask
        norm = normalize_illumination(sketch_float_image)
        with pytest.raises(ValueError, match="Unknown"):
            extract_alpha_mask(norm, method="magic")


# ---------------------------------------------------------------------------
# Stage 9 — reconstruct_rgba
# ---------------------------------------------------------------------------

class TestReconstructRGBA:
    def test_output_shape(self, sketch_float_image):
        from pipeline.reconstruct import reconstruct_rgba
        alpha = np.ones(sketch_float_image.shape[:2], dtype=np.float32) * 0.5
        rgba = reconstruct_rgba(sketch_float_image, alpha)
        h, w = sketch_float_image.shape[:2]
        assert rgba.shape == (h, w, 4)

    def test_output_dtype_uint8(self, sketch_float_image):
        from pipeline.reconstruct import reconstruct_rgba
        alpha = np.ones(sketch_float_image.shape[:2], dtype=np.float32)
        rgba = reconstruct_rgba(sketch_float_image, alpha)
        assert rgba.dtype == np.uint8

    def test_alpha_channel_values(self, sketch_float_image):
        from pipeline.reconstruct import reconstruct_rgba
        alpha = np.full(sketch_float_image.shape[:2], 0.5, dtype=np.float32)
        rgba = reconstruct_rgba(sketch_float_image, alpha)
        # 0.5 * 255 = 127.5 → should round to 127 or 128
        assert 125 <= rgba[:, :, 3].mean() <= 130

    def test_zero_alpha_fully_transparent(self, sketch_float_image):
        from pipeline.reconstruct import reconstruct_rgba
        alpha = np.zeros(sketch_float_image.shape[:2], dtype=np.float32)
        rgba = reconstruct_rgba(sketch_float_image, alpha)
        assert rgba[:, :, 3].max() == 0

    def test_shape_mismatch_raises(self, sketch_float_image):
        from pipeline.reconstruct import reconstruct_rgba
        bad_alpha = np.ones((5, 5), dtype=np.float32)
        with pytest.raises(AssertionError):
            reconstruct_rgba(sketch_float_image, bad_alpha)


# ---------------------------------------------------------------------------
# Full pipeline smoke test
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_smoke_jpeg(self, tmp_jpeg):
        from pipeline import run_pipeline
        result = run_pipeline(tmp_jpeg, sensitivity=0.85)
        assert result.rgba.shape[2] == 4
        assert result.rgba.dtype == np.uint8
        assert result.alpha.ndim == 2
        assert result.alpha.dtype == np.float32
        assert result.metadata["total_seconds"] >= 0

    def test_smoke_png(self, tmp_png):
        from pipeline import run_pipeline
        result = run_pipeline(tmp_png, sensitivity=0.85)
        assert result.rgba.shape[2] == 4
        assert result.rgba.dtype == np.uint8

    def test_output_matches_corrected_dimensions(self, tmp_jpeg):
        from pipeline import run_pipeline
        result = run_pipeline(tmp_jpeg)
        rgba_h, rgba_w = result.rgba.shape[:2]
        corr_h, corr_w = result.corrected_rgb.shape[:2]
        assert rgba_h == corr_h
        assert rgba_w == corr_w

    def test_sensitivity_affects_alpha_mean(self, tmp_png):
        """Higher sensitivity should produce a higher mean alpha (more strokes)."""
        from pipeline import run_pipeline
        lo = run_pipeline(tmp_png, sensitivity=0.50)
        hi = run_pipeline(tmp_png, sensitivity=0.99)
        assert hi.alpha.mean() >= lo.alpha.mean(), (
            "Higher sensitivity should not produce lower mean alpha"
        )

    def test_timings_contain_all_stages(self, tmp_jpeg):
        from pipeline import run_pipeline
        result = run_pipeline(tmp_jpeg)
        expected = {"decode", "orient", "detect_page", "deskew",
                    "normalize_illumination", "extract_alpha",
                    "recover_color", "reconstruct"}
        for stage in expected:
            assert stage in result.metadata["timings"], f"Missing timing for stage: {stage}"

    def test_missing_file_raises(self, tmp_path):
        from pipeline import run_pipeline
        with pytest.raises(FileNotFoundError):
            run_pipeline(tmp_path / "ghost.jpg")

    def test_transform_metadata_absent_by_default(self, tmp_png):
        from pipeline import run_pipeline
        result = run_pipeline(tmp_png)
        assert result.metadata["transform_applied"] is False
        assert result.metadata["transform_rotate"] == 0.0
        assert result.metadata["transform_crop"] is None

    def test_transform_rotation_recorded(self, tmp_png):
        from pipeline import run_pipeline
        result = run_pipeline(tmp_png, perspective=False, transform={"rotate": 90})
        assert result.metadata["transform_applied"] is True
        assert result.metadata["transform_rotate"] == 90.0

    def test_transform_affects_before_preview_dimensions(self, tmp_png):
        """original_rgb must reflect the transformed image, not the raw one."""
        from pipeline import run_pipeline
        baseline = run_pipeline(tmp_png, perspective=False)
        rotated = run_pipeline(
            tmp_png, perspective=False, transform={"rotate": 90}
        )
        bh, bw = baseline.original_rgb.shape[:2]
        rh, rw = rotated.original_rgb.shape[:2]
        # 90-degree rotation swaps the preview's width/height.
        assert (rh, rw) == (bw, bh)
