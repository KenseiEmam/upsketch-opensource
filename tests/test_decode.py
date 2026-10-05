"""Tests for pipeline/decode.py — Stage 1."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jpeg(path: Path, h: int = 80, w: int = 100) -> Path:
    img = Image.fromarray(np.full((h, w, 3), 200, dtype=np.uint8), "RGB")
    img.save(str(path), format="JPEG", quality=90)
    return path


def _write_png(path: Path, h: int = 80, w: int = 100) -> Path:
    img = Image.fromarray(np.full((h, w, 3), 180, dtype=np.uint8), "RGB")
    img.save(str(path), format="PNG")
    return path


def _write_png_rgba(path: Path, h: int = 80, w: int = 100) -> Path:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, :3] = 200
    arr[:, :, 3]  = 128
    img = Image.fromarray(arr, "RGBA")
    img.save(str(path), format="PNG")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDecodeImage:
    def test_jpeg_returns_float32_rgb(self, tmp_path):
        from pipeline.decode import decode_image
        p = _write_jpeg(tmp_path / "img.jpg")
        arr, dpi = decode_image(p)

        assert arr.dtype == np.float32
        assert arr.ndim == 3
        assert arr.shape[2] == 3  # RGB
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_png_returns_float32_rgb(self, tmp_path):
        from pipeline.decode import decode_image
        p = _write_png(tmp_path / "img.png")
        arr, dpi = decode_image(p)

        assert arr.dtype == np.float32
        assert arr.shape[2] == 3

    def test_png_rgba_composited_to_rgb(self, tmp_path):
        """RGBA PNG must be composited onto white and returned as 3-channel."""
        from pipeline.decode import decode_image
        p = _write_png_rgba(tmp_path / "img.png")
        arr, _ = decode_image(p)
        assert arr.shape[2] == 3, "Expected RGB (3 channels) after RGBA composite"

    def test_dpi_extracted(self, tmp_path):
        from pipeline.decode import decode_image
        img = Image.fromarray(np.full((50, 50, 3), 200, dtype=np.uint8), "RGB")
        p = tmp_path / "dpi.jpg"
        img.save(str(p), format="JPEG", dpi=(300, 300))
        _, dpi = decode_image(p)
        assert dpi[0] >= 290  # allow small JPEG rounding

    def test_default_dpi_when_missing(self, tmp_path):
        from pipeline.decode import decode_image
        p = _write_jpeg(tmp_path / "nodpi.jpg")
        _, dpi = decode_image(p)
        assert dpi == (72.0, 72.0)

    def test_file_not_found(self, tmp_path):
        from pipeline.decode import decode_image
        with pytest.raises(FileNotFoundError):
            decode_image(tmp_path / "missing.jpg")

    def test_bad_magic_bytes_rejected(self, tmp_path):
        from pipeline.decode import decode_image
        p = tmp_path / "fake.jpg"
        p.write_bytes(b"NOTAIMAGE" + b"\x00" * 100)
        with pytest.raises(ValueError, match="magic"):
            decode_image(p)

    def test_dimensions_preserved(self, tmp_path):
        from pipeline.decode import decode_image
        h, w = 123, 456
        img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), "RGB")
        p = tmp_path / "dims.png"
        img.save(str(p), "PNG")
        arr, _ = decode_image(p)
        assert arr.shape[:2] == (h, w)


class TestVerifyMagic:
    def test_valid_jpeg(self, tmp_path):
        from pipeline.decode import _verify_magic
        p = tmp_path / "v.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)
        _verify_magic(p)  # should not raise

    def test_valid_png(self, tmp_path):
        from pipeline.decode import _verify_magic
        p = tmp_path / "v.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)
        _verify_magic(p)  # should not raise

    def test_invalid_raises(self, tmp_path):
        from pipeline.decode import _verify_magic
        p = tmp_path / "bad.jpg"
        p.write_bytes(b"GIF89a" + b"\x00" * 20)
        with pytest.raises(ValueError):
            _verify_magic(p)
