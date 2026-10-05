"""
Shared pytest fixtures for the SketchLift pipeline test suite.

Synthetic images are generated entirely in-memory so tests run with no
external files and no network access.
"""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Image factories
# ---------------------------------------------------------------------------

def _make_rgb_array(h: int = 100, w: int = 100, color=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Return a float32 RGB array filled with a uniform colour."""
    img = np.ones((h, w, 3), dtype=np.float32)
    img[:, :, 0] = color[0]
    img[:, :, 1] = color[1]
    img[:, :, 2] = color[2]
    return img


def _make_rgba_array(h: int = 100, w: int = 100) -> np.ndarray:
    """Return a uint8 RGBA array with a semi-transparent grey stroke."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, :3] = 200  # light grey background
    arr[:, :, 3]  = 0    # fully transparent

    # Add a visible horizontal stroke with varying alpha
    arr[40:60, 10:90, :3] = 30   # dark pencil colour
    arr[40:60, 10:90, 3]  = 200  # mostly opaque
    arr[48:52, 30:70, 3]  = 255  # fully opaque centre
    return arr


def _pil_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures: temp files
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_jpeg(tmp_path: Path) -> Path:
    """Write a plain white JPEG to a temp file and return its path."""
    img = Image.fromarray(
        (np.ones((200, 300, 3), dtype=np.float32) * 230).astype(np.uint8),
        mode="RGB",
    )
    path = tmp_path / "test_sketch.jpg"
    img.save(str(path), format="JPEG", quality=92)
    return path


@pytest.fixture()
def tmp_png(tmp_path: Path) -> Path:
    """Write a white PNG with a dark stroke to a temp file."""
    arr = np.ones((200, 300, 3), dtype=np.uint8) * 240
    arr[80:120, 50:250, :] = 20  # dark stroke band
    img = Image.fromarray(arr, mode="RGB")
    path = tmp_path / "test_sketch.png"
    img.save(str(path), format="PNG")
    return path


@pytest.fixture()
def white_float_image() -> np.ndarray:
    """float32 RGB (100, 150, 3) — uniform near-white."""
    return _make_rgb_array(100, 150, color=(0.95, 0.95, 0.95))


@pytest.fixture()
def sketch_float_image() -> np.ndarray:
    """float32 RGB (100, 150, 3) — white background with a dark pencil stroke."""
    img = _make_rgb_array(100, 150, color=(0.95, 0.95, 0.95))
    # Add a dark horizontal stroke in the middle
    img[40:60, 20:130, :] = 0.1
    return img


@pytest.fixture()
def rgba_sketch() -> np.ndarray:
    """uint8 RGBA (100, 150, 4) with a visible stroke."""
    arr = np.zeros((100, 150, 4), dtype=np.uint8)
    arr[:, :, :3] = 200
    arr[:, :, 3]  = 0
    arr[40:60, 20:130, :3] = 30
    arr[40:60, 20:130, 3]  = 200
    return arr


@pytest.fixture()
def original_rgb() -> np.ndarray:
    """uint8 RGB (100, 150, 3) — a plain grey reference image."""
    return np.full((100, 150, 3), 200, dtype=np.uint8)
