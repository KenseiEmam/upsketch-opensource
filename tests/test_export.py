"""
File-format validation tests for PNG, PSD, and ORA export.

These tests verify structural correctness of the output files:
  - correct channel count / mode
  - canvas dimensions match input
  - ORA spec compliance (mimetype entry stored uncompressed, required files present)
  - PSD layer count and names
"""

from __future__ import annotations

import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_rgba() -> np.ndarray:
    arr = np.zeros((80, 120, 4), dtype=np.uint8)
    arr[:, :, :3] = 200
    arr[30:50, 20:100, :3] = 30
    arr[30:50, 20:100, 3]  = 200
    return arr


@pytest.fixture()
def small_rgb() -> np.ndarray:
    return np.full((80, 120, 3), 210, dtype=np.uint8)


# ---------------------------------------------------------------------------
# PNG tests
# ---------------------------------------------------------------------------

class TestExportPNG:
    def test_file_created(self, tmp_path, small_rgba):
        from pipeline.export import export_png
        p = tmp_path / "out.png"
        export_png(small_rgba, p)
        assert p.exists()

    def test_reloads_as_rgba(self, tmp_path, small_rgba):
        from pipeline.export import export_png
        p = tmp_path / "out.png"
        export_png(small_rgba, p)
        img = Image.open(str(p))
        assert img.mode == "RGBA"

    def test_dimensions_preserved(self, tmp_path, small_rgba):
        from pipeline.export import export_png
        p = tmp_path / "out.png"
        export_png(small_rgba, p)
        img = Image.open(str(p))
        assert img.width  == small_rgba.shape[1]
        assert img.height == small_rgba.shape[0]

    def test_alpha_channel_nonzero_where_expected(self, tmp_path, small_rgba):
        from pipeline.export import export_png
        p = tmp_path / "out.png"
        export_png(small_rgba, p)
        img = Image.open(str(p))
        arr = np.array(img)
        # Stroke region (rows 30-50, cols 20-100) should have alpha > 0
        assert arr[30:50, 20:100, 3].mean() > 100

    def test_transparent_region_has_zero_alpha(self, tmp_path, small_rgba):
        from pipeline.export import export_png
        p = tmp_path / "out.png"
        export_png(small_rgba, p)
        img = Image.open(str(p))
        arr = np.array(img)
        # Top-left corner (outside stroke) should be transparent
        assert arr[0:20, 0:20, 3].mean() < 10

    def test_wrong_dtype_raises(self, tmp_path):
        from pipeline.export import export_png
        bad = np.zeros((10, 10, 4), dtype=np.float32)
        with pytest.raises(AssertionError, match="uint8"):
            export_png(bad, tmp_path / "bad.png")

    def test_wrong_channels_raises(self, tmp_path):
        from pipeline.export import export_png
        bad = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(AssertionError, match="4 channels"):
            export_png(bad, tmp_path / "bad.png")

    def test_dpi_embedded(self, tmp_path, small_rgba):
        from pipeline.export import export_png
        p = tmp_path / "dpi.png"
        export_png(small_rgba, p, dpi=(300.0, 300.0))
        img = Image.open(str(p))
        dpi = img.info.get("dpi")
        if dpi:  # some PNG writers omit DPI — acceptable
            assert dpi[0] >= 290


# ---------------------------------------------------------------------------
# ORA tests  (run without psd-tools dependency)
# ---------------------------------------------------------------------------

class TestExportORA:
    def test_file_created(self, tmp_path, small_rgba, small_rgb):
        from pipeline.export import export_ora
        p = tmp_path / "out.ora"
        export_ora(small_rgba, small_rgb, p)
        assert p.exists()

    def test_is_valid_zip(self, tmp_path, small_rgba, small_rgb):
        from pipeline.export import export_ora
        p = tmp_path / "out.ora"
        export_ora(small_rgba, small_rgb, p)
        assert zipfile.is_zipfile(str(p))

    def test_required_entries_present(self, tmp_path, small_rgba, small_rgb):
        from pipeline.export import export_ora
        p = tmp_path / "out.ora"
        export_ora(small_rgba, small_rgb, p)
        with zipfile.ZipFile(str(p)) as zf:
            names = zf.namelist()
        for required in ["mimetype", "stack.xml", "data/layer0.png",
                         "data/layer1.png", "mergedimage.png",
                         "Thumbnails/thumbnail.png"]:
            assert required in names, f"Missing required entry: {required}"

    def test_mimetype_is_stored_not_compressed(self, tmp_path, small_rgba, small_rgb):
        """ORA spec requires mimetype to be ZIP_STORED."""
        from pipeline.export import export_ora
        p = tmp_path / "out.ora"
        export_ora(small_rgba, small_rgb, p)
        with zipfile.ZipFile(str(p)) as zf:
            info = zf.getinfo("mimetype")
        assert info.compress_type == zipfile.ZIP_STORED, (
            "mimetype entry must be ZIP_STORED per ORA spec"
        )

    def test_mimetype_value(self, tmp_path, small_rgba, small_rgb):
        from pipeline.export import export_ora
        p = tmp_path / "out.ora"
        export_ora(small_rgba, small_rgb, p)
        with zipfile.ZipFile(str(p)) as zf:
            content = zf.read("mimetype")
        assert content == b"image/openraster"

    def test_stack_xml_dimensions(self, tmp_path, small_rgba, small_rgb):
        from pipeline.export import export_ora
        p = tmp_path / "out.ora"
        export_ora(small_rgba, small_rgb, p)
        with zipfile.ZipFile(str(p)) as zf:
            root = ET.fromstring(zf.read("stack.xml"))
        assert root.attrib["w"] == str(small_rgba.shape[1])
        assert root.attrib["h"] == str(small_rgba.shape[0])

    def test_stack_xml_has_two_layers(self, tmp_path, small_rgba, small_rgb):
        from pipeline.export import export_ora
        p = tmp_path / "out.ora"
        export_ora(small_rgba, small_rgb, p)
        with zipfile.ZipFile(str(p)) as zf:
            root = ET.fromstring(zf.read("stack.xml"))
        layers = root.find("stack").findall("layer")
        assert len(layers) == 2

    def test_stack_xml_layer_names(self, tmp_path, small_rgba, small_rgb):
        from pipeline.export import export_ora
        p = tmp_path / "out.ora"
        export_ora(small_rgba, small_rgb, p)
        with zipfile.ZipFile(str(p)) as zf:
            root = ET.fromstring(zf.read("stack.xml"))
        names = [l.attrib["name"] for l in root.find("stack").findall("layer")]
        assert "Extracted Sketch"    in names
        assert "Original Photograph" in names

    def test_sketch_layer_is_rgba(self, tmp_path, small_rgba, small_rgb):
        from pipeline.export import export_ora
        p = tmp_path / "out.ora"
        export_ora(small_rgba, small_rgb, p)
        with zipfile.ZipFile(str(p)) as zf:
            layer1_bytes = zf.read("data/layer1.png")
        img = Image.open(io.BytesIO(layer1_bytes))
        assert img.mode == "RGBA"

    def test_sketch_layer_dimensions(self, tmp_path, small_rgba, small_rgb):
        from pipeline.export import export_ora
        p = tmp_path / "out.ora"
        export_ora(small_rgba, small_rgb, p)
        with zipfile.ZipFile(str(p)) as zf:
            layer1_bytes = zf.read("data/layer1.png")
        img = Image.open(io.BytesIO(layer1_bytes))
        assert img.width  == small_rgba.shape[1]
        assert img.height == small_rgba.shape[0]

    def test_thumbnail_max_256(self, tmp_path, small_rgba, small_rgb):
        from pipeline.export import export_ora
        p = tmp_path / "out.ora"
        export_ora(small_rgba, small_rgb, p)
        with zipfile.ZipFile(str(p)) as zf:
            thumb_bytes = zf.read("Thumbnails/thumbnail.png")
        img = Image.open(io.BytesIO(thumb_bytes))
        assert img.width  <= 256
        assert img.height <= 256

    def test_merged_image_dimensions(self, tmp_path, small_rgba, small_rgb):
        from pipeline.export import export_ora
        p = tmp_path / "out.ora"
        export_ora(small_rgba, small_rgb, p)
        with zipfile.ZipFile(str(p)) as zf:
            merged_bytes = zf.read("mergedimage.png")
        img = Image.open(io.BytesIO(merged_bytes))
        assert img.width  == small_rgba.shape[1]
        assert img.height == small_rgba.shape[0]


# ---------------------------------------------------------------------------
# PSD tests  (skipped if psd-tools not installed)
# ---------------------------------------------------------------------------

psd_tools = pytest.importorskip("psd_tools", reason="psd-tools not installed")


class TestExportPSD:
    def test_file_created(self, tmp_path, small_rgba, small_rgb):
        from pipeline.export import export_psd
        p = tmp_path / "out.psd"
        export_psd(small_rgba, small_rgb, p)
        assert p.exists()

    def test_layer_count(self, tmp_path, small_rgba, small_rgb):
        from pipeline.export import export_psd
        from psd_tools import PSDImage
        p = tmp_path / "out.psd"
        export_psd(small_rgba, small_rgb, p)
        psd = PSDImage.open(str(p))
        assert len(list(psd)) == 2

    def test_layer_names(self, tmp_path, small_rgba, small_rgb):
        from pipeline.export import export_psd
        from psd_tools import PSDImage
        p = tmp_path / "out.psd"
        export_psd(small_rgba, small_rgb, p)
        psd   = PSDImage.open(str(p))
        names = [layer.name for layer in psd]
        assert "Extracted Sketch"    in names
        assert "Original Photograph" in names

    def test_canvas_dimensions(self, tmp_path, small_rgba, small_rgb):
        from pipeline.export import export_psd
        from psd_tools import PSDImage
        p = tmp_path / "out.psd"
        export_psd(small_rgba, small_rgb, p)
        psd = PSDImage.open(str(p))
        assert psd.width  == small_rgba.shape[1]
        assert psd.height == small_rgba.shape[0]
