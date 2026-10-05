"""
Stage 10 — File export.

Produces PNG, PSD, and OpenRaster (.ora) files from the pipeline result.

PNG:  Pillow — transparent RGBA, lossless.
PSD:  psd-tools library — two-layer Photoshop document.
ORA:  Manual ZIP assembly per the OpenRaster 0.0.3 specification.
      (No mature Python library exists; the format is simple enough to
       write directly without one.)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------

def export_png(
    rgba: np.ndarray,
    output_path: str | Path,
    dpi: tuple[float, float] = (72.0, 72.0),
) -> None:
    """
    Write a uint8 RGBA array as a transparent PNG.

    Parameters
    ----------
    rgba:
        uint8 (H, W, 4).
    output_path:
        Destination file path (created or overwritten).
    dpi:
        (x_dpi, y_dpi) to embed in PNG metadata.
    """
    assert rgba.dtype == np.uint8,      "rgba must be uint8"
    assert rgba.ndim == 3,              "rgba must be 3-dimensional"
    assert rgba.shape[2] == 4,          "rgba must have 4 channels"

    img = Image.fromarray(rgba, mode="RGBA")
    img.save(str(output_path), format="PNG", dpi=dpi, optimize=False)


# ---------------------------------------------------------------------------
# PSD
# ---------------------------------------------------------------------------

def export_psd(
    rgba_sketch: np.ndarray,
    original_rgb: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Write a two-layer PSD using psd-tools.

    Layers (top → bottom as seen in Photoshop):
      • Extracted Sketch  — RGBA, visible
      • Original Photo    — RGB,  hidden (reference)

    Parameters
    ----------
    rgba_sketch:
        uint8 (H, W, 4) extracted sketch with transparency.
    original_rgb:
        uint8 (H, W, 3) perspective-corrected source photograph.
    output_path:
        Destination .psd file path.
    """
    # Import here so the rest of the pipeline works even if psd-tools is
    # not installed (e.g. when running unit tests with stubs).
    try:
        from psd_tools import PSDImage
    except ImportError as exc:
        raise ImportError(
            "psd-tools is required for PSD export.  "
            "Install it with: pip install psd-tools"
        ) from exc

    from psd_tools.api.layers import PixelLayer

    h, w = rgba_sketch.shape[:2]
    sketch_pil   = Image.fromarray(rgba_sketch, mode="RGBA")
    original_pil = Image.fromarray(original_rgb, mode="RGB")

    psd = PSDImage.new("RGB", (w, h))

    # Build each pixel layer explicitly and append exactly once. We do NOT use
    # PSDImage.compose_layer here: on psd-tools 1.23 it both returns a layer
    # AND mutates the document, which produced a duplicate of the first layer.
    # PixelLayer.frompil + a single append is deterministic.
    #
    # Append bottom-first so the final top-to-bottom stack reads:
    #   Extracted Sketch (top, visible)
    #   Original Photograph (bottom, hidden)
    photo_layer = PixelLayer.frompil(original_pil, psd, "Original Photograph")
    photo_layer.visible = False
    psd.append(photo_layer)

    sketch_layer = PixelLayer.frompil(sketch_pil, psd, "Extracted Sketch")
    sketch_layer.visible = True
    psd.append(sketch_layer)

    psd.save(str(output_path))


# ---------------------------------------------------------------------------
# ORA (OpenRaster)
# ---------------------------------------------------------------------------

def export_ora(
    rgba_sketch: np.ndarray,
    original_rgb: np.ndarray,
    output_path: str | Path,
    dpi: tuple[float, float] = (72.0, 72.0),
) -> None:
    """
    Write a valid OpenRaster 0.0.3 file.

    Structure
    ---------
    sketch.ora
    ├── mimetype              (ZIP_STORED, not compressed)
    ├── stack.xml
    ├── data/
    │   ├── layer0.png        (bottom: original photograph)
    │   └── layer1.png        (top:    extracted sketch, RGBA)
    ├── mergedimage.png       (flattened composite over white)
    └── Thumbnails/
        └── thumbnail.png    (≤ 256×256)

    Parameters
    ----------
    rgba_sketch:
        uint8 (H, W, 4).
    original_rgb:
        uint8 (H, W, 3).
    output_path:
        Destination .ora file path.
    dpi:
        Embedded resolution metadata.
    """
    h, w = rgba_sketch.shape[:2]
    xres, yres = int(dpi[0]), int(dpi[1])

    def _to_png_bytes(arr: np.ndarray, mode: str) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(arr, mode=mode).save(buf, format="PNG")
        return buf.getvalue()

    sketch_png   = _to_png_bytes(rgba_sketch,  "RGBA")
    original_png = _to_png_bytes(original_rgb, "RGB")

    # Flattened composite — sketch over white background
    bg = Image.new("RGB", (w, h), (255, 255, 255))
    sketch_img = Image.fromarray(rgba_sketch, "RGBA")
    bg.paste(sketch_img, mask=sketch_img.split()[3])
    merged_png = _to_png_bytes(np.array(bg), "RGB")

    # Thumbnail (spec: max 256×256)
    thumb = sketch_img.copy().convert("RGBA")
    thumb.thumbnail((256, 256), Image.LANCZOS)
    thumb_png = _to_png_bytes(np.array(thumb), "RGBA")

    stack_xml = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<image version="0.0.3" w="{w}" h="{h}" xres="{xres}" yres="{yres}">\n'
        f'  <stack>\n'
        f'    <layer name="Extracted Sketch"\n'
        f'           src="data/layer1.png"\n'
        f'           x="0" y="0"\n'
        f'           opacity="1"\n'
        f'           composite-op="svg:src-over"\n'
        f'           visibility="visible" />\n'
        f'    <layer name="Original Photograph"\n'
        f'           src="data/layer0.png"\n'
        f'           x="0" y="0"\n'
        f'           opacity="1"\n'
        f'           composite-op="svg:src-over"\n'
        f'           visibility="hidden" />\n'
        f'  </stack>\n'
        f'</image>\n'
    )

    with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zf:
        # mimetype MUST be the first entry and MUST NOT be compressed
        mi = zipfile.ZipInfo("mimetype")
        mi.compress_type = zipfile.ZIP_STORED
        zf.writestr(mi, "image/openraster")

        zf.writestr("stack.xml",                stack_xml)
        zf.writestr("data/layer0.png",          original_png)
        zf.writestr("data/layer1.png",          sketch_png)
        zf.writestr("mergedimage.png",           merged_png)
        zf.writestr("Thumbnails/thumbnail.png", thumb_png)


# ---------------------------------------------------------------------------
# Convenience: export all three formats from a PipelineResult
# ---------------------------------------------------------------------------

def export_all(
    result,  # PipelineResult — avoid circular import with type hint
    output_dir: str | Path,
    stem: str = "sketch",
) -> dict[str, Path]:
    """
    Write PNG, PSD, and ORA to output_dir/<stem>.{png,psd,ora}.

    Returns a dict mapping format name to output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dpi = result.metadata.get("dpi", (72.0, 72.0))

    paths: dict[str, Path] = {}

    png_path = output_dir / f"{stem}.png"
    export_png(result.rgba, png_path, dpi=dpi)
    paths["png"] = png_path

    psd_path = output_dir / f"{stem}.psd"
    export_psd(result.rgba, result.corrected_rgb, psd_path)
    paths["psd"] = psd_path

    ora_path = output_dir / f"{stem}.ora"
    export_ora(result.rgba, result.corrected_rgb, ora_path, dpi=dpi)
    paths["ora"] = ora_path

    return paths
