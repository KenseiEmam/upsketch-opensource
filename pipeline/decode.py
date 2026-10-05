"""
Stage 1 — Image decoding.

Loads a JPEG or PNG file into a float32 RGB array normalised to [0, 1].
This is the single point where format-specific logic lives; adding HEIC
later means a pre-decode conversion step here with no other changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

# Supported MIME types → magic-byte signatures for server-side verification.
SUPPORTED_MODES = {"RGB", "RGBA", "L", "LA", "P", "1"}
MAGIC_BYTES: dict[str, bytes] = {
    "jpeg": b"\xff\xd8\xff",
    "png": b"\x89PNG",
}


def decode_image(path: str | Path) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Decode an image file into a float32 RGB array.

    Parameters
    ----------
    path:
        Path to a JPEG or PNG file.

    Returns
    -------
    (image, dpi) where:
        image — float32 ndarray of shape (H, W, 3), values in [0, 1], RGB order
        dpi   — (x_dpi, y_dpi) tuple, defaults to (72, 72) if not embedded
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    _verify_magic(path)

    try:
        pil_img = Image.open(path)
        pil_img.load()  # force decode; catches truncated files early
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or corrupt image file: {path}") from exc

    # Apply EXIF orientation before anything else
    from .orient import apply_exif_transpose  # local import avoids circular dep
    pil_img = apply_exif_transpose(pil_img)

    # Extract DPI before any conversion (some modes lose it)
    dpi = _extract_dpi(pil_img)

    # Normalise to RGB (drop alpha if present; it will be re-generated)
    if pil_img.mode == "RGBA":
        # Composite onto white before processing; original alpha is not useful
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_img = bg
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return arr, dpi


def _verify_magic(path: Path) -> None:
    """Raise ValueError if the file header does not match a supported format."""
    with open(path, "rb") as fh:
        header = fh.read(8)

    for fmt, magic in MAGIC_BYTES.items():
        if header[: len(magic)] == magic:
            return

    raise ValueError(
        f"File does not appear to be a JPEG or PNG (bad magic bytes): {path}"
    )


def _extract_dpi(img: Image.Image) -> tuple[float, float]:
    """Return the DPI embedded in the image, or (72, 72) as a safe default."""
    try:
        dpi = img.info.get("dpi")
        if dpi and isinstance(dpi, (tuple, list)) and len(dpi) == 2:
            x, y = float(dpi[0]), float(dpi[1])
            # Some encoders write 1 or 0 as a placeholder — treat as 72
            if x >= 1 and y >= 1:
                return (x, y)
    except Exception:
        pass
    return (72.0, 72.0)
