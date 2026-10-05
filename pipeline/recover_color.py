"""
Stage 7 — Stroke colour recovery.

The alpha mask is derived from luminance, but the final RGBA should
carry the actual stroke colour.  Most pencil sketches are grey-black,
but coloured pencils, warm lighting, or tinted paper can introduce a
colour cast that should be preserved in the output.

Result: each output pixel's RGB equals the original corrected RGB,
so the colour character of the graphite is preserved.  The alpha
channel (from stage 6) controls what is visible.
"""

from __future__ import annotations

import numpy as np


def recover_color(
    corrected_rgb: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Keep the original RGB colour for stroke pixels; paper pixels will be
    made transparent by the alpha channel during RGBA reconstruction.

    Parameters
    ----------
    corrected_rgb:
        float32 RGB (H, W, 3), values in [0, 1] — perspective-corrected source.
    alpha:
        float32 (H, W), values in [0, 1] — continuous alpha mask.

    Returns
    -------
    float32 RGB (H, W, 3) where paper regions are blended toward a neutral
    colour and stroke regions retain their original colour.

    Note: we do NOT premultiply here.  The RGB values are straight (unassociated)
    colour; premultiplication happens optionally at the encoding stage.
    """
    # For pixels with very low alpha (paper), the colour is irrelevant because
    # those pixels will be transparent in the output.  We pass the original
    # colour through unchanged.  This keeps the export simple and correct.
    return corrected_rgb.astype(np.float32)
