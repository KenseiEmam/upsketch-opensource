"""
SketchLift local web UI — Flask development server.

Provides a browser-based interface for running the image-processing pipeline
locally. All processing is synchronous (in-process).

Start with:
    python local_ui/app.py
    # or as a module from the repository root:
    python -m local_ui.app

Then open http://localhost:5174 in your browser.

Routes
------
GET  /                      Serve the SPA shell
POST /api/process           Upload + process; returns job JSON
GET  /api/jobs/<id>         Poll job status (always "complete" here — sync)
GET  /api/jobs/<id>/result  Return base64-encoded result images
GET  /api/download/<id>/<fmt>  Download PNG / PSD / ORA
DELETE /api/jobs/<id>       Remove job files from temp dir
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
import time
import traceback
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, render_template, abort
from werkzeug.utils import secure_filename

# Add parent directory to path so 'pipeline' is importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.export import export_all

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# In-memory job registry: job_id -> { status, paths, metadata, error }
# Lives only for the duration of the server process.
_JOBS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _job_or_404(job_id: str) -> dict:
    job = _JOBS.get(job_id)
    if job is None:
        abort(404, description=f"Job {job_id!r} not found")
    return job


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/process")
def process():
    """
    Receive an uploaded image, run the full pipeline synchronously,
    and return the job result.

    Form fields (all optional):
        sensitivity     float 0-1 (default 0.85)
        kernel          int (default 80)
        illum_method    morphological|gaussian|clahe
        alpha_method    soft_threshold|bilateral|dog|clahe_inverted
        no_perspective  1|0 (default 0)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename or not _allowed(f.filename):
        return jsonify({"error": "Unsupported file type. Use JPEG or PNG."}), 400

    # Parse optional pipeline parameters
    sensitivity    = float(request.form.get("sensitivity", 0.85))
    kernel         = int(request.form.get("kernel", 80))
    illum_method   = request.form.get("illum_method", "morphological")
    alpha_method   = request.form.get("alpha_method", "soft_threshold")
    no_perspective = request.form.get("no_perspective", "0") == "1"

    # Clamp sensitivity
    sensitivity = max(0.0, min(1.0, sensitivity))

    job_id   = str(uuid.uuid4())
    work_dir = Path(tempfile.mkdtemp(prefix=f"sketchlift_{job_id[:8]}_"))

    # Save upload
    safe_name   = secure_filename(f.filename)
    source_path = work_dir / safe_name
    f.save(str(source_path))

    # ---- Run pipeline -------------------------------------------------------
    t0 = time.perf_counter()
    try:
        import numpy as np
        from pipeline.decode import decode_image
        from pipeline.orient import normalize_orientation
        from pipeline.detect_page import detect_page_quad
        from pipeline.deskew import correct_perspective
        from pipeline.normalize_illumination import normalize_illumination
        from pipeline.extract_alpha import extract_alpha_mask
        from pipeline.recover_color import recover_color
        from pipeline.reconstruct import reconstruct_rgba
        from pipeline import PipelineResult

        raw, dpi = decode_image(source_path)
        oriented  = normalize_orientation(raw)
        original_rgb_uint8 = (oriented * 255).clip(0, 255).astype(np.uint8)

        if no_perspective:
            corrected            = oriented
            perspective_detected = False
        else:
            quad, perspective_detected = detect_page_quad(oriented)
            corrected = correct_perspective(oriented, quad)

        corrected_uint8 = (corrected * 255).clip(0, 255).astype(np.uint8)
        norm    = normalize_illumination(corrected, kernel_size=kernel, method=illum_method)
        alpha   = extract_alpha_mask(norm, sensitivity, method=alpha_method)
        colored = recover_color(corrected, alpha)
        rgba    = reconstruct_rgba(colored, alpha)

        result = PipelineResult(
            rgba=rgba,
            alpha=alpha,
            corrected_rgb=corrected_uint8,
            original_rgb=original_rgb_uint8,
            metadata={
                "source_path":           str(source_path),
                "original_size":         oriented.shape[:2],
                "output_size":           rgba.shape[:2],
                "dpi":                   dpi,
                "perspective_corrected": perspective_detected,
                "sensitivity":           sensitivity,
                "illumination_kernel":   kernel,
                "timings":               {},
                "total_seconds":         round(time.perf_counter() - t0, 3),
            },
        )

        # Export all formats to work_dir/outputs/
        out_dir  = work_dir / "outputs"
        stem     = Path(safe_name).stem
        paths    = export_all(result, out_dir, stem=stem)
        elapsed  = round(time.perf_counter() - t0, 2)

        _JOBS[job_id] = {
            "status":   "complete",
            "job_id":   job_id,
            "filename": safe_name,
            "work_dir": str(work_dir),
            "paths":    {fmt: str(p) for fmt, p in paths.items()},
            "source":   str(source_path),
            "metadata": result.metadata,
            "elapsed":  elapsed,
            "error":    None,
        }

    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 2)
        _JOBS[job_id] = {
            "status":  "failed",
            "job_id":  job_id,
            "work_dir": str(work_dir),
            "error":   str(exc),
            "traceback": traceback.format_exc(),
            "elapsed": elapsed,
        }
        return jsonify(_public_job(_JOBS[job_id])), 500

    return jsonify(_public_job(_JOBS[job_id])), 200


@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    """Return job status and metadata."""
    return jsonify(_public_job(_job_or_404(job_id)))


@app.get("/api/jobs/<job_id>/result")
def get_result(job_id: str):
    """
    Return base64-encoded images for the before/after preview.
    Keeps images small for the UI by resizing the source preview.
    """
    job = _job_or_404(job_id)
    if job["status"] != "complete":
        return jsonify({"error": "Job not complete"}), 400

    import numpy as np
    from PIL import Image as PilImage

    # Source preview (JPEG, max 1200px wide)
    src_img = PilImage.open(job["source"]).convert("RGB")
    src_img.thumbnail((1200, 1200))
    src_b64 = _pil_to_b64_jpeg(src_img)

    # Output preview (PNG to preserve transparency, max 1200px wide)
    out_img = PilImage.open(job["paths"]["png"]).convert("RGBA")
    out_img.thumbnail((1200, 1200))
    out_b64 = _pil_to_b64_png(out_img)

    return jsonify({
        "source_b64": src_b64,
        "output_b64": out_b64,
        "width":  out_img.width,
        "height": out_img.height,
    })


@app.get("/api/download/<job_id>/<fmt>")
def download(job_id: str, fmt: str):
    """Stream a processed output file to the browser."""
    job = _job_or_404(job_id)
    if job["status"] != "complete":
        abort(400, description="Job not complete")

    if fmt not in job["paths"]:
        abort(404, description=f"Format {fmt!r} not available")

    file_path = Path(job["paths"][fmt])
    if not file_path.exists():
        abort(404, description="Output file missing")

    mime_map = {"png": "image/png", "psd": "application/octet-stream",
                "ora": "application/zip"}
    return send_file(
        str(file_path),
        mimetype=mime_map.get(fmt, "application/octet-stream"),
        as_attachment=True,
        download_name=file_path.name,
    )


@app.delete("/api/jobs/<job_id>")
def delete_job(job_id: str):
    """Delete a job and its temporary files."""
    job = _job_or_404(job_id)
    work_dir = Path(job["work_dir"])
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    del _JOBS[job_id]
    return "", 204


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _public_job(job: dict) -> dict:
    """Strip internal paths from the job dict before sending to the client."""
    meta = job.get("metadata", {})
    timings = meta.get("timings", {})
    return {
        "job_id":     job["job_id"],
        "status":     job["status"],
        "filename":   job.get("filename"),
        "elapsed":    job.get("elapsed"),
        "error":      job.get("error"),
        "formats":    list(job["paths"].keys()) if job.get("paths") else [],
        "output_size": meta.get("output_size"),
        "dpi":         meta.get("dpi"),
        "perspective_corrected": meta.get("perspective_corrected"),
        "timings":    timings,
    }


def _pil_to_b64_jpeg(img) -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _pil_to_b64_png(img) -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5174))
    print(f"\n  SketchLift local UI  ->  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=True)
