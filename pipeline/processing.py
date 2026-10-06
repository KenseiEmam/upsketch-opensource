"""
Shared processing core — the single place the pipeline is invoked.

This isolates the one thing that must never diverge between the different ways
the pipeline is driven (local Flask UI, CLI, and — in downstream projects —
local and production workers): how a job's options are turned into a pipeline
run and a set of export files.

Callers differ only in their I/O (where the source bytes come from, where
outputs go, how status is recorded) — never in the image processing itself.
Neither this module nor the pipeline it calls knows anything about S3, SQS,
DynamoDB, HTTP, or a filesystem queue. Callers provide a local source file path
and an output directory; they get back the output file paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# Options may arrive from an HTTP API as camelCase JSON. Map to run_pipeline kwargs.
def pipeline_kwargs(options: dict[str, Any] | None) -> dict[str, Any]:
    """Translate camelCase option keys into run_pipeline keyword arguments.

    Omitted/None fields fall back to run_pipeline's own defaults.
    """
    options = options or {}
    kwargs: dict[str, Any] = {}
    if options.get("sensitivity") is not None:
        kwargs["sensitivity"] = float(options["sensitivity"])
    if options.get("illuminationKernel") is not None:
        kwargs["illumination_kernel"] = int(options["illuminationKernel"])
    if options.get("perspective") is not None:
        kwargs["perspective"] = bool(options["perspective"])
    if options.get("illuminationMethod") is not None:
        kwargs["illumination_method"] = str(options["illuminationMethod"])
    if options.get("alphaMethod") is not None:
        kwargs["alpha_method"] = str(options["alphaMethod"])
    # Passed through as a dict; run_pipeline / apply_transform own the
    # interpretation (rotate degrees + normalized crop). Do not reshape it here.
    if options.get("transform") is not None:
        kwargs["transform"] = options["transform"]
    return kwargs


def process_to_files(
    source_path: str | Path,
    output_dir: str | Path,
    stem: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Run the pipeline on a local file and export all formats to output_dir.

    Parameters
    ----------
    source_path:
        Local path to the (already-downloaded) source image.
    output_dir:
        Local directory to write PNG/PSD/ORA into.
    stem:
        Base filename (without extension) for the output files.
    options:
        Optional camelCase options dict (e.g. from an HTTP API).

    Returns
    -------
    dict mapping format name ('png'|'psd'|'ora') to the written Path.

    Raises
    ------
    Any exception from the pipeline or export stages. Callers are responsible
    for catching these and recording the job as failed.
    """
    # Imported lazily so importing this module doesn't require OpenCV etc.
    from . import run_pipeline
    from .export import export_all

    kwargs = pipeline_kwargs(options)
    result = run_pipeline(source_path, **kwargs)
    written = export_all(result, output_dir, stem=stem)
    return {fmt: Path(p) for fmt, p in written.items()}
