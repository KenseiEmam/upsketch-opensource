#!/usr/bin/env python3
"""
SketchLift CLI -- standalone image-processing prototype.

Usage examples
--------------
Basic: process one image, write PNG/PSD/ORA to ./out/
    python cli.py sketch.jpg

Custom output directory and sensitivity
    python cli.py sketch.jpg --output-dir ./results --sensitivity 0.90

Compare all alpha-extraction methods side-by-side
    python cli.py sketch.jpg --benchmark

Save a debug image showing the detected page quad
    python cli.py sketch.jpg --debug

Skip perspective correction (useful if the page fills the frame)
    python cli.py sketch.jpg --no-perspective

Process every JPEG/PNG in a folder
    python cli.py photos/ --output-dir ./batch_out

Use a larger illumination kernel for images with large shaded areas
    python cli.py sketch.jpg --kernel 120
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Colour output helpers
# ---------------------------------------------------------------------------

def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"

def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"

def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"

def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sketchlift",
        description="Convert a pencil sketch photograph to transparent digital artwork.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "input",
        metavar="INPUT",
        help="Path to a JPEG/PNG image file, or a directory to process in batch.",
    )
    p.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        default="./out",
        help="Directory for output files (default: ./out).",
    )
    p.add_argument(
        "--sensitivity", "-s",
        metavar="FLOAT",
        type=float,
        default=0.85,
        help=(
            "Alpha extraction sensitivity 0.0-1.0 (default: 0.85). "
            "Higher = more strokes kept; lower = cleaner background."
        ),
    )
    p.add_argument(
        "--kernel", "-k",
        metavar="INT",
        type=int,
        default=80,
        help=(
            "Illumination normalisation kernel size in pixels (default: 80). "
            "Increase to 120+ for images with large shaded/filled areas."
        ),
    )
    p.add_argument(
        "--illum-method",
        choices=["morphological", "gaussian", "clahe"],
        default="morphological",
        help="Illumination normalisation algorithm (default: morphological).",
    )
    p.add_argument(
        "--alpha-method",
        choices=["soft_threshold", "bilateral", "dog", "clahe_inverted"],
        default="soft_threshold",
        help="Alpha mask extraction algorithm (default: soft_threshold).",
    )
    p.add_argument(
        "--formats", "-f",
        nargs="+",
        choices=["png", "psd", "ora"],
        default=["png", "psd", "ora"],
        help="Export formats (default: png psd ora).",
    )
    p.add_argument(
        "--no-perspective",
        action="store_true",
        help="Skip perspective correction (use when page fills the frame).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Save a debug image showing the detected page quad.",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Run all alpha-extraction methods and save side-by-side comparison. "
            "Useful for tuning on a new image type."
        ),
    )
    p.add_argument(
        "--no-psd",
        action="store_true",
        help="Skip PSD export (saves time; PSD generation is the slowest step).",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    return p


# ---------------------------------------------------------------------------
# Single image processing
# ---------------------------------------------------------------------------

def process_image(
    input_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> bool:
    """
    Run the pipeline on one image and write outputs.
    Returns True on success, False on failure.
    """
    quiet = args.quiet

    if not quiet:
        print(f"\n{_bold('Input:')}  {input_path}")
        print(f"{_bold('Output:')} {output_dir}/")

    # ---- Import pipeline here so import errors surface cleanly ------------
    try:
        from pipeline import run_pipeline, PipelineResult
        from pipeline.export import export_png, export_psd, export_ora
        from pipeline.detect_page import draw_debug_quad
        from pipeline.normalize_illumination import normalize_illumination
        from pipeline.extract_alpha import extract_alpha_mask
        from pipeline.reconstruct import reconstruct_rgba
        from pipeline.recover_color import recover_color
    except ImportError as exc:
        print(_red(f"Import error: {exc}"))
        print(_yellow("Run: pip install -e '.[dev]' from the repository root."))
        return False

    t_total = time.perf_counter()

    # ---- Decode + orient (always run) -------------------------------------
    try:
        from pipeline.decode import decode_image
        from pipeline.orient import normalize_orientation
        raw, dpi = decode_image(input_path)
        oriented = normalize_orientation(raw)
    except Exception as exc:
        print(_red(f"  FAILED (decode): {exc}"))
        return False

    # ---- Page detection + perspective correction --------------------------
    if not args.no_perspective:
        from pipeline.detect_page import detect_page_quad
        from pipeline.deskew import correct_perspective

        quad, detected = detect_page_quad(oriented)

        if args.debug:
            debug_img = draw_debug_quad(oriented, quad, detected)
            debug_path = output_dir / f"{input_path.stem}_debug_quad.png"
            Image.fromarray(debug_img).save(str(debug_path))
            if not quiet:
                label = _green("detected") if detected else _yellow("fallback (full bounds)")
                print(f"  Page quad: {label}")
                print(f"  Debug image: {debug_path}")

        corrected = correct_perspective(oriented, quad)
    else:
        corrected = oriented
        detected = False
        if not quiet:
            print(f"  {_dim('Perspective correction skipped')}")

    # ---- Illumination normalisation ---------------------------------------
    from pipeline.normalize_illumination import normalize_illumination
    normalized = normalize_illumination(
        corrected,
        kernel_size=args.kernel,
        method=args.illum_method,
    )

    # ---- Benchmark mode: compare all alpha methods -----------------------
    if args.benchmark:
        _run_benchmark(corrected, normalized, dpi, output_dir, input_path.stem, quiet)

    # ---- Alpha extraction -------------------------------------------------
    alpha = extract_alpha_mask(normalized, args.sensitivity, method=args.alpha_method)

    # ---- Colour recovery + reconstruction ---------------------------------
    colored = recover_color(corrected, alpha)
    rgba = reconstruct_rgba(colored, alpha)
    corrected_uint8 = (corrected * 255).clip(0, 255).astype(np.uint8)

    # ---- Export -----------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    written: list[str] = []

    if "png" in args.formats:
        path = output_dir / f"{stem}.png"
        export_png(rgba, path, dpi=dpi)
        written.append(f"  {_green('PNG')}  {path}  ({_file_size(path)})")

    if "psd" in args.formats and not args.no_psd:
        path = output_dir / f"{stem}.psd"
        try:
            export_psd(rgba, corrected_uint8, path)
            written.append(f"  {_green('PSD')}  {path}  ({_file_size(path)})")
        except ImportError:
            written.append(f"  {_yellow('PSD')}  skipped (psd-tools not installed)")
        except Exception as exc:
            written.append(f"  {_red('PSD')}  FAILED: {exc}")

    if "ora" in args.formats:
        path = output_dir / f"{stem}.ora"
        export_ora(rgba, corrected_uint8, path, dpi=dpi)
        written.append(f"  {_green('ORA')}  {path}  ({_file_size(path)})")

    elapsed = time.perf_counter() - t_total

    if not quiet:
        for line in written:
            print(line)
        size_str = f"{corrected.shape[1]}x{corrected.shape[0]}px"
        corr_str = _green("yes") if detected else _dim("no (full bounds)")
        print(
            f"\n  {_bold('Size:')} {size_str}  "
            f"{_bold('Perspective:')} {corr_str}  "
            f"{_bold('Time:')} {elapsed:.2f}s"
        )

    return True


# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------

def _run_benchmark(
    corrected: np.ndarray,
    normalized: np.ndarray,
    dpi: tuple,
    output_dir: Path,
    stem: str,
    quiet: bool,
) -> None:
    """
    Run all four alpha-extraction methods and save alpha visualisations.
    Useful for choosing the best method for a new image type.
    """
    from pipeline.extract_alpha import (
        extract_soft_threshold,
        extract_bilateral,
        extract_dog,
        extract_clahe_inverted,
    )
    from pipeline.reconstruct import reconstruct_rgba
    from pipeline.recover_color import recover_color
    from pipeline.export import export_png

    methods = {
        "soft_threshold":   extract_soft_threshold,
        "bilateral":        extract_bilateral,
        "dog":              extract_dog,
        "clahe_inverted":   extract_clahe_inverted,
    }

    bench_dir = output_dir / f"{stem}_benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"\n  {_bold('Benchmark:')} running {len(methods)} methods...")

    colored = recover_color(corrected, np.ones(corrected.shape[:2], dtype=np.float32))

    for name, fn in methods.items():
        t0 = time.perf_counter()
        alpha = fn(normalized, sensitivity=0.85)
        rgba = reconstruct_rgba(colored, alpha)
        path = bench_dir / f"{stem}_{name}.png"
        export_png(rgba, path, dpi=dpi)
        elapsed = time.perf_counter() - t0
        if not quiet:
            print(f"    {name:<20} {elapsed:.2f}s  -> {path.name}")

    if not quiet:
        print(f"  Benchmark images saved to: {bench_dir}/")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def process_directory(
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    extensions = {".jpg", ".jpeg", ".png"}
    images = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    if not images:
        print(_yellow(f"No JPEG/PNG images found in {input_dir}"))
        sys.exit(1)

    print(f"{_bold(str(len(images)))} image(s) found in {input_dir}\n")

    ok = failed = 0
    for img_path in images:
        img_out = output_dir / img_path.stem
        success = process_image(img_path, img_out, args)
        if success:
            ok += 1
        else:
            failed += 1

    print(f"\n{_bold('Done.')} {_green(str(ok))} succeeded, {_red(str(failed))} failed.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_size(path: Path) -> str:
    try:
        size = path.stat().st_size
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.0f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(_red(f"Error: input not found: {input_path}"))
        sys.exit(1)

    if input_path.is_dir():
        process_directory(input_path, output_dir, args)
    else:
        success = process_image(input_path, output_dir, args)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
