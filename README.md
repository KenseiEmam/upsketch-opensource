# SketchLift Image Pipeline

Python image-processing pipeline that converts pencil sketch photographs
into transparent RGBA artwork. The pipeline runs locally and can be used as a
Python package, through the command-line tool, or with the local web UI.

---

## What this does

1. Loads a JPEG or PNG photograph of a pencil sketch
2. Corrects EXIF orientation
3. Detects and crops the paper quad (perspective correction)
4. Removes uneven lighting / paper background
5. Extracts a **continuous** alpha mask (not a binary threshold)
6. Reconstructs transparent RGBA preserving the original stroke colour
7. Exports **PNG**, **PSD** (Photoshop layers), and **ORA** (GIMP / Krita layers)

---

## Requirements

- Python 3.11+
- pip

No GPU required.  Works on Windows, macOS, and Linux.

---

## Installation

```bash

# Create a virtual environment (recommended)
python -m venv .venv

# Activate it
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows Git Bash / bash
source .venv/Scripts/activate
# macOS / Linux
source .venv/bin/activate

# Install runtime dependencies
pip install -e .

# Install dev dependencies (tests)
pip install -e ".[dev]"
```

---

## Usage

### Local web UI  *(recommended for visual testing)*

```bash
python local_ui/app.py
```

Open **http://localhost:5174** in your browser.

- Drag and drop any JPEG or PNG
- Adjust pipeline controls in the sidebar
- Inspect the before/after split-slider preview
- Download PNG, PSD, or ORA

The port can be changed with the `PORT` environment variable:

```bash
PORT=8080 python local_ui/app.py
```

---

### CLI tool

```bash
# Basic — process one image, write PNG + PSD + ORA to ./out/
python cli.py sketch.jpg

# Choose output directory
python cli.py sketch.jpg --output-dir ./results

# Tune the main knobs
python cli.py sketch.jpg --sensitivity 0.90 --kernel 120

# Skip perspective correction (use when the page fills the frame)
python cli.py sketch.jpg --no-perspective

# Save a debug image showing the detected page quadrilateral
python cli.py sketch.jpg --debug

# Compare all four alpha-extraction algorithms side by side
python cli.py sketch.jpg --benchmark

# Process an entire folder
python cli.py photos/ --output-dir ./batch_out

# Only write PNG (skip PSD — faster)
python cli.py sketch.jpg --formats png ora
```

Full help:

```bash
python cli.py --help
```

---

## Pipeline controls explained

| Control | Default | Effect |
|---|---|---|
| `--sensitivity` | `0.85` | Higher → more strokes retained (may include paper noise). Lower → cleaner background (risk of losing faint lines). |
| `--kernel` | `80` | Illumination normalisation kernel size in pixels. Increase to `120`+ for images with large shaded / filled areas. |
| `--illum-method` | `morphological` | Background estimation algorithm. See benchmarking section. |
| `--alpha-method` | `soft_threshold` | Alpha mask extraction algorithm. See benchmarking section. |
| `--no-perspective` | off | Skip page detection + perspective warp. Use when the paper already fills the frame cleanly. |

---

## Benchmarking algorithms

The pipeline exposes multiple candidate algorithms so you can compare quality
on your own sketches before committing to one.

### Quick benchmark (CLI)

```bash
python cli.py sketch.jpg --benchmark
```

Runs all four alpha-extraction methods and saves a side-by-side PNG comparison
to `./out/<stem>_benchmark/`.

### Full benchmark (all fixture images)

```bash
python -c "
from tests.test_pipeline_regression import run_benchmark
run_benchmark()
"
```

Prints recall / FPR / MAE metrics for every method on every fixture image
that has a ground-truth mask.

### Algorithm options

**Illumination methods** (`--illum-method`):

| Value | Description | When to use |
|---|---|---|
| `morphological` | Morphological closing background estimation *(default)* | Most cases |
| `gaussian` | Heavy Gaussian blur background estimation | Fast alternative; less accurate at sharp shadow edges |
| `clahe` | Adaptive histogram equalisation | Does not remove gradient; useful as comparison baseline |

**Alpha extraction methods** (`--alpha-method`):

| Value | Description | When to use |
|---|---|---|
| `soft_threshold` | Gamma-boosted soft ramp *(default)* | General purpose |
| `bilateral` | Bilateral filter + soft threshold | Reduces paper grain; ~3× slower |
| `dog` | Difference of Gaussians | Sharp ink/pen lines; may miss soft graphite fills |
| `clahe_inverted` | CLAHE + soft threshold | Low-contrast images with flat lighting |

---

## Running tests

```bash
pytest

# With coverage
pytest --cov=pipeline --cov-report=term-missing

# Only structural export tests (no images needed)
pytest tests/test_export.py tests/test_decode.py -v

# Regression tests (require fixture images — see below)
pytest tests/test_pipeline_regression.py -v
```

### Adding fixture images for regression tests

1. Copy a representative sketch photograph to `tests/fixtures/images/`
2. Create a hand-labelled grayscale alpha mask and save it to
   `tests/fixtures/ground-truth/<same_stem>.alpha.png`
   (16-bit or 8-bit grayscale PNG; white = stroke, black = paper)
3. Run `pytest tests/test_pipeline_regression.py -v`

Without ground-truth masks, the regression tests are skipped automatically.
The `test_pipeline_does_not_crash` tests still run on any image present in
`tests/fixtures/images/` — useful for catching hard failures on real inputs.

---

## Project layout

```
sketchlift_OS/
  pipeline/
    __init__.py          run_pipeline() entry point + PipelineResult dataclass
    decode.py            Stage 1 — load JPEG/PNG, EXIF orientation, float32 RGB
    orient.py            Stage 2 — apply_exif_transpose helper
    detect_page.py       Stage 3 — Canny + contour quad detection
    deskew.py            Stage 4 — perspective warp (homography)
    normalize_illumination.py  Stage 5 — background estimation + divide
    extract_alpha.py     Stage 6 — continuous alpha mask extraction
    recover_color.py     Stage 7 — preserve original stroke colour
    reconstruct.py       Stage 9 — compose float32 → uint8 RGBA
    export.py            Stage 10 — PNG / PSD / ORA file writers
  local_ui/
    app.py               Flask dev server
    templates/
      index.html         Single-page UI (no external JS frameworks)
  tests/
    conftest.py          Shared synthetic image fixtures
    test_decode.py       Stage 1 unit tests
    test_pipeline_stages.py  Stages 3–9 + full pipeline smoke tests
    test_export.py       PNG / ORA / PSD structural validation
    test_pipeline_regression.py  Quality metric tests against fixture corpus
    utils.py             load_alpha_ground_truth, evaluate_alpha, print_metrics
    fixtures/
      images/            Add .jpg / .png test photographs here
      ground-truth/      Add <stem>.alpha.png ground-truth masks here
  cli.py                 Command-line tool
  pyproject.toml         Dependencies + project config
  PIPELINE_DECISIONS.md  Algorithm selection log (fill in after benchmarking)
```

---

## Output files

All three formats contain two layers:

| Layer | Content |
|---|---|
| Extracted Sketch | RGBA — transparent background, pencil strokes only |
| Original Photograph | RGB — perspective-corrected source (hidden by default) |

**PNG** — RGBA, lossless, opens in any graphics application.

**PSD** — Photoshop document.  Opens in Photoshop, Affinity Photo, and
Procreate (import via Files app).

**ORA** — OpenRaster ZIP archive.  Opens natively in GIMP and Krita.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'cv2'`**
```bash
pip install -e .
```

**`ModuleNotFoundError: No module named 'pipeline'`**
Install the project in editable mode from the repository root with
`pip install -e .`.

**PSD export fails with `AttributeError`**
The psd-tools API changed between versions.  Pin to the version in
`pyproject.toml`, or reinstall the project with `pip install -e .`.

**Output looks grey / washed-out**
Try increasing `--sensitivity` (e.g. `0.92`) and/or disabling perspective
correction (`--no-perspective`) if the background was uneven.

**Faint strokes disappear**
Increase `--sensitivity` toward `1.0`, and try `--kernel 120` if the image
has large shaded areas that are pulling the background estimate too dark.

**Paper texture visible in output**
Decrease `--sensitivity` (e.g. `0.75`), or try `--alpha-method bilateral`
which applies edge-preserving smoothing before thresholding.

---

## Pipeline API

The reusable entry point is `pipeline.run_pipeline(source_path, ...)`. It
returns a `PipelineResult` containing the transparent RGBA image, alpha mask,
corrected source image, and processing metadata. Export helpers are available
from `pipeline.export`.
