# AGENTS.md

Guidance for AI agents and contributors working in the `sketchlift-pipeline`
repository. Read this before making changes.

## What this project is

A local Python image-processing pipeline that converts pencil-sketch
photographs into transparent RGBA artwork. It runs as a Python package, a CLI
(`cli.py`), and a local Flask UI (`local_ui/`). No GPU required.

## Hard rules

- **No generative changes.** Every stage transforms the user's own pixels
  (geometry, lighting, alpha, colour). Never synthesise or hallucinate image
  content. Reframing, filtering, and compositing are fine; inventing strokes or
  detail is not.
- **Never modify the user's source file on disk.** Operate on in-memory copies
  (or a temp file). The input photograph must come out byte-for-byte unchanged.
- **Keep each pipeline stage pure and separately testable.** A stage function
  should take its inputs as arguments and return a new array — no reading
  globals, no filesystem access, no in-place mutation of inputs.

## Architecture

The entry point is `pipeline.run_pipeline(source_path, ...) -> PipelineResult`
in `pipeline/__init__.py`. It calls each stage in order and records per-stage
timings plus a `metadata` dict.

Stage order (each is its own module):

1. `decode.py` — load JPEG/PNG, apply EXIF orientation, return float32 RGB
2. `orient.py` — orientation normalisation (currently a pass-through hook)
   - `transform.py` — **optional** manual geometric reframing (rotate + crop),
     applied right after orientation and before page detection. No-op unless a
     `transform` dict is passed.
3. `detect_page.py` — Canny + contour paper-quad detection
4. `deskew.py` — perspective warp (homography)
5. `normalize_illumination.py` — background estimation + divide
6. `extract_alpha.py` — continuous alpha mask extraction
7. `recover_color.py` — preserve original stroke colour
8. (ML refinement — deferred, not implemented)
9. `reconstruct.py` — compose float32 → uint8 RGBA
10. `export.py` — PNG / PSD / ORA writers

`processing.py` is the single shared core that turns a camelCase `options` dict
(e.g. from an HTTP API) into `run_pipeline` kwargs via `pipeline_kwargs`, runs
the pipeline, and exports files. Callers (CLI, Flask UI, downstream workers)
differ only in I/O, never in the image processing.

## Image/array conventions

- The pipeline carries **float32 RGB arrays, shape `(H, W, 3)`, values in
  `[0, 1]`** between stages. Paper white is `1.0` on all channels.
- The final RGBA output is **uint8 `(H, W, 4)`**; the alpha mask is **float32
  `(H, W)` in `[0, 1]`**.
- When filling exposed/empty regions, use white (`1.0`), not black or
  transparent, so downstream illumination normalisation still behaves.

## Options mapping (`processing.py::pipeline_kwargs`)

Options may arrive as camelCase JSON. Map each to the `run_pipeline` kwarg and
let the pipeline own interpretation — do not reshape values here:

| camelCase option      | run_pipeline kwarg      |
|-----------------------|-------------------------|
| `sensitivity`         | `sensitivity` (float)   |
| `illuminationKernel`  | `illumination_kernel`   |
| `perspective`         | `perspective` (bool)    |
| `illuminationMethod`  | `illumination_method`   |
| `alphaMethod`         | `alpha_method`          |
| `transform`           | `transform` (dict, pass-through) |

New knobs should be keyword-only on `run_pipeline` so existing callers stay
unaffected, and added to `pipeline_kwargs` with an `is not None` guard so
omitted fields fall back to the pipeline's own defaults.

## Development

```bash
pip install -e ".[dev]"   # runtime + pytest/pytest-cov
```

- Dependencies and version live in `pyproject.toml`. **Bump the version** when
  adding a user-visible feature so the editable install downstream picks it up,
  and note it in the README changelog.

## Testing

```bash
pytest                    # full suite
pytest tests/test_pipeline_stages.py::TestTransform   # one class
```

- Tests use synthetic in-memory images (`tests/conftest.py`); no real
  photographs or network needed.
- Regression tests (`test_pipeline_regression.py`) auto-skip unless fixture
  images with ground-truth masks exist under `tests/fixtures/`.
- When adding a stage, mirror the existing stage tests: assert dtype/shape/range
  invariants, no input mutation, and known-input → known-output behaviour.
- **Always run the suite before finishing.** Fix failures rather than adjusting
  tests to pass.

## Cleanup

- Remove any temporary files created during verification (scratch outputs, log
  captures). The workspace root should contain only committed project files.
- Build/cache artifacts (`.pytest_cache/`, `__pycache__/`, `*.egg-info/`,
  `out/`, `.venv/`) are gitignored — leave them, don't commit them.
