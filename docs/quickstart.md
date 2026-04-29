# Quickstart

**Goal:** count colonies on a plate photo, end-to-end, in under 15 minutes.

You don't need any Python experience to follow the CLI sections. The Python
section at the end is optional and shows the same flow from a script or
notebook for people who want to customize further.

---

## 1. Install (5 minutes)

Blenny isn't on PyPI yet, so for now you install from a checkout:

```bash
git clone https://github.com/your-org/blenny.git
cd blenny
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .
```

Verify it worked:

```bash
blenny --version
# blenny 0.0.1
```

If `blenny: command not found`, your virtual environment isn't activated
or the install failed silently. Re-run `pip install -e .` and read the
output for errors.

---

## 2. Your first count (2 minutes, no input image needed)

Blenny ships a synthetic plate generator so you can verify the pipeline
runs before you supply your own data:

```bash
python examples/01_count_colonies.py
```

You should see something like:

```
Generated synthetic plate with 30 colonies → examples/output/synthetic_plate.png
Pipeline([load_image, detect_plate, correct_illumination, threshold_segment, ...])

Detected colonies: 30
Ground-truth count: 30
Mean area (px):     156.4

No quality flags raised.

Provenance:
  load_image                 7.2 ms  {'as_gray': False, ...}
  detect_plate             464.9 ms  {'crop': True, ...}
  correct_illumination     947.7 ms  ...
  ...
```

If the detected count matches the ground-truth count, your install is
healthy. If not, something is wrong in your environment — open an issue
with the printed error and your `pip list` output.

---

## 3. Run on your own plate photo (3 minutes)

### a. Scaffold a starter pipeline

```bash
blenny init --out pipeline.yaml
```

This writes a `pipeline.yaml` that captures a sensible default
classical-CV workflow: load → detect plate → correct illumination →
threshold + watershed → measure → export CSV + annotated PNG. Open it
in any text editor and you'll see every step with comments next to the
parameters you're most likely to tune.

### b. Run it on one image

```bash
blenny run pipeline.yaml --input plate.jpg --output results/
```

You'll see one line per image processed:

```
Running pipeline pipeline.yaml on 1 image(s) -> results/
  [OK]   plate.jpg  colonies=147  (12.4s)
Done: 1/1 succeeded in 12.4s
```

### c. Look at what it produced

```
results/
├── config.yaml          ← the exact pipeline that ran (with all defaults filled in)
├── summary.csv          ← one row per image: count, status, time
└── plate/               ← per-image outputs, one folder per input
    ├── annotated.png    ← original image with detected colonies outlined and numbered
    ├── colonies.csv     ← one row per colony: area, centroid, intensity, ...
    └── provenance.json  ← every pipeline step that ran, with timings and params
```

**Open `results/plate/annotated.png` first.** The numbered red outlines
should land on real colonies. If most of them are right, you're done.
If most are wrong, the next section explains how to fix it.

---

## 4. Understanding the outputs

### `summary.csv`

A one-row-per-image roll-up of the batch. Useful for sanity checks across
many plates at a glance:

```csv
input,stem,status,colony_count,n_quality_flags,duration_s
plates/A.jpg,A,ok,147,0,12.4
plates/B.jpg,B,ok,89,1,11.9
plates/bad.jpg,bad,failed,,,0.3
```

### `<image>/colonies.csv`

One row per detected colony with measurements:

| column | meaning |
|---|---|
| `label` | unique integer id, matches the number drawn on the annotated image |
| `area_px` | colony area in pixels |
| `centroid_x`, `centroid_y` | pixel coordinates of the centroid |
| `equivalent_diameter_px` | diameter of a circle with the same area |
| `eccentricity` | 0 = perfect circle, → 1 = elongated |
| `mean_intensity` | average pixel intensity inside the colony (0–1 float) |
| `bbox_y0`, `bbox_x0`, `bbox_y1`, `bbox_x1` | bounding box |
| `touches_edge` | true if the colony touches the image edge (likely partially clipped) |
| `source` | the input file the row came from |

### `<image>/provenance.json`

The full audit trail for the run on this image:

- every step's name, params, and timing
- image metadata (dimensions, original size if it was resized, plate geometry)
- every quality flag that was raised, with severity and the step that raised it

Useful for: filing a bug report, reproducing a result a year later, or
explaining your methods in a paper.

### `config.yaml` (in the output directory root)

A snapshot of the pipeline that ran, with every default filled in and
placeholders resolved. **This is the resolved-for-the-first-image copy
for transparency**; to re-run the same batch, use your original
`pipeline.yaml`.

---

## 5. Batch mode (1 minute)

Quote the glob pattern so the shell doesn't expand it before Blenny sees it:

```bash
blenny run pipeline.yaml --input "plates/*.jpg" --output results/
```

Each input gets its own `results/<image-stem>/` directory. The
`{stem}`, `{output_dir}`, `{input}`, and `{name}` placeholders inside
`pipeline.yaml` are substituted per image:

```yaml
- name: export_csv
  params:
    output_path: "{output_dir}/{stem}/colonies.csv"
```

If one image fails (corrupt file, weird format, etc.), the rest keep
running — `summary.csv` records which ones succeeded and which failed,
and the `blenny run` command exits with a nonzero status if there were
any failures. Pass `--fail-fast` to stop on the first error instead.

---

## 6. Tuning the pipeline

Real lab images vary a lot, and the defaults are tuned for "reasonable
phone photo of a plate" — they won't be perfect for everything. The
`pipeline.yaml` file is yours to edit.

### See every parameter of every module

```bash
blenny modules
```

This prints every registered module with a one-line description and the
full list of parameters with their defaults. Use it as the canonical
reference instead of memorizing.

For machine consumption (e.g. building a config UI):

```bash
blenny modules --json
```

### Common adjustments

**Annotated image shows a ring of false "colonies" along the plate rim.**
The plate's bright reflective rim is being detected. Increase the rim
exclusion margin:

```yaml
- name: detect_plate
  params:
    margin_frac: 0.12     # default 0.08; try 0.10–0.15 for wide reflective rims
```

**Lots of false detections from pen marks or scratches.** The shape
filters in the segmenter drop very-elongated and irregular blobs.
Tighten them:

```yaml
- name: threshold_segment
  params:
    min_circularity: 0.8   # default 0.7; closer to 1 demands more circular shapes
    min_solidity: 0.9      # default 0.85; closer to 1 demands more compact shapes
```

If you instead have *real* elongated colonies (filaments, yeast tetrads,
etc.) and want to keep them, set these to `0` to disable the filters.

**Touching colonies are being merged into one.** The watershed step
splits them, but its sensitivity depends on colony size. Try lowering
`peak_min_distance`:

```yaml
- name: threshold_segment
  params:
    peak_min_distance: 3   # default 5; smaller = more aggressive splitting
```

**Phone photos are very slow.** By default, images larger than 2000 px
on the long side are downscaled before analysis (a one-time `info`
quality flag records this). To analyze at native resolution:

```yaml
- name: load_image
  params:
    max_dimension: null    # disable resize (slower, higher precision)
```

**Many small false positives, especially in noisy backgrounds.** Raise
the minimum colony area:

```yaml
- name: threshold_segment
  params:
    min_area: 25           # default 10 pixels
```

**Counts are way off and you want to see what each step is doing.**
Open `results/<image>/provenance.json` — the `quality_flags` and
per-step timings will usually point you at the culprit. If the plate
detection step shows a `plate_radius` that doesn't match the real plate,
plate detection is failing and downstream steps inherit the problem.

---

## 7. Quality flags: when to trust the count

Blenny raises *quality flags* on results that look unreliable. Always
glance at them before quoting a number in a paper.

Each flag has:
- `code` — short machine-readable id (e.g. `low_contrast`, `many_edge_touches`)
- `message` — human-readable explanation
- `severity` — `info`, `warning`, or `error`
- `step` — which pipeline step raised it

Common flags from the built-in modules:

| code | meaning | what to do |
|---|---|---|
| `image_resized` (info) | the loader downscaled a large image | usually fine; disable with `max_dimension: null` if you need exact pixel scale |
| `plate_not_found` (warning) | Hough circles didn't find a plate | try clearer/higher-contrast photos, or set `crop: false` and run on the whole frame |
| `many_edge_touches` (warning) | >10% of detected colonies touch the image edge | crop the input image so the plate sits clearly inside the frame |
| `no_objects` (warning) | the segmenter found nothing | usually a very dim or empty plate; check `correct_illumination` is producing visible signal |

The `summary.csv` includes a `n_quality_flags` column so you can spot
suspicious rows quickly across a big batch.

---

## 8. Doing the same thing from Python

If you want to drop into a notebook or a script:

```python
from blenny import Pipeline

# Load and run a YAML pipeline
pipe = Pipeline.from_yaml("pipeline.yaml")
result = pipe.run("plate.jpg")

print("colonies:", result.metadata["colony_count"])
print("flags:", [(f.code, f.message) for f in result.quality_flags])

# Per-colony rows
for row in result.measurements[:5]:
    print(row)
```

Or build a pipeline directly without YAML:

```python
from blenny import Pipeline

pipe = Pipeline.from_config([
    {"name": "load_image"},
    {"name": "detect_plate", "params": {"margin_frac": 0.07}},
    {"name": "correct_illumination", "params": {"radius": 25}},
    {"name": "threshold_segment", "params": {"roi_mask_key": "plate"}},
    {"name": "measure_colonies"},
])
result = pipe.run("plate.jpg")
```

The result is an `ImageData` object — see the [`pipeline/context.py`
docstrings](../src/blenny/pipeline/context.py) for the full field list
(`image`, `masks`, `measurements`, `metadata`, `provenance`, etc.).

---

## What to read next

- **[`design.md`](../design.md)** — the project's vision, scope, and
  guiding principles.
- **[`README.md`](../README.md)** — install + project status at a glance.
- **`blenny modules` output** — the canonical reference for what every
  built-in module does and what its parameters mean.

If you hit something that surprises you, please open an issue — the
project is in pre-alpha and your feedback is the best signal we have.
