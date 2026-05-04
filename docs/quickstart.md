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

## 2. Run on your own plate photo

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
├── reproducible_config.yaml   ← the exact pipeline that ran (all defaults filled in)
└── plate/                     ← per-image outputs, one folder per input stem
    ├── annotated.png          ← original image with detected colonies outlined and numbered
    ├── colonies.csv           ← one row per colony: ID, coordinates, area, RGB/HSV
    └── log.txt                ← human-readable report with counts, flags, and per-colony table
```

**Open `results/plate/annotated.png` first.** Red outlines are counted colonies.
Magenta outlines are detections that were rejected as artifacts (rim fragments, etc.).
Numbers match the IDs in `colonies.csv`.

If most outlines land on real colonies, you're done. If most are wrong, the
tuning section below explains how to fix it.

### d. Optional outputs

```bash
# Save a full audit trail (params, timings, quality flags) per image
blenny run pipeline.yaml -i plate.jpg -o results/ --provenance

# Force batch summary files even for a single image
blenny run pipeline.yaml -i plate.jpg -o results/ --summary

# Save intermediate images from every pipeline step for debugging
blenny run pipeline.yaml -i plate.jpg -o results/ --debug-dir debug/
```

---

## 3. Understanding the outputs

### `<image>/colonies.csv`

One row per detected colony:

| column | meaning |
|---|---|
| `label` | unique integer ID, matches the number drawn on the annotated image |
| `centroid_x`, `centroid_y` | pixel coordinates of the colony centre |
| `area_px` | colony area in pixels |
| `mean_r`, `mean_g`, `mean_b` | mean RGB values (0–255) |
| `mean_h`, `mean_s`, `mean_v` | mean HSV values (0–1 float) |
| `is_artifact` | `True` if the colony was rejected as a rim/noise artifact |

### `<image>/log.txt`

A human-readable report containing:
- Count statistics (total colonies, artifacts found, size distribution)
- Quality flags with their severity and explanations
- A full per-colony table (ID, X, Y, Area, RGB, HSV, Type)
- The pipeline provenance (which steps ran, in order)

### `summary.csv` (batch runs only)

One row per image:

```csv
input,stem,status,colony_count,n_quality_flags,flag_codes,duration_s
plates/A.jpg,A,ok,147,0,,12.4
plates/B.jpg,B,ok,89,1,low_plate_confidence,11.9
```

### `reproducible_config.yaml`

A snapshot of the pipeline that ran with every default filled in. Useful
for sharing your exact analysis settings with a collaborator or reviewer.

### `provenance.json` (opt-in, `--provenance`)

The full audit trail for a single image:
- every step's name, params, and wall-clock timing
- image metadata (dimensions, plate geometry)
- every quality flag that was raised, with the step that raised it
- full per-colony measurements

---

## 4. Batch mode

Quote the glob pattern so the shell doesn't expand it before Blenny sees it:

```bash
blenny run pipeline.yaml --input "plates/*.jpg" --output results/
```

Each input gets its own `results/<stem>/` directory. A `summary.csv` and
`batch_log.txt` are automatically written to the output root.

If one image fails (corrupt file, weird format, etc.), the rest keep
running. Pass `--fail-fast` to stop on the first error instead.

---

## 5. Tuning the pipeline

Real lab images vary a lot, and the defaults are tuned for "reasonable
phone photo of a plate." The `pipeline.yaml` file is yours to edit.

### See every parameter of every module

```bash
blenny modules
```

This prints every registered module with a one-line description and the
full list of parameters with their defaults.

For machine consumption:

```bash
blenny modules --json
```

### Common adjustments

**A ring of false "colonies" appears along the plate rim.**
The plate's bright reflective rim is being detected. Increase the rim
exclusion margin:

```yaml
- name: detect_plate
  params:
    margin_frac: 0.12     # default 0.08; try 0.10–0.15 for wide reflective rims
```

**Colonies near the edge of the plate are missed.**
The photo was taken at a slight angle so the plate looks elliptical, and
the circle fitter chose a circle smaller than the actual plate:

```yaml
- name: detect_plate
  params:
    radius_expand_frac: 0.10   # default 0.05; raise for tilted shots
```

**Real edge colonies are being wrongly rejected as artifacts.**
The `classify_by_interior` module compares edge-zone detections against
interior colonies. Widen the "safe" interior zone:

```yaml
- name: classify_by_interior
  params:
    interior_radius_frac: 0.90   # default 0.85; higher = more permissive at edge
    iqr_multiplier: 3.0          # default 2.0; higher = wider acceptable size range
```

**Lots of false detections from pen marks or scratches.**

```yaml
- name: threshold_segment
  params:
    min_circularity: 0.8   # default 0.7; closer to 1 demands more circular shapes
    min_solidity: 0.9      # default 0.85; closer to 1 demands more compact shapes
```

**Touching colonies are merged into one.**

```yaml
- name: threshold_segment
  params:
    peak_min_distance: 3   # default is scale-aware; smaller = more aggressive splitting
```

**Phone photos are slow to process.**
Images are loaded at native resolution by default. If speed is a concern,
you can downscale large phone photos:

```yaml
- name: load_image
  params:
    max_dimension: 2000    # downscale longest side to 2000 px (faster, less precise)
```

**Many small false positives.**

```yaml
- name: threshold_segment
  params:
    min_area: 25           # default 10 pixels
```

### One-off overrides from the command line

You can override any parameter without editing the YAML:

```bash
blenny run pipeline.yaml -i plate.jpg -o results/ \
  -v detect_plate.margin_frac=0.12 \
  -v classify_by_interior.interior_radius_frac=0.90
```

---

## 6. Quality flags

Blenny raises quality flags on results that look unreliable. Always
glance at them before quoting a count in a paper. They appear in
`log.txt` and (as `flag_codes`) in `summary.csv`.

| code | severity | meaning | what to do |
|---|---|---|---|
| `plate_not_found` | warning | Hough circles didn't find a plate | try clearer photos, or set `crop: false` |
| `low_plate_confidence` | warning | Best circle fit scored below threshold | inspect `annotated.png`; may still be correct |
| `many_low_circularity_rejected` | warning | Many non-circular objects removed | may indicate rim contamination or image noise |
| `many_edge_touches` | warning | >10% of colonies touch the image edge | ensure the plate sits clearly inside the frame |
| `no_objects` | warning | Segmenter found nothing | check illumination correction is producing visible signal |
| `suspect_high_count` | warning | Count > 600 or coverage > 50% of plate | likely many false positives; tighten shape filters |
| `plate_likely_empty` | warning | Detection set looks like noise | plate may be blank; strict shape filter was applied |
| `multiplicity_estimated` | info | Some detections tagged as merged colonies | check `xN` labels on annotated image |
| `artifacts_removed` | info | Edge detections rejected by interior classifier | inspect magenta outlines in `annotated.png` |

---

## 7. Interactive GUI

```bash
blenny gui
```

The GUI provides point-and-click access to everything above. After a run:

1. The **annotated image** appears immediately (red = colony, magenta = artifact).
2. The **Interactive Review** table lets you check or uncheck "Artifact?" on any row.
3. The image and colony count update **instantly** — no rerunning required.
4. Click **"Save Reviewed Results"** to write the final `colonies.csv` and `log.txt`.

Sidebar sliders correspond to the most commonly tuned parameters:
**Plate Rim Margin**, **Min Colony Area**, **Min Circularity**, and
**Interior Radius Frac**. Sliders respect values loaded from a
`reproducible_config.yaml` so you can restore a previous run's settings.

---

## 8. Doing the same thing from Python

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

Or build a pipeline without YAML:

```python
from blenny import Pipeline

pipe = Pipeline.from_config([
    {"name": "load_image"},
    {"name": "detect_plate", "params": {"margin_frac": 0.08}},
    {"name": "correct_illumination"},
    {"name": "threshold_segment", "params": {"roi_mask_key": "plate"}},
    {"name": "measure_colonies"},
    {"name": "classify_by_interior"},
])
result = pipe.run("plate.jpg")
```

The result is an `ImageData` object — see
[`pipeline/context.py`](../src/blenny/pipeline/context.py) for the full
field list (`image`, `masks`, `measurements`, `metadata`, `provenance`, etc.).

---

## What to read next

- **[`design.md`](../design.md)** — the project's vision, scope, and guiding principles.
- **[`README.md`](../README.md)** — install and project status at a glance.
- **`blenny modules`** — the canonical reference for every built-in module and its parameters.

If you hit something that surprises you, please open an issue — the project is
in pre-alpha and your feedback is the best signal we have.
