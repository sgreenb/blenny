# Blenny — Bug Report

Audit performed on commit `65418ae` (branch `dev`, tag `pre-confidence-control` for the
pre-confidence-control baseline). Every item below was either reproduced by running the
code or verified by direct code-path analysis. No code changes were made during this
audit; this document exists so the bugs can be fixed in a later pass.

Severity guide:
- **HIGH** — breaks a feature outright, silently loses data, or produces wrong results.
- **MEDIUM** — wrong behaviour in a real (if less common) flow, or broken tooling.
- **LOW** — cosmetic, dead code, stale tests/comments, or edge cases.

---

## HIGH

### 1. GUI "Manual Circle" and "Manual Shape" modes crash with the default pipeline  ✅ FIXED
- **File:** `gui/app.py:404-406`
- **What happens:** In Manual Circle / Manual Shape mode the GUI injects
  `force_cy/force_cx/force_r` / `force_mask_path` into **both** `detect_plate` and
  `detect_facile` override entries. `FacileDetector.Params`
  (`src/blenny/modules/detect_facile.py`) has **none** of those fields, and `BlennyParams`
  uses `extra="forbid"`, so building the pipeline raises a pydantic `ValidationError`
  (`Extra inputs are not permitted`), which the GUI surfaces as a generic `st.error`.
  Since the default GUI pipeline (`pipeline_yolo_facile.yaml`) uses `detect_facile`,
  **both manual modes are completely broken out of the box.**
- **Reproduced:** `Pipeline.from_config([{"name": "detect_facile", "params": {"force_cy": 500}}])`
  → `ValidationError: force_cy … Extra inputs are not permitted`. Same for `force_mask_path`.
- **Fix direction:** either add the force-* params to `FacileDetector.Params`, or only apply
  manual overrides to steps that actually accept them.

  **Status: FIXED** — added `force_cy/force_cx/force_r/force_mask_path` to
  `FacileDetector.Params` with a `_apply_forced` path that bypasses detection and builds a
  single ROI (plus a plate mask) in the same shape the auto-detection ROI branch produces,
  so `sub_pipeline` and the GUI review flow work unchanged. Manual coordinates are scaled by
  `resize_scale` when the image was downscaled at load. Verified by running both GUI
  override paths end-to-end on a synthetic plate and with new tests in
  `tests/test_detect_facile.py`.

### 2. YOLO detector feeds the model RGB data that ultralytics treats as BGR (channel swap)  ✅ FIXED
- **File:** `src/blenny/modules/yolo_detector.py:76-80`
- **What happens:** `load_image` produces an **RGB** numpy array (PIL). `yolo_detector`
  passes that array straight to `model.predict(image, ...)`. In ultralytics 8.4.48,
  `LoadPilAndNumpy._single_check` returns numpy inputs **as-is** ("assumed BGR"), and
  `BasePredictor.preprocess` then does `im[..., ::-1]` (BGR→RGB). Net effect: the model
  receives the image with **red and blue channels swapped** — different from the colour
  distribution the model was trained on (training loads via cv2/BGR paths that convert
  correctly).
- **Reproduced:** by reading installed ultralytics source
  (`ultralytics.data.loaders.LoadPilAndNumpy`, `ultralytics.engine.predictor.BasePredictor.preprocess`).
- **Fix direction:** convert to BGR before calling predict
  (`image[..., ::-1]` / `cv2.cvtColor(image, cv2.COLOR_RGB2BGR)`), or pass the array
  through ultralytics' PIL loader path.

  **Status: FIXED** — `yolo_detector.segment` now converts RGB → BGR
  (`cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)`) before `predict`, converting float inputs to
  uint8 first (which also prevents 0–1 float images from being divided by 255 twice).
  Verified with a mock capturing the array passed to `predict` (pure-red RGB pixel arrives
  as BGR `[0,0,255]`); regression test added in `tests/test_yolo_detector.py`.

### 3. CSV exporter silently drops 14 measurement columns (incl. multiplicity, classification, bbox)  ✅ FIXED
- **File:** `src/blenny/modules/export_csv.py:52-97` (`preferred_order` whitelist +
  `extrasaction="ignore"`); same list duplicated in `src/blenny/cli/main.py` for
  `batch_colonies.csv`.
- **What happens:** the exporter only writes columns in its hard-coded `preferred_order`
  list. Columns produced by modules but absent from that list are **dropped from the CSV**,
  including:
  - `colony_count_estimate` — the multiplicity estimate. CSV row count therefore disagrees
    with `metadata["colony_count"]` whenever merged colonies are present (e.g. 11 CSV rows
    vs a count of 12).
  - `segment_label` — the only key that maps a CSV row back to the label mask used for
    annotation.
  - `classification` / `class_color` — the entire output of `classify_by_threshold` is lost.
  - `area_ppm`, `perimeter_px`, `equivalent_diameter_px`, `touches_edge`, `normalized_dist`,
    `zone`, `bbox_y0/x0/y1/x1` — the README explicitly promises "Area (px, ppm)" and
    "Bounding Box" in the CSV; they never appear.
- **Reproduced:** ran the classic pipeline (measure → multiplicity → classify interior →
  classify threshold → export CSV) on a synthetic plate; header contains 16 columns while
  measurement rows carry 30 keys. 14 keys missing from header.
- **Fix direction:** include all keys present in rows (or a richer explicit order), at least
  `colony_count_estimate` and `segment_label`.

  **Status: FIXED** — both `CSVExporter` and the CLI's `_write_batch_colonies_csv` now
  append every remaining row key (after the preferred order) by first appearance, so
  nothing is dropped. Verified: classic pipeline run now writes all 30 columns (was 16),
  including `colony_count_estimate`, `segment_label`, `classification`, and bbox columns.

### 4. GUI tuning sliders "Min Size (ppm)", "Min Circularity", "Interior Radius", "Max Eccentricity" have no effect  ✅ FIXED
- **File:** `gui/app.py:306-316` (controls) vs `gui/app.py:395-402` (overrides dict)
  vs `pipeline_yolo_facile.yaml` (template).
- **What happens:** the GUI maps these sliders to `threshold_segment` and
  `classify_by_interior` params. The default YOLO facile pipeline's `sub_pipeline` contains
  only `apply_exclusion_mask, yolo_detector, add_manual_colonies, measure_colonies,
  filter_by_id, export_annotated` — **none** of `threshold_segment` /
  `classify_by_interior` / `estimate_multiplicity`. Verified: applying the GUI override map
  to the template only touches `detect_facile` (radius_scale) and `yolo_detector`
  (conf_threshold). Users adjusting those four sliders see zero change in results.
- **Fix direction:** either add the corresponding steps to the shipped YOLO pipelines, or
  hide/disable the sliders when the active pipeline lacks the target module.

  **Status: FIXED** — both. `classify_by_interior` was added to every YOLO template and
  root pipeline (`count_colonies_yolo_facile[_grid]`, `count_colonies_multi`, and the
  corresponding root YAMLs), so the Interior Radius / Max Eccentricity sliders now apply.
  The GUI now inspects the active pipeline and only renders sliders whose target module
  exists (`threshold_segment` → Min Size / Min Circularity; `classify_by_interior` →
  Interior Radius / Max Eccentricity), so no slider can silently do nothing.
  NOTE: this adds active artifact rejection to CLI YOLO runs (previously YOLO mode had
  none) — intended per the README's core-feature list.

### 5. GUI checkboxes "Save debug images", "Generate annotated images", "Save as Subfolders" do nothing  ✅ FIXED
- **File:** `gui/app.py:317-319`
- **What happens:** `enable_debug`, `generate_annotated`, `save_subfolders` are read into
  variables but never referenced again anywhere in the file (verified by grep — each
  identifier appears exactly once). The analysis run
  (`gui/app.py:431-432`) never passes a `debug_dir`, never strips `export_annotated` steps,
  and never flattens output paths, so none of the three checkboxes affects the run.
- **Fix direction:** wire them up mirroring the CLI flags (`--debug-dir`,
  `--no-annotated-images`, `--flat`), or remove them from the UI.

  **Status: FIXED** — all three are now wired up in the GUI run block: unchecking
  "Generate annotated images" strips every `export_annotated` step (incl. inside
  `sub_pipeline`), unchecking "Save as Subfolders" flattens `{output_dir}/{stem}/` paths
  (same improved logic as the CLI `--flat` fix), and "Save debug images" passes a per-image
  `debug_dir` to the pipeline runner. Verified by replicating the strip/flatten logic on the
  template and checking the run call wiring.

### 6. GUI "Save/Update All results" writes multi-plate annotated images to the *old* output directory
- **File:** `src/blenny/modules/sub_pipeline.py:120` (copies `output_dir` into each
  sub-result at analysis time) + `sub_pipeline.py:274-283` (`export()` uses the stale
  sub-result path).
- **What happens:** at analysis time `SubPipeline.run` copies the parent's `output_dir`
  into every sub-result's metadata. When the GUI "Save/Update All results" button re-exports
  to a newly chosen directory, it swaps the **parent** `data.metadata["output_dir"]`
  (`gui/app.py:553-557`) but `SubPipeline.export` writes each per-plate annotated image
  using the **sub-result's** stale `output_dir` — i.e. to the original analysis folder, not
  the folder the user selected. Parent-level CSV/summary exports do land in the new folder.
- **Reproduced:** ran a 1-ROI `SubPipeline` with `output_dir=A`, then set parent metadata to
  `B` and called `sub.export()`; annotated image existed under `A/...` and not `B/...`.
- **Fix direction:** re-stamp `output_dir` on sub-results before re-exporting (e.g. in
  `SubPipeline.export`), or resolve paths from the parent context.

---

## MEDIUM

### 7. Four CLI tests are stale and fail on a clean checkout
- **File:** `tests/test_cli.py`
- **Failures (all reproduced on the clean baseline before any audit changes):**
  - `test_init_writes_to_default_file` — expects `"Wrote YOLO ML template to pipeline_yolo.yaml"`;
    code prints `"Wrote YOLO ML (Auto) template to pipeline_yolo.yaml"` (`src/blenny/cli/main.py`).
  - `test_run_single_image_with_template` — expects `out_dir/plate/log.txt`; the
    `count-colonies` template writes `{stem}_run_summary.txt` (`src/blenny/templates/count_colonies.yaml`).
  - `test_run_batch_with_glob` — same `log.txt` expectation; also expects `summary.csv` and
    `batch_log.txt` which the CLI no longer produces (it writes `batch_summary.csv`).
  - `test_run_keeps_going_after_per_image_failure` — reads `summary.csv` (old name).
- **Fix direction:** update assertions to current CLI output names/messages.

### 8. CLI `--flat` mangles multi-plate annotated filenames (duplicated stem)
- **File:** `src/blenny/cli/main.py:198-209` (`flatten_paths`)
- **What happens:** flattening blindly replaces `{output_dir}/{stem}/` with
  `{output_dir}/{stem}_`. For the multi-plate annotated path
  `{output_dir}/{stem}/{stem}_{plate_label}_annotated.png` this yields
  `{output_dir}/{stem}_{stem}_{plate_label}_annotated.png` (stem repeated).
- **Fix direction:** make the replacement pattern-aware (replace `{stem}/` with `_` only,
  or rebuild the flattened path more carefully).

### 9. Multi-plate counts ignore `colony_count_estimate`
- **File:** `src/blenny/modules/sub_pipeline.py:226-266`
- **What happens:** `colony_count` and `per_plate_counts` are computed as raw
  non-artifact **detection** counts (`+1` per row). Single-plate mode
  (`InteriorColonyClassifier.update_count`) sums `colony_count_estimate`. If a custom
  sub-pipeline includes `estimate_multiplicity`, single-plate and multi-plate runs of the
  same plate report different counts. (Not observable with the shipped YOLO templates,
  which don't include multiplicity in the sub-pipeline.)
- **Fix direction:** sum `colony_count_estimate` in `sub_pipeline` when present.

### 10. `ThresholdClassifier` doesn't validate `color` length → annotated export crashes
- **File:** `src/blenny/modules/classify_threshold.py:30` (`color: list[int] | None`)
  + `src/blenny/modules/export_annotated.py` (`rgb[b] = np.array(r["class_color"])`)
- **What happens:** a rule with a 2- or 4-element color list passes pydantic validation
  but crashes the annotated exporter with
  `ValueError: shape mismatch: value array of shape (2,) could not be broadcast ...`.
- **Reproduced:** `ThresholdClassifier(rules=[{'feature':'mean_v','min':0.5,'label':'x','color':[255,0]}])`
  then `AnnotatedImageExporter.render(...)` → `ValueError`.
- **Fix direction:** validate length-3 RGB in the `color` field (pydantic `Field(min_length=3, max_length=3)`).

### 11. Template drift: `blenny init` writes a different `pipeline_multi.yaml` than the committed one
- **Files:** `src/blenny/templates/count_colonies_multi.yaml` vs `pipeline_multi.yaml`
- **Diff:** grid `[2, 3]` vs `[3, 2]`, `max_subplate_dimension: 1280` vs `3200`, and the
  template carries an extra `detect_multi_plate.min_confidence_score: 0.20`. Running
  `blenny init` therefore produces a config that behaves differently from the repo's own
  pipeline file.
- **Fix direction:** reconcile the template and the root YAML (pick one as canonical).

### 12. `scripts/evaluate_labeled.py` references a directory that doesn't exist
- **File:** `scripts/evaluate_labeled.py:28-30` → `example_plates/labels.csv`
- **What happens:** the script hard-codes `REPO / "example_plates"`, which is not in the
  repository; the script fails immediately out of the box.
- **Fix direction:** point at an existing dataset dir or document the required layout.

---

## LOW

### 13. GUI dead code / unused constants
- `gui/app.py:33` `resize_default` and `gui/app.py:32` `max_dimension_default` are unused
  (the widgets hard-code 3200 etc.).
- `gui/app.py:322` `manual_exclude_ids` is initialised in session state but never used
  (the "filter_by_id" GUI path is unimplemented).

### 14. `detect_facile` all-outliers path produces no quality flag
- `src/blenny/modules/detect_facile.py` — after the size-consistency filter removes every
  circle, the code writes an all-True plate mask and returns without raising any flag, so
  the user gets no warning that detection collapsed.

### 15. `batch_colonies.csv` silently skipped when no measurements
- `src/blenny/cli/main.py:_write_batch_colonies_csv` returns early on an empty list without
  writing the file; failed/all-empty batches produce no `batch_colonies.csv` at all.

### 16. Debug output collisions in multi-plate mode
- `src/blenny/pipeline/debug.py` — sub-plates reuse the same step names and the writer's
  counter resets per sub-pipeline run, so debug images/masks from plate 2 overwrite plate 1's.

### 17. Weak test assertion
- `tests/test_classify_interior.py:test_iqr_multiplier_controls_strictness` asserts
  `strict_rejected or not lenient_rejected`, which can pass even when the classifier
  behaves identically at both multipliers.

### 18. Stale comment in test
- `tests/test_modules.py:test_image_file_loader_does_not_upscale_small_images` says
  "default max_dim=2000"; the actual default is `None`.

### 19. Stale local artifact in `gui_uploads/` (not tracked)
- `gui_uploads/` and `sandbox/` are already in `.gitignore` and contain **zero tracked
  files**, so nothing here is committed to the repo (an earlier draft of this audit claimed
  otherwise). The only residual point: a stale local `gui_uploads/uploaded_pipeline.yaml`
  still contains the now-removed duplicate top-level `export_annotated` step, so if a user
  re-selects that file in the GUI it produces an extra parent-level annotated export.
  Operational note only: delete the stale file; no repo change needed.

### 20. GUI override forces `crop: False` on `detect_plate`  ✅ FIXED
- `gui/app.py:397` unconditionally sets `crop: False` for `detect_plate`; a user-uploaded
  classic pipeline that sets `crop: true` would behave differently under the GUI than under
  the CLI.

  **Status: FIXED** — the GUI override now only sets `radius_scale` for `detect_plate` and
  `detect_facile`, leaving each pipeline's own `crop` setting intact.

### 21. GUI `data_editor` `edited_rows` may be re-applied on unrelated reruns
- `gui/app.py:522-545` reads `edited_rows` from widget state and applies them on every
  rerun until the underlying dataframe changes; generally idempotent, but worth confirming
  interactively that repeated application + `st.rerun()` cannot loop. (Low confidence —
  needs interactive verification.)

---

## Verified-fine (investigated, not bugs)

- `SubPipeline` inner exporters **do** run per sub-plate during `run()` (the inner
  `Pipeline.run` executes all steps incl. exporters); `SubPipeline.export()` is an extra
  re-export path used by the GUI Save button, not a missing path.
- `yolo_count_only.py` `result.save(filename=..., labels=False, conf=False)` is valid —
  kwargs forward to `Results.plot` in installed ultralytics (8.4.48).
- `detect_plate` on tiny images (8×8, 16×16) degrades gracefully to the all-True fallback
  mask with a `plate_not_found` flag (no crash).
- `remove_small_objects(min_size→max_size)` deprecation shim handles both skimage API
  generations.
- Multiplicity rounding (`int(ratio + 0.2)`), interior-classifier IQR guard, and registry
  duplicate-name rejection are all covered by passing tests.
- **`ml_training/` YAMLs use machine-specific absolute paths (e.g. `C:/Users/samgr/...`).**
  Not a bug: the whole `ml_training/` directory is in `.gitignore`, is not tracked by git
  (0 files), and is intentionally local-only tooling for this machine. Training configs are
  regenerated here rather than distributed.

---

## Incidental observation (not a bug in the code)

- `.gitignore` lists `pipeline_classic.yaml`, `pipeline_multi.yaml`, and `pipeline_yolo.yaml`
  as ignored, but `pipeline_classic.yaml` and `pipeline_multi.yaml` are actually **tracked**
  (they were committed before the ignore rules were added; git ignores don't untrack
  files). `pipeline_yolo.yaml` and `pipeline_yolo_facile_grid.yaml` are genuinely untracked
  (generated by `blenny init`). Inconsistent, but harmless — just means edits to the two
  tracked YAMLs show up in git.
