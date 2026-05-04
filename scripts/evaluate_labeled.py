"""Quantitative evaluation of the Blenny pipeline against manually labeled plates.

Reads ``example_plates/labels.csv``, runs the default pipeline on every image
(caching results in ``example_plates/output/``), then compares detected colony
counts against manual ground-truth counts and writes:

    example_plates/output/REPORT.md   — human-readable results and breakdowns
    example_plates/output/results.csv — per-image numbers for further analysis

Usage:
    python scripts/evaluate_labeled.py            # use cached runs
    python scripts/evaluate_labeled.py --force    # re-run everything
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
IN_DIR = REPO / "example_plates"
OUT_DIR = IN_DIR / "output"
LABELS = IN_DIR / "labels.csv"
REPORT = OUT_DIR / "REPORT.md"
RESULTS = OUT_DIR / "results.csv"

# ---------------------------------------------------------------------------
# Pipeline config (mirrors the shipped count_colonies template).
# Keeping it here so the script is self-contained and version-controlled.
# ---------------------------------------------------------------------------
# Omit tunable params so module defaults apply automatically.
# When a default changes, re-running with --force reflects it here too.
PIPELINE_STEPS = [
    {"name": "load_image", "params": {"max_dimension": 2000}},
    {"name": "detect_plate", "params": {"crop": True, "radius_expand_frac": 0.05}},
    {"name": "correct_illumination", "params": {"radius": 25}},
    {"name": "threshold_segment", "params": {"roi_mask_key": "plate", "split_touching": True}},
    {"name": "measure_colonies"},
    {"name": "estimate_multiplicity"},
    {"name": "classify_by_interior"},
    {"name": "export_annotated", "params": {"output_path": ""}},  # filled per-image
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bool(val: object) -> bool | None:
    """Parse True/False/yes/no/1/0 from CSV strings; None if blank."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    return None


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n // d}%)" if d else "—"


def _mae(errors: list[float]) -> str:
    return f"{np.mean(np.abs(errors)):.1f}" if errors else "—"


def _median_ae(errors: list[float]) -> str:
    return f"{np.median(np.abs(errors)):.1f}" if errors else "—"


def _rmse(errors: list[float]) -> str:
    return f"{np.sqrt(np.mean(np.array(errors) ** 2)):.1f}" if errors else "—"


def _within(errors: list[float], manuals: list[int], pct: float) -> str:
    if not errors:
        return "—"
    n = sum(1 for e, m in zip(errors, manuals, strict=True) if m > 0 and abs(e) / m <= pct)
    return _pct(n, len(errors))


def _metrics_block(errors: list[float], manuals: list[int], n: int) -> list[str]:
    """Return a list of markdown bullet lines for a set of error values."""
    if not errors:
        return ["- n = 0 (no data)"]
    abs_errors = [abs(e) for e in errors]
    lines = [
        f"- n = {n}",
        f"- Mean absolute error:   **{np.mean(abs_errors):.1f}** colonies",
        f"- Median absolute error: **{np.median(abs_errors):.1f}** colonies",
        f"- RMSE:                  **{_rmse(errors)}** colonies",
        f"- Within ±10%:           **{_within(errors, manuals, 0.10)}**",
        f"- Within ±20%:           **{_within(errors, manuals, 0.20)}**",
        f"- Within ±30%:           **{_within(errors, manuals, 0.30)}**",
    ]
    return lines


# ---------------------------------------------------------------------------
# Run one image through the pipeline
# ---------------------------------------------------------------------------


def run_image(img_path: Path, out_subdir: Path) -> dict:
    """Run the pipeline; return a result dict.  Raises on hard failure."""
    import copy

    from blenny import Pipeline

    out_subdir.mkdir(parents=True, exist_ok=True)

    steps = copy.deepcopy(PIPELINE_STEPS)
    for step in steps:
        if step["name"] == "export_annotated":
            step["params"]["output_path"] = str(out_subdir / "annotated.png")

    pipe = Pipeline.from_config(steps)
    data = pipe.run(img_path)

    count = data.metadata.get("colony_count", 0)
    artifact_count = data.metadata.get("artifact_count", 0)
    flags = [f.code for f in data.quality_flags]
    plate_found = "plate_not_found" not in flags
    coverage = data.metadata.get("colony_coverage_frac")

    # Write a summary so the cache-check works
    global_stats = data.metadata.get("detection_global_stats", {})
    summary = (
        f"image:          {img_path.name}\n"
        f"colony count:   {count}\n"
        f"artifact count: {artifact_count}\n"
        f"plate_found:    {plate_found}\n"
        f"coverage_frac:  {coverage}\n"
        f"quality_flags:  {', '.join(flags) if flags else '(none)'}\n"
        f"global_stats:   {global_stats}\n"
    )
    (out_subdir / "summary.txt").write_text(summary)

    return {
        "detected": count,
        "artifact_count": artifact_count,
        "plate_found": plate_found,
        "quality_flags": flags,
        "coverage_frac": coverage,
    }


def load_cached(out_subdir: Path) -> dict | None:
    """Try to read a previous run's summary.txt.  Returns None if not found."""
    p = out_subdir / "summary.txt"
    if not p.exists():
        return None
    info: dict = {}
    for line in p.read_text().splitlines():
        if line.startswith("colony count:"):
            info["detected"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("artifact count:"):
            info["artifact_count"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("plate_found:"):
            info["plate_found"] = line.split(":", 1)[1].strip() == "True"
        elif line.startswith("coverage_frac:"):
            raw = line.split(":", 1)[1].strip()
            info["coverage_frac"] = float(raw) if raw != "None" else None
        elif line.startswith("quality_flags:"):
            raw = line.split(":", 1)[1].strip()
            info["quality_flags"] = [] if raw == "(none)" else raw.split(", ")
    info.setdefault("artifact_count", 0)
    info.setdefault("coverage_frac", None)
    return info if "detected" in info else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ISSUE_FLAGS = [
    "has_pen_text",
    "has_shadow",
    "has_overlapping_colonies",
    "has_rim_artifacts",
    "uneven_illumination",
    "poor_focus",
]

ISSUE_LABELS = {
    "has_pen_text": "Pen text on plate/lid",
    "has_shadow": "Shadow across plate",
    "has_overlapping_colonies": "Overlapping / touching colonies",
    "has_rim_artifacts": "Rim artifacts",
    "uneven_illumination": "Uneven illumination",
    "poor_focus": "Poor focus",
}


def main() -> int:
    # ------------------------------------------------------------------
    # 0. Imports & args
    # ------------------------------------------------------------------
    import csv

    force = "--force" in sys.argv

    if not LABELS.exists():
        print(f"Labels file not found: {LABELS}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Read labels
    # ------------------------------------------------------------------
    with LABELS.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print("labels.csv is empty.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # 2. Run pipeline (or load cache)
    # ------------------------------------------------------------------
    results: list[dict] = []

    for row in rows:
        fname = row["filename"].strip()
        img = IN_DIR / fname
        stem = Path(fname).stem
        sub = OUT_DIR / stem

        if not img.exists():
            print(f"  [SKIP] {fname}: file not found in example_plates/")
            results.append(
                {
                    **row,
                    "detected": None,
                    "status": "missing",
                    "plate_found": None,
                    "quality_flags": [],
                }
            )
            continue

        cached = None if force else load_cached(sub)
        if cached:
            print(f"  [cached] {fname}: {cached['detected']} colonies")
            results.append({**row, **cached, "status": "ok"})
            continue

        print(f"  [run]    {fname} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            run_result = run_image(img, sub)
            elapsed = time.perf_counter() - t0
            print(f"{run_result['detected']} colonies ({elapsed:.1f}s)")
            results.append({**row, **run_result, "status": "ok"})
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"FAILED ({elapsed:.1f}s): {e}")
            traceback.print_exc()
            (sub / "FAILED.txt").write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")
            results.append(
                {
                    **row,
                    "detected": None,
                    "status": "failed",
                    "plate_found": None,
                    "quality_flags": [],
                }
            )

    # ------------------------------------------------------------------
    # 3. Write results.csv
    # ------------------------------------------------------------------
    fieldnames = [
        *rows[0].keys(),
        "detected",
        "error",
        "abs_error",
        "pct_error",
        "plate_found",
        "quality_flags",
        "status",
    ]
    with RESULTS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            try:
                manual = int(r.get("manual_count") or 0)
                detected = int(r["detected"]) if r["detected"] is not None else None
                countable = _bool(r.get("countable"))
                if detected is not None and countable:
                    error = detected - manual
                    abs_error = abs(error)
                    pct_error = round(abs_error / manual * 100, 1) if manual else None
                else:
                    error = abs_error = pct_error = None
            except (TypeError, ValueError):
                error = abs_error = pct_error = None
            writer.writerow(
                {
                    **r,
                    "error": error,
                    "abs_error": abs_error,
                    "pct_error": pct_error,
                    "quality_flags": ", ".join(r.get("quality_flags") or []),
                }
            )

    # ------------------------------------------------------------------
    # 4. Build REPORT.md
    # ------------------------------------------------------------------
    # Separate countable vs blank plates
    countable_rows = [
        r
        for r in results
        if _bool(r.get("countable")) is True
        and r.get("detected") is not None
        and r.get("status") == "ok"
    ]
    blank_rows = [r for r in results if _bool(r.get("countable")) is False]
    failed_rows = [r for r in results if r.get("status") in ("failed", "missing")]

    def _errors_manuals(subset: list[dict]) -> tuple[list[float], list[int]]:
        errors, manuals = [], []
        for r in subset:
            try:
                m = int(r["manual_count"])
                d = int(r["detected"])
                errors.append(float(d - m))
                manuals.append(m)
            except (TypeError, ValueError):
                pass
        return errors, manuals

    all_errors, all_manuals = _errors_manuals(countable_rows)

    md: list[str] = []

    md += [
        "# Blenny — Evaluation Report",
        "",
        "Pipeline: default colony-counting config (classical CV)  ",
        "Dataset:  `example_plates/labels.csv`  ",
        f"Images:   {len(results)} total "
        f"({len(countable_rows)} countable, "
        f"{len(blank_rows)} blank/uncountable, "
        f"{len(failed_rows)} failed/missing)",
        "",
        "---",
        "",
    ]

    # -- Overall metrics --
    md += ["## Overall metrics", ""]
    md += _metrics_block(all_errors, all_manuals, len(countable_rows))
    md += [""]

    # -- Per-image table --
    md += [
        "## Per-image results",
        "",
        "| Image | Manual | Detected | Error | % error | Coverage | Flags | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in results:
        fname = r["filename"]
        stem = Path(fname).stem
        countable = _bool(r.get("countable"))
        manual = r.get("manual_count", "")
        detected = r.get("detected")
        pf = "" if r.get("plate_found") else " no-plate"
        cov = r.get("coverage_frac")
        cov_str = f"{float(cov):.1%}" if cov is not None else "—"
        flag_codes = [c for c in (r.get("quality_flags") or []) if c not in ("image_resized",)]
        flags_str = ", ".join(f"`{c}`" for c in flag_codes) if flag_codes else "—"

        if detected is None:
            md.append(
                f"| `{fname}` | {manual} | **{r['status'].upper()}** | — | — | — | — | {r.get('notes', '')[:60]} |"
            )
            continue

        if not countable:
            md.append(
                f"| `{fname}` *(blank)* | 0 | {detected} | — | — | {cov_str} | {flags_str}{pf} | {r.get('notes', '')[:60]} |"
            )
            continue

        try:
            m = int(manual)
            d = int(detected)
            err = d - m
            ae = abs(err)
            pe = f"{ae / m * 100:.0f}%" if m else "—"
            sign = "+" if err >= 0 else ""
            md.append(
                f"| [{fname}]({stem}/annotated.png) | {m} | {d} | {sign}{err} | {pe} | {cov_str} | {flags_str}{pf} | {r.get('notes', '')[:60]} |"
            )
        except (TypeError, ValueError):
            md.append(
                f"| `{fname}` | {manual} | {detected} | — | — | {cov_str} | {flags_str}{pf} | |"
            )

    md += [""]

    # -- Blank plate check --
    if blank_rows:
        md += ["## Blank / uncountable plates (false-positive check)", ""]
        md += ["These plates have `countable = FALSE`. Ideally detected ≈ 0.", ""]
        for r in blank_rows:
            d = r.get("detected")
            fname = r["filename"]
            stem = Path(fname).stem
            ann = (
                f"[view]({stem}/annotated.png)"
                if (OUT_DIR / stem / "annotated.png").exists()
                else ""
            )
            md.append(f"- `{fname}`: detected **{d}** colonies {ann}")
        md += [""]

    # -- Breakdown by source --
    md += ["## Breakdown by image source", ""]
    for src in sorted({r.get("image_source", "") for r in countable_rows}):
        subset = [r for r in countable_rows if r.get("image_source", "") == src]
        errs, mans = _errors_manuals(subset)
        md += [f"### {src} (n={len(subset)})", ""]
        md += _metrics_block(errs, mans, len(subset))
        md += [""]

    # -- Breakdown by issue flag --
    md += ["## Breakdown by issue flag", ""]
    md += ["For each flag: images *with* the issue vs. images *without* it.", ""]

    for flag in ISSUE_FLAGS:
        label = ISSUE_LABELS[flag]
        with_flag = [r for r in countable_rows if _bool(r.get(flag)) is True]
        without_flag = [r for r in countable_rows if _bool(r.get(flag)) is False]
        if not with_flag and not without_flag:
            continue

        md += [f"### {label}", ""]
        for group_name, group in [("With", with_flag), ("Without", without_flag)]:
            errs, mans = _errors_manuals(group)
            if not errs:
                md.append(f"**{group_name}** (n={len(group)}): no data")
            else:
                ae = np.mean(np.abs(errs))
                md.append(f"**{group_name}** (n={len(group)}): MAE = {ae:.1f} colonies")
        md += [""]

    # -- Breakdown by density --
    md += ["## Breakdown by colony density", ""]
    for density in ["sparse", "moderate", "dense", "TNTC"]:
        subset = [
            r for r in countable_rows if str(r.get("colony_density", "")).lower() == density.lower()
        ]
        if not subset:
            continue
        errs, mans = _errors_manuals(subset)
        md += [f"### {density.capitalize()} (n={len(subset)})", ""]
        md += _metrics_block(errs, mans, len(subset))
        md += [""]

    # -- Failed --
    if failed_rows:
        md += ["## Failed / missing images", ""]
        for r in failed_rows:
            md.append(f"- `{r['filename']}`: {r.get('status', '?')} — {r.get('notes', '')[:80]}")
        md += [""]

    md += [
        "---",
        "",
        "*Generated by `scripts/evaluate_labeled.py`. "
        "Re-run with `--force` to refresh all pipeline results.*",
    ]

    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {REPORT}")
    print(f"Wrote {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
