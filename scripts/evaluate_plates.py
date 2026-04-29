"""Run the current default pipeline on every image in ``example_plates/``
and dump a directory of diagnostics per image.

Intentionally NOT a test and NOT tuned per-image. The point is to see how
the current defaults behave on real data so we can decide what to fix.

Output layout (under ``example_plates/output/<image-stem>/``):
    01_input.jpg                  thumbnail of the input
    02_plate_overlay.jpg          input with detected plate circle drawn
    03_cropped.jpg                image after PlateDetector cropped it
    04_illumination_corrected.jpg corrected gray image, contrast-stretched
    05_segmentation.jpg           colored label overlay
    06_annotated.jpg              final numbered overlay (full resolution)
    measurements.csv              per-colony rows with provenance
    summary.txt                   timings, count, quality flags
And a top-level ``output/INDEX.md`` linking everything together.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage import color
from skimage.color import label2rgb

from blenny import ImageData
from blenny.modules import (
    AnnotatedImageExporter,
    ColonyMeasurer,
    CSVExporter,
    IlluminationCorrection,
    ImageFileLoader,
    PlateDetector,
    ThresholdSegmenter,
)

REPO = Path(__file__).resolve().parent.parent
IN_DIR = REPO / "example_plates"
OUT_DIR = IN_DIR / "output"
THUMB_MAX = 1200  # max dimension for diagnostic JPEGs


# --- helpers ----------------------------------------------------------------


def to_uint8_for_display(arr: np.ndarray) -> np.ndarray:
    """Min-max stretch a float array into uint8 for visual inspection."""
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


def save_thumb(arr: np.ndarray, path: Path, quality: int = 85) -> None:
    im = Image.fromarray(arr)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.thumbnail((THUMB_MAX, THUMB_MAX))
    im.save(path, quality=quality)


def draw_plate_overlay(rgb: np.ndarray, cy: int, cx: int, r: int) -> np.ndarray:
    im = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(im)
    width = max(3, int(min(rgb.shape[:2]) / 250))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0), width=width)
    d.line([(cx - 20, cy), (cx + 20, cy)], fill=(255, 0, 0), width=width)
    d.line([(cx, cy - 20), (cx, cy + 20)], fill=(255, 0, 0), width=width)
    return np.asarray(im)


# --- per-image pipeline with intermediates ----------------------------------


def evaluate(image_path: Path, out_subdir: Path) -> tuple[ImageData, list[str]]:
    out_subdir.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    t_total = time.perf_counter()

    data = ImageData(source=str(image_path))

    # 1. Load
    t0 = time.perf_counter()
    ImageFileLoader().run(data)
    log.append(
        f"load_image           {(time.perf_counter() - t0) * 1000:8.0f} ms  "
        f"shape={data.image.shape}"
    )
    save_thumb(data.image, out_subdir / "01_input.jpg")

    # 2. Plate detect
    t0 = time.perf_counter()
    PlateDetector().run(data)
    dt = (time.perf_counter() - t0) * 1000
    pc = data.metadata.get("plate_center")
    pr = data.metadata.get("plate_radius")
    log.append(
        f"detect_plate         {dt:8.0f} ms  center={pc} radius={pr} "
        f"cropped_shape={data.image.shape}"
    )
    if pc is not None and pr is not None:
        overlay = draw_plate_overlay(data.original_image, pc[0], pc[1], pr)
        save_thumb(overlay, out_subdir / "02_plate_overlay.jpg")
    save_thumb(data.image, out_subdir / "03_cropped.jpg")

    # 3. Illumination correction
    t0 = time.perf_counter()
    IlluminationCorrection().run(data)
    log.append(f"correct_illumination {(time.perf_counter() - t0) * 1000:8.0f} ms")
    save_thumb(to_uint8_for_display(data.image), out_subdir / "04_illumination_corrected.jpg")

    # 4. Threshold + segment
    t0 = time.perf_counter()
    ThresholdSegmenter().run(data)
    labels = data.masks["objects"]
    n_labels = int(labels.max())
    log.append(
        f"threshold_segment    {(time.perf_counter() - t0) * 1000:8.0f} ms  n_labels={n_labels}"
    )
    if n_labels > 0:
        base = data.artifacts.get("pre_illumination", data.image)
        base_gray = color.rgb2gray(base) if base.ndim == 3 else base.astype(float)
        rgb_label = label2rgb(labels, image=base_gray, bg_label=0, alpha=0.55, kind="overlay")
        save_thumb((rgb_label * 255).astype(np.uint8), out_subdir / "05_segmentation.jpg")
    else:
        save_thumb(to_uint8_for_display(data.image), out_subdir / "05_segmentation.jpg")

    # 5. Measure
    t0 = time.perf_counter()
    ColonyMeasurer().run(data)
    log.append(
        f"measure_colonies     {(time.perf_counter() - t0) * 1000:8.0f} ms  "
        f"rows={len(data.measurements)}"
    )

    # 6. Final annotated full-res output (no thumbnail; user wants to zoom)
    AnnotatedImageExporter(output_path=str(out_subdir / "06_annotated.jpg")).run(data)
    CSVExporter(
        output_path=str(out_subdir / "measurements.csv"),
        include_provenance=True,
    ).run(data)

    total_ms = (time.perf_counter() - t_total) * 1000
    summary = [
        f"image:        {image_path.name}",
        f"input shape:  {data.metadata.get('image_shape')}",
        f"colony count: {data.metadata.get('colony_count', '?')}",
        f"total time:   {total_ms:.0f} ms",
        "",
        "step timings:",
        *(f"  {line}" for line in log),
        "",
        "quality flags:",
    ]
    if data.quality_flags:
        summary.extend(
            f"  [{f.severity}] {f.code} (in {f.step}): {f.message}" for f in data.quality_flags
        )
    else:
        summary.append("  (none)")

    (out_subdir / "summary.txt").write_text("\n".join(summary) + "\n")
    return data, log


# --- driver -----------------------------------------------------------------


def main() -> int:
    if not IN_DIR.exists():
        print(f"No example_plates directory at {IN_DIR}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    images = sorted(p for p in IN_DIR.iterdir() if p.suffix.lower() in exts and p.is_file())
    if not images:
        print(f"No images found in {IN_DIR}", file=sys.stderr)
        return 1

    skip_done = "--force" not in sys.argv
    fresh = 0
    cached = 0
    failures = 0
    for img in images:
        name = img.stem.replace(" ", "_")
        out_subdir = OUT_DIR / name
        done_marker = out_subdir / "summary.txt"
        if skip_done and done_marker.exists():
            print(f"=== {img.name} === (cached, skipping; pass --force to redo)", flush=True)
            cached += 1
            continue
        print(f"\n=== {img.name} ===", flush=True)
        t0 = time.perf_counter()
        try:
            evaluate(img, out_subdir)
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  FAILED after {elapsed:.0f} ms: {type(e).__name__}: {e}")
            traceback.print_exc()
            (out_subdir / "FAILED.txt").parent.mkdir(parents=True, exist_ok=True)
            (out_subdir / "FAILED.txt").write_text(f"{type(e).__name__}: {e}\n")
            failures += 1
            continue
        fresh += 1
        print(f"  done in {(time.perf_counter() - t0) * 1000:.0f} ms")

    _build_index(images)
    print(f"\nWrote {OUT_DIR / 'INDEX.md'}")
    print(f"fresh={fresh}  cached={cached}  failures={failures}  total={len(images)}")
    return 0


def _parse_summary(path: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    flags: list[str] = []
    in_flags = False
    for line in path.read_text().splitlines():
        if line.startswith("colony count:"):
            info["count"] = line.split(":", 1)[1].strip()
        elif line.startswith("input shape:"):
            info["shape"] = line.split(":", 1)[1].strip()
        elif line.startswith("total time:"):
            info["total_time"] = line.split(":", 1)[1].strip()
        elif line.strip() == "quality flags:":
            in_flags = True
        elif in_flags:
            s = line.strip()
            if s and s != "(none)":
                flags.append(s)
    info["flags"] = "\n".join(flags)
    return info


def _build_index(images: list[Path]) -> None:
    index = [
        "# Plate evaluation \u2014 default pipeline, no per-image tuning",
        "",
        "For each image, look at the annotated output and the diagnostic",
        "intermediates and call out what's wrong. We tune *after* this, not before.",
        "",
        f"Total images: **{len(images)}**",
        "",
        "| Image | Input shape | Detected | # flags | Time |",
        "| --- | --- | --- | --- | --- |",
    ]
    details: list[str] = []
    for img in images:
        name = img.stem.replace(" ", "_")
        sub = OUT_DIR / name
        summary_path = sub / "summary.txt"
        failed_path = sub / "FAILED.txt"
        if failed_path.exists():
            err = failed_path.read_text().strip()
            index.append(f"| `{img.name}` | \u2014 | **FAILED** | \u2014 | \u2014 |")
            details.append(f"## `{img.name}` \u2014 FAILED\n\n```\n{err}\n```\n\n---\n")
            continue
        if not summary_path.exists():
            index.append(f"| `{img.name}` | (not yet processed) | \u2014 | \u2014 | \u2014 |")
            continue
        info = _parse_summary(summary_path)
        flag_lines = info["flags"].splitlines()
        index.append(
            f"| `{img.name}` | `{info.get('shape', '?')}` | **{info.get('count', '?')}** | "
            f"{len(flag_lines)} | {info.get('total_time', '?')} |"
        )
        sec = [f"## `{img.name}`", ""]
        sec.append(f"- Detected colonies: **{info.get('count', '?')}**")
        sec.append(f"- Input shape: `{info.get('shape', '?')}`")
        sec.append(f"- Total time: {info.get('total_time', '?')}")
        if flag_lines:
            sec.append("- Quality flags:")
            sec.extend(f"  - {fl}" for fl in flag_lines)
        sec.extend(
            [
                "",
                "**Final annotated:**",
                "",
                f"![annotated]({name}/06_annotated.jpg)",
                "",
                "**Stages (input \u2192 plate \u2192 cropped \u2192 illumination \u2192 segmentation):**",
                "",
                f"![input]({name}/01_input.jpg)",
                f"![plate]({name}/02_plate_overlay.jpg)",
                f"![cropped]({name}/03_cropped.jpg)",
                f"![illum]({name}/04_illumination_corrected.jpg)",
                f"![seg]({name}/05_segmentation.jpg)",
                "",
                "---",
                "",
            ]
        )
        details.append("\n".join(sec))
    index.append("")
    index.extend(details)
    (OUT_DIR / "INDEX.md").write_text("\n".join(index))


if __name__ == "__main__":
    sys.exit(main())
