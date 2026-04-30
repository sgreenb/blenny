"""Run the default pipeline on a single image and dump every intermediate.

Useful for diagnosing why a specific colony was missed or counted in error.

Usage:
    python scripts/debug_one.py example_plates/ex5.jpg [--output debug_out/]

Outputs a directory with one image per pipeline stage, plus a summary.txt
that lists the count change at each stage so you can see which step lost
or kept each colony.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import filters, measure, morphology, segmentation
from skimage.color import label2rgb

from blenny import ImageData
from blenny.modules import (
    AnnotatedImageExporter,
    ColonyMeasurer,
    IlluminationCorrection,
    ImageFileLoader,
    InteriorColonyClassifier,
    PlateDetector,
)


def _to_uint8_display(arr: np.ndarray) -> np.ndarray:
    """Stretch a float array to uint8 for visual inspection."""
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _save(arr: np.ndarray, path: Path) -> None:
    im = Image.fromarray(arr)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.save(path, quality=92)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--radius-expand-frac", type=float, default=0.05)
    ap.add_argument("--margin-frac", type=float, default=0.08)
    ap.add_argument("--tophat-radius", type=int, default=25)
    ap.add_argument("--min-area", type=int, default=10)
    ap.add_argument("--min-circularity", type=float, default=0.7)
    ap.add_argument("--min-solidity", type=float, default=0.85)
    args = ap.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1

    out = args.output or (args.image.parent / "output" / args.image.stem / "debug")
    out.mkdir(parents=True, exist_ok=True)

    data = ImageData(source=str(args.image))

    # --- 1. Load + plate detect (so we have a cropped image and plate mask) ---
    ImageFileLoader().run(data)
    PlateDetector(
        crop=True, radius_expand_frac=args.radius_expand_frac, margin_frac=args.margin_frac
    ).run(data)
    _save(data.image, out / "01_cropped.jpg")
    _save(data.masks["plate"].astype(np.uint8) * 255, out / "02_plate_mask.jpg")

    # --- 2. Illumination correction ---
    IlluminationCorrection(radius=args.tophat_radius).run(data)
    illum = data.image  # 2D float
    _save(_to_uint8_display(illum), out / "03_illumination_corrected.jpg")

    # --- 3. Otsu threshold (before any filtering) ---
    otsu_t = float(filters.threshold_otsu(illum))
    binary_raw = illum > otsu_t
    plate_mask = data.masks["plate"].astype(bool)
    binary_in_roi = binary_raw & plate_mask
    raw_labels = measure.label(binary_in_roi)
    n_raw = int(raw_labels.max())
    _save(_to_uint8_display(binary_in_roi.astype(float)), out / "04_binary_otsu.jpg")
    _save(
        (label2rgb(raw_labels, image=illum, bg_label=0, alpha=0.55) * 255).astype(np.uint8),
        out / "04b_labels_raw.jpg",
    )

    # --- 4. After remove_small_objects + opening ---
    binary_clean = morphology.remove_small_objects(binary_in_roi, min_size=args.min_area)
    binary_clean = morphology.opening(binary_clean, morphology.disk(1))
    clean_labels = measure.label(binary_clean)
    n_clean = int(clean_labels.max())
    _save(
        (label2rgb(clean_labels, image=illum, bg_label=0, alpha=0.55) * 255).astype(np.uint8),
        out / "05_after_clean.jpg",
    )

    # --- 5. After watershed split ---
    from scipy import ndimage as ndi
    from skimage import feature

    distance = ndi.distance_transform_edt(binary_clean)
    coords = feature.peak_local_max(distance, min_distance=5, labels=binary_clean)
    seed_mask = np.zeros(distance.shape, dtype=bool)
    if len(coords):
        seed_mask[tuple(coords.T)] = True
    markers = measure.label(seed_mask)
    ws_labels = segmentation.watershed(-distance, markers, mask=binary_clean)
    n_ws = int(ws_labels.max())
    _save(
        (label2rgb(ws_labels, image=illum, bg_label=0, alpha=0.55) * 255).astype(np.uint8),
        out / "06_after_watershed.jpg",
    )

    # --- 6. After circularity/solidity filters (mimics ThresholdSegmenter end) ---
    import math as _math

    filtered = ws_labels.copy()
    rejected_circ = []
    rejected_sol = []
    for prop in measure.regionprops(filtered):
        drop = False
        if args.min_circularity > 0 and prop.perimeter > 0:
            circ = 4.0 * _math.pi * prop.area / (prop.perimeter**2)
            if circ < args.min_circularity:
                drop = True
                rejected_circ.append((prop.label, prop.area, circ))
        if not drop and args.min_solidity > 0 and prop.solidity < args.min_solidity:
            drop = True
            rejected_sol.append((prop.label, prop.area, prop.solidity))
        if drop:
            filtered[filtered == prop.label] = 0
    final_labels = measure.label(filtered > 0)
    n_final = int(final_labels.max())
    _save(
        (label2rgb(final_labels, image=illum, bg_label=0, alpha=0.55) * 255).astype(np.uint8),
        out / "07_after_shape_filters.jpg",
    )

    # --- 7. Run the rest of the pipeline (measure + classify + annotate) ---
    data.masks["objects"] = final_labels.astype(np.int32)
    ColonyMeasurer().run(data)
    InteriorColonyClassifier().run(data)
    AnnotatedImageExporter(output_path=str(out / "08_annotated.jpg")).run(data)

    n_after_class = data.metadata.get("colony_count", 0)
    n_artifacts = data.metadata.get("artifact_count", 0)

    # --- 8. "Fate map": for every blob in the cleaned binary, paint a circle
    #         on the original image colored by what happened to it.
    counted_centroids = [
        (float(r["centroid_y"]), float(r["centroid_x"]))
        for r in data.measurements
        if not r.get("is_artifact")
    ]
    artifact_centroids = [
        (float(r["centroid_y"]), float(r["centroid_x"]))
        for r in data.measurements
        if r.get("is_artifact")
    ]

    from PIL import ImageDraw  # local import to keep Image at the top

    fate_img = Image.fromarray(_to_uint8_display(data.artifacts.get("pre_illumination", illum)))
    if fate_img.mode != "RGB":
        fate_img = fate_img.convert("RGB")
    draw = ImageDraw.Draw(fate_img)

    GREEN = (40, 220, 40)
    ORANGE = (255, 140, 0)
    RED = (255, 30, 30)

    n_counted = 0
    n_marked = 0
    n_dropped = 0

    # Walk every blob in the cleaned binary and decide its fate by nearest-centroid.
    for prop in measure.regionprops(clean_labels):
        cy, cx = prop.centroid
        # Find nearest measurement centroid within reasonable radius.
        best_d2 = 25**2  # within 25 px counts as "same blob"
        best_kind: str | None = None
        for ccy, ccx in counted_centroids:
            d2 = (cy - ccy) ** 2 + (cx - ccx) ** 2
            if d2 < best_d2:
                best_d2, best_kind = d2, "counted"
        for acy, acx in artifact_centroids:
            d2 = (cy - acy) ** 2 + (cx - acx) ** 2
            if d2 < best_d2:
                best_d2, best_kind = d2, "artifact"

        r = max(4, int(prop.equivalent_diameter / 2) + 2)
        bbox = (cx - r, cy - r, cx + r, cy + r)
        if best_kind == "counted":
            draw.ellipse(bbox, outline=GREEN, width=2)
            n_counted += 1
        elif best_kind == "artifact":
            draw.ellipse(bbox, outline=ORANGE, width=2)
            n_marked += 1
        else:
            draw.ellipse(bbox, outline=RED, width=2)
            n_dropped += 1

    fate_img.save(out / "09_fate_map.jpg", quality=92)

    # --- Write a step-by-step count summary ---
    fate_summary = (
        f"\nfate map (every blob in clean_labels classified):\n"
        f"  counted (green):    {n_counted}\n"
        f"  marked artifact (orange): {n_marked}\n"
        f"  dropped silently (red):   {n_dropped}\n"
    )
    lines = [
        f"image:          {args.image.name}",
        f"params:         radius_expand_frac={args.radius_expand_frac}  margin_frac={args.margin_frac}",
        f"                tophat_radius={args.tophat_radius}  min_area={args.min_area}",
        f"                min_circularity={args.min_circularity}  min_solidity={args.min_solidity}",
        "",
        f"otsu threshold:                {otsu_t:.4f}",
        "",
        "step-by-step object counts:",
        f"  raw threshold (in ROI):      {n_raw}",
        f"  after remove_small + open:   {n_clean}    ({n_clean - n_raw:+d})",
        f"  after watershed split:       {n_ws}    ({n_ws - n_clean:+d})",
        f"  after circularity/solidity:  {n_final}    ({n_final - n_ws:+d})",
        f"  after interior classifier:   {n_after_class}    ({n_after_class - n_final:+d}, "
        f"{n_artifacts} marked is_artifact)",
        "",
        f"rejections by circularity:     {len(rejected_circ)}",
        f"rejections by solidity:        {len(rejected_sol)}",
        fate_summary,
    ]
    if rejected_circ[:10]:
        lines.append("")
        lines.append("first 10 circularity rejections (label, area, circ):")
        for lab, area, circ in rejected_circ[:10]:
            lines.append(f"  {lab:4d}  area={area:6.0f}  circ={circ:.3f}")
    if rejected_sol[:10]:
        lines.append("")
        lines.append("first 10 solidity rejections (label, area, solidity):")
        for lab, area, sol in rejected_sol[:10]:
            lines.append(f"  {lab:4d}  area={area:6.0f}  sol={sol:.3f}")

    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}/")
    print("\n".join(lines[6:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
