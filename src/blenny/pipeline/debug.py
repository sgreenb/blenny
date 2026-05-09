"""Utilities for dumping pipeline diagnostics (Step 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color, util
from skimage.color import label2rgb

from blenny.pipeline.context import ImageData


class DebugWriter:
    """Serializes a step-by-step audit trail of an image's analysis.

    Output layout (under ``debug_dir/``):
        01_load_image.jpg         (image after loader)
        02_detect_plate.jpg       (image after plate detection)
        ...
        04b_mask_objects.png      (mask created by threshold_segment)
        ...
        summary.txt               (timings and flags)
    """

    def __init__(self, debug_dir: Path) -> None:
        self.debug_dir = debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._step_counter = 1

    def write_step(self, step_name: str, data: ImageData) -> None:
        """Dump the current state of ``data.image`` and any new masks."""
        prefix = f"{self._step_counter:02d}_{step_name}"

        # 1. Save the current working image
        if data.image is not None:
            self._save_image(data.image, self.debug_dir / f"{prefix}.jpg")

        # 2. Save any masks that are label/binary images
        for key, mask in data.masks.items():
            # Only save the mask if it hasn't been saved yet, or if this is the
            # step that likely produced it. Since we don't track who produced
            # what, we just save all masks that look like images.
            mask_path = self.debug_dir / f"{prefix}_mask_{key}.png"
            if not mask_path.exists():
                self._save_mask(mask, mask_path, background=data.image)

        self._step_counter += 1

    def write_summary(self, data: ImageData) -> None:
        """Write a text summary of timings and flags."""
        lines = [
            f"source:        {data.source}",
            f"colony_count:  {data.metadata.get('colony_count', '?')}",
            "",
            "provenance:",
        ]
        for p in data.provenance:
            lines.append(f"  {p.duration_s * 1000:8.0f} ms  {p.step} ({p.module_class})")

        lines.append("\nquality_flags:")
        if data.quality_flags:
            for f in data.quality_flags:
                lines.append(f"  [{f.severity}] {f.code} (in {f.step}): {f.message}")
        else:
            lines.append("  (none)")

        (self.debug_dir / "debug_log.txt").write_text("\n".join(lines) + "\n")

    def _save_image(self, arr: np.ndarray, path: Path) -> None:
        try:
            # Min-max stretch float images for display
            if arr.dtype.kind == "f":
                arr = util.img_as_ubyte(np.clip(arr, 0, 1))
            elif arr.dtype == bool:
                arr = arr.astype(np.uint8) * 255
            im = Image.fromarray(arr)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            im.thumbnail((1200, 1200))
            im.save(path, quality=85)
        except Exception:
            pass  # Don't let debug-saving crash the pipeline

    def _save_mask(
        self, mask: np.ndarray, path: Path, background: np.ndarray | None = None
    ) -> None:
        try:
            mask_arr = np.asarray(mask)
            if mask_arr.ndim != 2:
                return

            if mask_arr.dtype == bool:
                # Simple binary mask
                vis = mask_arr.astype(np.uint8) * 255
            else:
                # Label image - use color overlay if background is available
                if background is not None:
                    bg = color.rgb2gray(background) if background.ndim == 3 else background
                    # label2rgb expects bg in [0, 1]
                    if bg.dtype == np.uint8:
                        bg = bg.astype(float) / 255.0
                    vis = (label2rgb(mask_arr, image=bg, bg_label=0) * 255).astype(np.uint8)
                else:
                    vis = (label2rgb(mask_arr, bg_label=0) * 255).astype(np.uint8)

            im = Image.fromarray(vis)
            im.thumbnail((1200, 1200))
            im.save(path)
        except Exception:
            pass
