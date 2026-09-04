"""Shared batch-output writers used by both the CLI and the GUI.

Keeping the column logic in one place prevents the CLI and GUI batch CSVs from
silently diverging (e.g. the GUI dropping measurement columns the CLI keeps).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from blenny.pipeline.context import ImageData


def count_plates(data: ImageData) -> int:
    """Number of plates analysed for a single run result.

    A run has more than one plate when it produced more than one ROI
    (auto ``detect_facile`` or grid ``detect_multi_plate``), or when the
    sub-pipeline recorded per-plate counts for more than one slot. Otherwise
    it is a single plate (classic ``detect_plate``, manual mode, or a single
    plate found by auto mode) and counts as one.

    Used by the CLI and GUI to decide whether to emit batch files: a batch
    is written whenever a run analyses more than one plate (e.g. one scan
    image containing five plates), not merely when more than one *image* was
    passed in.
    """
    rois = data.metadata.get("rois")
    if rois:
        return len(rois)
    per_plate = data.metadata.get("per_plate_counts")
    if per_plate:
        occupied = [v for v in per_plate.values() if v != "NA"]
        # A grid run that found plates but left every slot empty shouldn't be
        # treated as a genuine batch of plates; fall back to a single plate.
        return len(occupied) if occupied else 1
    return 1


def batch_colonies_csv_text(measurements: list[dict[str, Any]]) -> str:
    """Return one row per measurement as CSV text.

    Column order follows the per-image :class:`CSVExporter`'s preferred order;
    any remaining columns produced by modules are appended (first appearance
    across rows) so no data is silently dropped. Returns a ``# no measurements``
    placeholder when there is nothing to write, so batch consumers always find
    the file.
    """
    import io

    if not measurements:
        return "# no measurements\n"

    preferred_order = [
        "plate_label",
        "label",
        "centroid_x",
        "centroid_y",
        "centroid_x_global",
        "centroid_y_global",
        "area_px",
        "circularity",
        "solidity",
        "eccentricity",
        "mean_r",
        "mean_g",
        "mean_b",
        "mean_h",
        "mean_s",
        "mean_v",
        "is_artifact",
        "artifact_reason",
        "source",
    ]

    fieldnames = [p for p in preferred_order if any(p in m for m in measurements)]

    # Append any remaining columns produced by modules so the batch CSV
    # matches the per-image CSVExporter format (no silent column drops).
    seen = set(fieldnames)
    for m in measurements:
        for key in m:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    fh = io.StringIO()
    writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(measurements)
    return fh.getvalue()


def write_batch_colonies_csv(path: Path, measurements: list[dict[str, Any]]) -> None:
    """Write one row per measurement to ``path`` (see :func:`batch_colonies_csv_text`).

    Bytes are written directly so the CSV's native ``\r\n`` line endings are
    preserved on Windows (``write_text`` would translate them to ``\r\r\n``).
    """
    path.write_bytes(batch_colonies_csv_text(measurements).encode("utf-8"))
