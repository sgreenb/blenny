"""Write per-object measurements to a CSV (or TSV) file."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

from blenny.pipeline import BlennyParams, Exporter, ImageData, register


@register("export_csv")
class CSVExporter(Exporter):
    class Params(BlennyParams):
        output_path: str
        """Destination file. Parent directories are created if missing."""

        delimiter: Literal[",", "\t", ";", "|"] = ","
        include_provenance: bool = False
        """If True, prepend a ``# provenance:`` comment line listing the steps run."""

    def export(self, data: ImageData) -> None:
        path = Path(self.params.output_path)  # type: ignore[attr-defined]
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = data.measurements
        
        # Ensure 'is_artifact' column is present even if no module set it.
        for row in rows:
            row.setdefault("is_artifact", False)
            row.setdefault("artifact_reason", "")

        # Define a nice column order for researchers. 
        # Any remaining columns found in the data will follow these.
        preferred_order = [
            "label", "centroid_x", "centroid_y", "area_px", "eccentricity", 
            "mean_r", "mean_g", "mean_b", "mean_h", "mean_s", "mean_v", 
            "is_artifact", "artifact_reason", "source"
        ]
        
        fieldnames: list[str] = []
        # 1. Add preferred columns if they exist in the data
        for p in preferred_order:
            if any(p in r for r in rows):
                fieldnames.append(p)
        
        # 2. Add any other columns found in the data
        for row in rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)

        with path.open("w", newline="", encoding="utf-8") as fh:
            if self.params.include_provenance:  # type: ignore[attr-defined]
                steps = " -> ".join(p.step for p in data.provenance)
                fh.write(f"# provenance: {steps}\n")
            if not fieldnames:
                # No data — still write an empty file with a header comment so
                # downstream tools don't trip on missing files.
                fh.write("# (no measurements)\n")
                return
            writer = csv.DictWriter(
                fh,
                fieldnames=fieldnames,
                delimiter=self.params.delimiter,  # type: ignore[attr-defined]
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
