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
        variables = {
            "stem": data.metadata.get("stem", "image"),
            "input": data.source or "unknown",
            "output_dir": data.metadata.get("output_dir", "."),
            "plate_label": data.metadata.get("plate_label", "default"),
        }
        sub_path = self.params.output_path
        for k, v in variables.items():
            sub_path = sub_path.replace(f"{{{k}}}", str(v))

        path = Path(sub_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            fh.write(self.generate_csv(data))

    def generate_csv(self, data: ImageData) -> str:
        """Return the CSV as a string."""
        import io

        fh = io.StringIO()
        rows = data.measurements

        # Ensure 'is_artifact' column is present even if no module set it.
        for row in rows:
            row.setdefault("is_artifact", False)
            row.setdefault("artifact_reason", "")

        # Define a nice column order for researchers.
        # Any remaining columns found in the data will follow these.
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

        fieldnames: list[str] = []
        # 1. Add preferred columns if they exist in the data
        for p in preferred_order:
            if any(p in r for r in rows):
                fieldnames.append(p)

        # 2. In default export, we only keep the minimal set requested by the user.
        # We don't add "any other columns" here.

        if self.params.include_provenance:  # type: ignore[attr-defined]
            steps = " -> ".join(p.step for p in data.provenance)
            fh.write(f"# provenance: {steps}\n")

        if not fieldnames:
            # No data — still write an empty file with a header comment so
            # downstream tools don't trip on missing files.
            fh.write("# (no measurements)\n")
            return fh.getvalue()

        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            delimiter=self.params.delimiter,  # type: ignore[attr-defined]
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        return fh.getvalue()
