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
        # Stable, predictable column order: collect keys in first-seen order.
        fieldnames: list[str] = []
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
