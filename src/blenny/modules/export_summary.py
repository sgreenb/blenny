"""Export a human-readable summary of the analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blenny.pipeline import BlennyParams, Exporter, ImageData, register


@register("export_summary")
class SummaryExporter(Exporter):
    """Write a technical log and summary of the plate analysis."""

    class Params(BlennyParams):
        output_path: str
        """Where to write the log file (e.g. "{output_dir}/{stem}/log.txt")."""

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
        text = self.generate_text(data)
        path.write_text(text, encoding="utf-8")

    def generate_text(self, data: ImageData) -> str:
        """Generate the human-readable summary text."""
        m = data.metadata
        plate_label = m.get("plate_label", "")
        label_header = f"Plate Label:      {plate_label}\n" if plate_label else ""

        # We only want to summarize non-artifact colonies in the main stats,
        # but the table at the bottom will show everything.
        colonies = [r for r in data.measurements if not r.get("is_artifact", False)]
        areas = [float(r["area_px"]) for r in colonies if "area_px" in r]
        has_warnings = any(f.severity in ("warning", "error") for f in data.quality_flags)

        lines = [
            "=== Blenny Processing Log ===",
            f"Source Image:     {data.source}",
            label_header.strip(),
            f"Analysis Status:  {'SUCCESS' if not has_warnings else 'COMPLETED WITH WARNINGS'}",
            f"Timestamp:        {m.get('run_at', 'N/A')}",
            "",
            "--- Count Statistics ---",
            f"Total Colonies:   {m.get('colony_count', 0)}",
            f"Artifacts Found:  {m.get('artifact_count', 0)}",
            "",
            "--- Size Distribution of Counted Colonies (pixels) ---",
        ]
        # Remove empty strings from list if plate_label was missing
        lines = [line for line in lines if line]

        if areas:
            lines.extend(
                [
                    f"Minimum Size:     {np.min(areas):.1f}",
                    f"Maximum Size:     {np.max(areas):.1f}",
                    f"Average Size:     {np.mean(areas):.1f}",
                    f"Std Deviation:    {np.std(areas):.1f}",
                ]
            )
        else:
            lines.append("No colonies detected to measure.")

        if m.get("per_plate_counts"):
            lines.extend(["", "--- Per-Plate Counts ---"])
            for label, count in m["per_plate_counts"].items():
                lines.append(f"{label}: {count}")

        if m.get("classification_counts"):
            lines.extend(["", "--- Classifications ---"])
            for label, count in m["classification_counts"].items():
                lines.append(f"{label}: {count}")

        if data.quality_flags:
            lines.extend(["", "--- Quality Flags & Warnings ---"])
            for flag in data.quality_flags:
                lines.append(f"[{flag.severity.upper()}] {flag.code}: {flag.message}")

        lines.extend(["", "--- Pipeline Provenance ---"])
        provenance = " -> ".join(p.step for p in data.provenance)
        lines.append(provenance)

        return "\n".join(lines) + "\n"
