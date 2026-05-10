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
        lines = [l for l in lines if l]

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

        if m.get("classification_counts"):
            lines.extend(["", "--- Classifications ---"])
            for label, count in m["classification_counts"].items():
                lines.append(f"{label}: {count}")

        if data.quality_flags:
            lines.extend(["", "--- Quality Flags & Warnings ---"])
            for flag in data.quality_flags:
                lines.append(f"[{flag.severity.upper()}] {flag.code}: {flag.message}")

        # --- Per-Colony Table (Step 3 Requirement) ---
        if data.measurements:
            lines.extend(["", "--- Per-Colony Measurements ---"])
            # Header
            header = f"{'Plate':<10} {'ID':<4} {'X':>7} {'Y':>7} {'Area':>8} {'R':>5} {'G':>5} {'B':>5} {'H':>5} {'S':>5} {'V':>5} {'Type':<10}"
            lines.append(header)
            lines.append("-" * len(header))

            # Show colonies first, then artifacts. Measurements are already
            # re-labeled and re-ordered in classify_interior / filter_by_id.
            for r in data.measurements:
                plabel = str(r.get("plate_label", plate_label))[:10]
                cid = str(r.get("label", "?"))
                x = f"{float(r.get('centroid_x', 0)):.1f}"
                y = f"{float(r.get('centroid_y', 0)):.1f}"
                area = str(int(r.get("area_px", 0)))

                # Colors (optional)
                rgb_hsv = []
                for k in ["mean_r", "mean_g", "mean_b", "mean_h", "mean_s", "mean_v"]:
                    val = r.get(k)
                    rgb_hsv.append(f"{float(val):.2f}" if val is not None else "-")

                ctype = "Colony"
                if r.get("is_artifact"):
                    ctype = "Artifact"
                elif int(r.get("colony_count_estimate", 1)) >= 2:
                    ctype = f"Merged(x{r['colony_count_estimate']})"

                lines.append(
                    f"{plabel:<10} {cid:<4} {x:>7} {y:>7} {area:>8} "
                    f"{rgb_hsv[0]:>5} {rgb_hsv[1]:>5} {rgb_hsv[2]:>5} "
                    f"{rgb_hsv[3]:>5} {rgb_hsv[4]:>5} {rgb_hsv[5]:>5} "
                    f"{ctype:<10}"
                )

        lines.extend(["", "--- Pipeline Provenance ---"])
        provenance = " -> ".join(p.step for p in data.provenance)
        lines.append(provenance)

        return "\n".join(lines) + "\n"
