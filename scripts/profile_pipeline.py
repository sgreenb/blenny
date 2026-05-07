"""Profile the per-step timing of the Blenny pipeline on one or more images.

Runs the pipeline normally (using the existing provenance timing recorded by
the runner) and prints a formatted breakdown showing how long each step takes
and what fraction of total wall-clock time it consumes.

Usage
-----
    # Single image, default pipeline
    python scripts/profile_pipeline.py example_plates/A2.png

    # Single image, custom pipeline
    python scripts/profile_pipeline.py example_plates/A2.png --pipeline pipeline.yaml

    # Average over multiple images for a more stable estimate
    python scripts/profile_pipeline.py example_plates/batch/*.png

    # Run N times on the same image and average (to smooth JIT / cache effects)
    python scripts/profile_pipeline.py example_plates/A2.png --repeat 3

Output is written to stdout only — no files are modified.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

# Make sure the local src/ tree is importable when run from the repo root.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from blenny.config import extract_steps, load_yaml, substitute_paths  # noqa: E402
from blenny.pipeline import Pipeline  # noqa: E402

DEFAULT_PIPELINE = REPO / "pipeline.yaml"


def run_once(pipeline_path: Path, image_path: Path) -> list[tuple[str, str, float]]:
    """Run the pipeline on one image and return [(step_name, module_class, duration_s)]."""
    raw_config = load_yaml(pipeline_path)
    raw_steps = extract_steps(raw_config)
    output_dir = REPO / "_profile_tmp"
    resolved = substitute_paths(raw_steps, input_path=image_path, output_dir=output_dir)
    pipe = Pipeline.from_config(resolved)
    data = pipe.run(image_path)
    return [(p.step, p.module_class, p.duration_s) for p in data.provenance]


def format_bar(frac: float, width: int = 30) -> str:
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def print_report(
    step_times: dict[str, list[float]],
    step_classes: dict[str, str],
    image_paths: list[Path],
    repeat: int,
    pipeline_path: Path,
    total_wall: float,
) -> None:
    n_runs = len(image_paths) * repeat
    avg_totals = {step: sum(times) / len(times) for step, times in step_times.items()}
    grand_total = sum(avg_totals.values())

    print()
    print("=" * 70)
    print("  Blenny Pipeline Timing Profile")
    print("=" * 70)
    print(f"  Pipeline : {pipeline_path}")
    print(f"  Images   : {len(image_paths)} image(s), {repeat} repeat(s) each  ({n_runs} total runs)")
    print(f"  Wall time: {total_wall:.1f}s total  |  {total_wall/n_runs:.1f}s avg per run")
    print(f"  Pipeline : {grand_total:.2f}s avg step total (excludes runner overhead)")
    print("-" * 70)
    print(f"  {'Step':<30} {'Module':<28} {'Avg(s)':>7} {'%':>6}  Bar")
    print("-" * 70)

    # Sort by average duration descending
    for step, avg in sorted(avg_totals.items(), key=lambda x: -x[1]):
        frac = avg / grand_total if grand_total > 0 else 0
        cls = step_classes.get(step, "")
        bar = format_bar(frac, width=20)
        print(f"  {step:<30} {cls:<28} {avg:>7.3f} {frac*100:>5.1f}%  {bar}")

    print("-" * 70)
    print(f"  {'TOTAL':<59} {grand_total:>7.3f}")
    print("=" * 70)
    print()
    print("  Tip: the slowest steps are the best candidates for optimisation.")
    print("  Consider profiling with --repeat 3 on a high-res image for stable results.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile per-step timing of the Blenny pipeline."
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="One or more input image paths.",
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=DEFAULT_PIPELINE,
        help=f"Path to the pipeline YAML (default: {DEFAULT_PIPELINE})",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Run each image N times and average the results (default: 1).",
    )
    args = parser.parse_args()

    # Validate inputs
    missing = [p for p in args.images if not p.exists()]
    if missing:
        for p in missing:
            print(f"Error: image not found: {p}", file=sys.stderr)
        sys.exit(1)
    if not args.pipeline.exists():
        print(f"Error: pipeline not found: {args.pipeline}", file=sys.stderr)
        sys.exit(1)
    if args.repeat < 1:
        print("Error: --repeat must be >= 1", file=sys.stderr)
        sys.exit(1)

    step_times: dict[str, list[float]] = defaultdict(list)
    step_classes: dict[str, str] = {}

    print(f"\nProfiling {len(args.images)} image(s) x {args.repeat} repeat(s)...")

    wall_start = time.perf_counter()
    for img in args.images:
        for run_idx in range(args.repeat):
            label = f"  {img.name}" + (f" (run {run_idx+1}/{args.repeat})" if args.repeat > 1 else "")
            print(label, end="", flush=True)
            t0 = time.perf_counter()
            try:
                results = run_once(args.pipeline, img)
            except Exception as e:
                print(f"  FAILED: {e}", file=sys.stderr)
                sys.exit(1)
            elapsed = time.perf_counter() - t0
            print(f"  →  {elapsed:.1f}s")
            for step, cls, duration in results:
                step_times[step].append(duration)
                step_classes[step] = cls
    total_wall = time.perf_counter() - wall_start

    # Clean up temp output dir silently
    import shutil
    tmp = REPO / "_profile_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)

    print_report(step_times, step_classes, args.images, args.repeat, args.pipeline, total_wall)


if __name__ == "__main__":
    main()
