import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure the local src/ tree is importable
sys.path.insert(0, str(Path("src").resolve()))

try:
    from blenny.pipeline import Pipeline
    from blenny.config import extract_steps, load_yaml, substitute_paths
except ImportError:
    print("Error: Could not find blenny source. Ensure you are running from the repo root.")
    sys.exit(1)

def profile_pipeline(
    pipeline_path: str, 
    image_path: str, 
    output_prof: str = "sandbox/pipeline_profile.prof",
    grid: Optional[list[int]] = None
):
    """
    Runs a Blenny pipeline under cProfile and saves the results.
    """
    img_p = Path(image_path)
    pipe_p = Path(pipeline_path)
    
    if not img_p.exists():
        print(f"Error: Image {image_path} not found.")
        return
    
    print(f"--- Deep Profiling: {img_p.name} ---")
    print(f"Pipeline: {pipe_p.name}")
    
    # Setup output directory
    debug_out = Path("sandbox/profile_results")
    debug_out.mkdir(parents=True, exist_ok=True)

    # 1. Prepare the pipeline
    raw_config = load_yaml(pipe_p)
    raw_steps = extract_steps(raw_config)
    
    # Apply grid override if provided (useful for multiplate)
    if grid:
        for step in raw_steps:
            if step["name"] == "detect_multi_plate":
                step.setdefault("params", {})["grid"] = grid
                print(f"Applying grid override: {grid}")

    resolved = substitute_paths(raw_steps, input_path=img_p, output_dir=debug_out)
    pipe = Pipeline.from_config(resolved)

    # 2. Profiling execution
    profiler = cProfile.Profile()
    
    print("Execution starting...")
    start_time = time.perf_counter()
    
    profiler.enable()
    try:
        # We run the actual pipeline logic here
        data = pipe.run(img_p, output_dir=debug_out)
    finally:
        profiler.disable()
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"Execution finished in {duration:.2f}s")

    # 3. Save and report
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    # Save the raw profile data for Snakeviz/Flamegraphs
    stats.dump_stats(output_prof)
    print(f"Profile data saved to: {output_prof}")
    
    print("\n--- Top 20 Bottlenecks (Cumulative Time) ---")
    stats.print_stats(20)

    print("\nNext Steps:")
    print(f"1. To view the Flame Graph, run: pip install snakeviz")
    print(f"2. Then run: snakeviz {output_prof}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default="pipeline_multi.yaml")
    parser.add_argument("--image", default="ml_training/datasets/multiplate_data/C_1-5_005.jpg")
    parser.add_argument("--grid", type=int, nargs=2)
    args = parser.parse_args()

    profile_pipeline(
        pipeline_path=args.pipeline,
        image_path=args.image,
        grid=args.grid
    )
