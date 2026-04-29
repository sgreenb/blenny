"""Count colonies on a synthetic plate, end-to-end.

Run as a script:

    python examples/01_count_colonies.py

This is the smallest demo of Blenny's pipeline. It generates a synthetic
plate (so the example needs no input files), runs the full classical-CV
colony-counting pipeline, and writes a CSV + an annotated overlay PNG to
``examples/output/``.

The ``# %%`` markers let you open this file as cells in VS Code, PyCharm,
Spyder, or convert it to a Jupyter notebook with ``jupytext``.
"""

# %% Imports
from pathlib import Path

from PIL import Image

from blenny import Pipeline
from blenny.testing import make_synthetic_plate

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# %% Generate a synthetic plate so we don't need a real photo to demo with.
plate = make_synthetic_plate(n_colonies=30, image_size=(512, 512), seed=0)
plate_path = OUTPUT_DIR / "synthetic_plate.png"
Image.fromarray(plate.image).save(plate_path)
print(f"Generated synthetic plate with {plate.n_colonies} colonies → {plate_path}")

# %% Build the pipeline from a config (the same shape a YAML file will have).
pipe = Pipeline.from_config(
    [
        {"name": "load_image"},
        {"name": "detect_plate", "params": {"crop": True}},
        {"name": "correct_illumination", "params": {"radius": 20}},
        {"name": "threshold_segment", "params": {"roi_mask_key": "plate"}},
        {"name": "measure_colonies"},
        {
            "name": "export_csv",
            "params": {
                "output_path": str(OUTPUT_DIR / "colonies.csv"),
                "include_provenance": True,
            },
        },
        {
            "name": "export_annotated",
            "params": {"output_path": str(OUTPUT_DIR / "annotated.png")},
        },
    ]
)
print(pipe)

# %% Run it.
result = pipe.run(plate_path)

# %% Inspect the result.
print(f"\nDetected colonies: {result.metadata['colony_count']}")
print(f"Ground-truth count: {plate.n_colonies}")
print(f"Mean area (px):     {result.metadata.get('area_px_mean', 0):.1f}")

if result.quality_flags:
    print("\nQuality flags:")
    for flag in result.quality_flags:
        print(f"  [{flag.severity}] {flag.code} (in {flag.step}): {flag.message}")
else:
    print("\nNo quality flags raised.")

print("\nProvenance:")
for record in result.provenance:
    print(f"  {record.step:<22} {record.duration_s * 1000:7.1f} ms  {record.params}")

print(f"\nOutputs written to {OUTPUT_DIR}/")
