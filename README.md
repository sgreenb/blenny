# Blenny

**A free, open-source toolkit for analyzing plates and microscopy images.**

---

## Quick Start

If you have Python 3.11+ installed, you can start counting in seconds.

### 1. Installation
```bash
git clone https://github.com/your-org/blenny.git
cd blenny
pip install -e .
```

### 2. Run your first analysis
```bash
# Generate a starter pipeline (creates pipeline.yaml)
blenny init

# Run on an image (or a folder using glob patterns)
blenny run pipeline.yaml --input plate.jpg --output results/
```

### 3. View Results
Results land in `results/<image_name>/`:
- **`summary.txt`**: A human-readable report with counts, size stats, and a detailed table of per-colony coordinates and color data.
- **`annotated.png`**: An image overlay showing exactly what was counted (outlines and ID numbers).
- **`colonies.csv`**: Full measurements for every colony (area, RGB, HSV, eccentricity) for use in Excel/R.
- **`debug/`**: (Optional) Use `--debug-dir debug/` to see every intermediate step for auditing.

---

## Core Features

- **Automated Plate Detection**: Finds the circular Petri dish and crops the image automatically.
- **Scale-Aware Processing**: Illumination correction and filters adapt to your image resolution.
- **Color & Intensity Quantification**: Extracts RGB and HSV (Hue, Saturation, Value) statistics for every colony by default.
- **Smart Multiplicity**: Uses geometric heuristics to identify and count overlapping/merged colonies.
- **Artifact Rejection**: Filters out plate rims, scratches, and pen marks using interior-anchored classification.
- **Quality Flags**: Automatically warns you if a plate has a suspect count, many edge-touches, or poor detection confidence.
- **Transparent Logic**: Every decision is auditable via debug images and quality flags.

---

## Command Line Interface

Blenny is designed to be "CLI-first," making it easy to process hundreds of plates at once.

| Command | Description |
| :--- | :--- |
| `blenny run` | The workhorse. Processes images through a YAML pipeline. |
| `blenny init` | Scaffolds a starter `pipeline.yaml` with sensible defaults. |
| `blenny modules` | Lists all available analysis modules (Otsu, Watershed, etc.) and their params. |
| `blenny gui` | Launches the visual interface. |

### Parameter Overrides
You can tweak any pipeline parameter on the fly without editing the YAML file:
```bash
# Example: change the minimum colony area and exclusion margin
blenny run pipeline.yaml -i plate.jpg -o results/ -v threshold_segment.min_area=50 -v detect_plate.margin_frac=0.05
```

---

## Python API

You can also use Blenny as a library in your own scripts or Jupyter notebooks:

```python
from blenny import Pipeline

# Load and run a pipeline
pipe = Pipeline.from_yaml("pipeline.yaml")
result = pipe.run("plate.jpg")

print(f"Found {result.metadata['colony_count']} colonies.")
```

---

## Roadmap & Status

**Status:** Pre-alpha. The API is stabilizing but subject to change.

- [x] Core modular pipeline architecture
- [x] Classical-CV colony counting vertical slice
- [x] YAML-based configurations & reproducibility
- [x] Human-readable summary exports
- [x] Automated debug/audit trail generation
- [x] Scale-aware watershed seeding & filters
- [x] Color & Intensity quantification
- [x] GUI skeleton & Interactive Review
- [ ] **Next:** Sector counting (quadrant analysis)
- [ ] **Next:** ML-based segmentation (Cellpose integration)
- [ ] **Next:** Support for Time-lapse / Growth curves

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
