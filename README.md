# Blenny

A free, open-source toolkit for analyzing plates and microscopy images.

> **Status:** Pre-alpha. The API will change. See [`design.md`](design.md) for the project vision.

## Mission

Provide lab researchers — particularly graduate students and postdocs working with bacteria
and yeast — a free, transparent, and extensible tool for quantitative analysis of colonies
and microscopy images. Routine analyses should be effortless, advanced analyses possible,
and every step inspectable.

## Install (development)

```bash
git clone https://github.com/your-org/blenny.git
cd blenny
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Quick taste

### From the command line

```bash
# Scaffold a starter pipeline YAML
blenny init --out pipeline.yaml

# Run on one image, or many (quote the glob!)
blenny run pipeline.yaml --input plate.jpg          --output results/
blenny run pipeline.yaml --input "plates/*.jpg"     --output results/

# Inspect every registered module and its params
blenny modules
```

Results land in `results/<image-stem>/colonies.csv` + `annotated.png`,
plus a per-image `provenance.json` recording every step that ran. A
`results/summary.csv` rolls up counts and timings across the batch, and
`results/config.yaml` records the exact (resolved) pipeline used.

### From Python

```python
from blenny import Pipeline

pipe = Pipeline.from_yaml("pipeline.yaml")
result = pipe.run("plate.jpg")
print(f"Found {result.metadata['colony_count']} colonies")
```

For a runnable end-to-end demo on a synthetic plate (no input image required):

```bash
python examples/01_count_colonies.py
```

## Roadmap

- [x] Step 0 — Project bootstrap
- [x] Step 1 — Core pipeline abstractions (Loader/Preprocessor/Segmenter/...)
- [x] Step 2 — Vertical slice: classical-CV colony counting on plate photos
- [x] Step 3 — YAML pipeline configs + CLI (`blenny run pipeline.yaml`)
- [ ] Step 4 — Transparency: per-step debug output, quality flags surfacing
- [ ] Step 5 — Documentation and examples

## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE).
