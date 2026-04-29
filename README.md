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

```python
from blenny.pipeline import Pipeline
from blenny.modules.identity import IdentityPreprocessor

pipe = Pipeline([IdentityPreprocessor()])
result = pipe.run({"image": ...})
```

A real colony-counting pipeline is the next milestone — see the roadmap below.

## Roadmap

- [x] Step 0 — Project bootstrap (this commit)
- [ ] Step 1 — Core pipeline abstractions (Loader/Preprocessor/Segmenter/...)
- [ ] Step 2 — Vertical slice: classical-CV colony counting on plate photos
- [ ] Step 3 — YAML pipeline configs + CLI (`blenny run pipeline.yaml`)
- [ ] Step 4 — Transparency: per-step debug output, quality flags
- [ ] Step 5 — Documentation and examples

## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE).
