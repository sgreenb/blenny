Project Vision Document: Blenny
A free, open-source toolkit for analyzing plates and microscopy images

1. Mission Statement
To provide lab researchers — particularly graduate students and postdocs working with bacteria and yeast — a free, transparent, and extensible tool for quantitative analysis of colonies and microscopy images. The tool should make routine analyses effortless, advanced analyses possible, and every step inspectable.

2. Target Users
Primary: Graduate students and postdocs in microbiology, molecular biology, and adjacent fields who:

Work with bacterial or yeast colonies on plates, or cells under brightfield/phase-contrast microscopy
Have variable programming experience (from none to fluent)
Need reproducible, publication-ready results
Operate in resource-constrained environments (no budget for commercial software)

Secondary (future): Undergraduates, teaching labs, citizen scientists, plugin developers from the broader bioimage community.

3. Scope
In Scope (Phase 1 — MVP)

Image sources: Phone photos of plates, flatbed scans, brightfield and phase-contrast microscopy
Organisms: Bacteria and yeast
Analyses: Colony counting, colony size measurement, basic color binning
Batch mode: Apply a single analysis pipeline to many images
Export: CSV, TSV, plain text, and annotated images
Platforms: Windows and macOS (Linux as a bonus)
Operation: Fully offline after installation

In Scope (Future Phases)

Fluorescence intensity and color quantification
Zone-of-inhibition measurements
Yeast budding analysis
Time-lapse / growth curves
Per-image custom pipelines within a batch
Optional reproducibility/provenance tracking
User-trained models (if pretrained options prove insufficient)

Explicit Non-Goals

Cloud / web-hosted services (local only)
Commercial licensing or paid features
Replacing specialized tools like CellProfiler or QuPath
Real-time analysis from camera feeds


4. Guiding Principles

4.1 Accessibility Without Dumbing Down
A first-time user should be able to count colonies on a plate within minutes of installing the tool. A power user should be able to script complex pipelines, swap algorithms, and write plugins. Sensible defaults exist, but every default is overridable.

4.2 Composable, Not Monolithic
Workflows are built from interchangeable steps. Users (and developers) can add, remove, or reorder steps without breaking the whole. The unit of functionality is the module, not the application.

4.3 Transparent, Not Magical
Every analysis step is inspectable. Users can see intermediate outputs, understand why a colony was counted (or wasn't), and trust the results. No black boxes — even when using ML models, the inputs, outputs, and confidences are visible.

4.4 Robust to Real-World Inputs
Real lab images are messy: uneven lighting, glare, fingerprints on plates, blurry phone photos, varying agar colors. The tool should degrade gracefully on imperfect inputs and clearly flag when results are unreliable.

4.5 Reproducible by Default (Aspirational)
Analyses should be re-runnable. Where feasible without slowing development, parameters and pipeline configurations are saved alongside results so any analysis can be reproduced exactly.

4.6 Free

No paid services or APIs
All dependencies must be free for any user, including commercial labs
Models bundled or downloaded must be redistributable under permissive terms
Source code is open and modifiable

4.7 Honest About Limitations
The tool surfaces uncertainty. If colonies overlap and counts are ambiguous, say so. If a model has low confidence, show it. A wrong answer presented confidently is worse than no answer.

4.8 Batteries Included
Out of the box: sensible defaults, example data, tutorial workflows, pretrained models for common tasks. The user shouldn't need to assemble parts before getting started.

5. Design Themes

5.1 Library-First, GUI-Eventually
The core will be a Python library with a clean API. Initial interaction will be via Jupyter notebooks and a CLI; a GUI will be added once the core stabilizes. This staging ensures the GUI is built on a solid foundation.

5.2 Pipelines as First-Class Citizens
A workflow is an ordered series of modules: Load → Preprocess → Segment → Measure → Classify → Export. E

5.3 Configuration Over Code
A pipeline can be expressed as a YAML or JSON file. Beginners use templates; advanced users customize them. This also enables reproducibility, sharing, and (eventually) GUI editing of pipelines.

5.4 Pretrained First, Trainable Later
The tool will use pretrained models (e.g., Cellpose, StarDist) where they exist and meet our licensing criteria. User-trainable workflows will come later, only if pretrained options prove insufficient for our target tasks.

5.5 Classical CV + ML, Not One or the Other
Some problems (e.g., colony counting on a clean plate) are solved better and faster with classical computer vision. Others (e.g., touching cells under a microscope) benefit from deep learning. The tool offers both, and lets users choose.

5.6 CPU-Friendly with Optional GPU Acceleration
Default workflows must run on a modern laptop without a GPU. GPU support is automatic when available, providing speedups for ML steps but never required.

6. Technical Pillars
6.1 Language and Core Stack

Python 3.14+ as the implementation language
NumPy, scikit-image, OpenCV for core image processing
scikit-learn, PyTorch for ML
Cellpose / StarDist as candidate pretrained segmentation models (licensing to verify)
pandas for tabular outputs
Pydantic + YAML for configuration

6.2 Architecture

Modular pipeline architecture with abstract interfaces (Loader, Preprocessor, Segmenter, FeatureExtractor, Classifier, Exporter)
Plugin-friendly: new modules drop in by implementing an interface and registering
Separation of concerns: I/O, processing, analysis, presentation are distinct layers

6.3 Distribution

Installable via pip and conda
Cross-platform (Windows, macOS, Linux) — verified by CI
Bundled or auto-downloaded models for fully offline operation after first run
Single-command install as a goal: pip install [toolname] and you're ready

6.4 Interfaces (Phased)

Phase 1: Python API + Jupyter notebooks + CLI (using Typer or Click)
Phase 2: Local GUI (Napari plugin or standalone with Qt/Streamlit)
Phase 3: Pipeline editor in the GUI

6.5 Outputs

Tabular: CSV, TSV (per-object measurements, summary statistics)
Text: Plain-text reports, logs
Images: Annotated overlays, segmentation masks, intermediate-step visualizations
Configurations: YAML files describing the pipeline used (foundation for reproducibility)


7. Licensing and Openness
Recommendation: GNU General Public License v3 (GPLv3)

8. Success Criteria
The project succeeds if:

A grad student with no programming experience can count colonies on their plate photos in under 15 minutes of opening the documentation.
A grad student with Python experience can write a custom analysis module in an afternoon.
The same image and config produce the same results on Windows and Mac, today and a year from now.
At least one external user contributes a module or plugin within the first year of release.
Results are publication-quality: defensible, reproducible, and properly cited (the tool should output the methods/citations needed for a paper).


9. Open Questions and Future Decisions
These are deferred but worth tracking:

Provenance/reproducibility depth: Lightweight (save config alongside results) vs. heavyweight (full DVC-style tracking)?
GUI framework: Napari plugin (leverages existing bioimage community) vs. standalone Qt app (more control) vs. Streamlit (simplest, web-flavored)?
Plugin discovery: Python entry points vs. a plugin folder vs. a registry?
Model hosting: Bundle models in the install (large download) vs. fetch on first use (requires one-time internet)?
Naming and branding: Project name, logo, identity.
Community infrastructure: GitHub Discussions, a forum, a Discord/Slack, or none initially?


10. Anti-Patterns to Avoid
Lessons from existing tools we want to learn from, not repeat:

ImageJ macros: powerful but cryptic; we'll favor readable Python and YAML
CellProfiler: great modularity but heavyweight UI and learning curve; we'll aim for a gentler on-ramp
One-off lab scripts: unmaintainable and unshareable; we'll provide structure without bureaucracy
Black-box "AI" tools: impressive demos, opaque results, irreproducible; we'll show our work