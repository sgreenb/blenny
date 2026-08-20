import base64
import os
import shutil
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFile

# Allow loading of slightly truncated images (common in some scanner/camera outputs)
ImageFile.LOAD_TRUNCATED_IMAGES = True

from blenny import templates
from blenny.modules.classify_interior import InteriorColonyClassifier
from blenny.modules.export_annotated import AnnotatedImageExporter
from blenny.modules.export_csv import CSVExporter
from blenny.modules.export_summary import SummaryExporter
from blenny.pipeline import Pipeline
from blenny.roi.gui import render_roi_main, render_roi_sidebar

# --- Globals & Defaults ---
radius_scale_default = 1.0
min_area_ppm_default = 0
min_circ_default = 0.0
interior_radius_default = 1.0
fallback_ecc_default = 1.0
yolo_conf_default = 0.15
max_dimension_default = 3200
resize_default = False

# Scratch dir for uploaded files the pipeline must read from disk (gitignored).
# Streamlit's file_uploader hands us in-memory objects, so we write real files
# here; the dir is pruned on process start (below) and on input change (see
# the "Input Change Detection" block) so uploads don't accumulate forever.
GUI_TEMP_DIR = Path("gui_uploads")
GUI_TEMP_DIR.mkdir(exist_ok=True)
if not (GUI_TEMP_DIR / f".pid_{os.getpid()}").exists():
    # First execution of this server process: anything already here belongs to
    # a previous run (uploads are re-written on every rerun while the widget
    # holds them), so drop it.
    for p in GUI_TEMP_DIR.iterdir():
        if p.is_file() and not p.name.startswith(".pid_"):
            try:
                p.unlink()
            except OSError:
                pass
    (GUI_TEMP_DIR / f".pid_{os.getpid()}").touch()
PLATE_MASK_PATH = GUI_TEMP_DIR / "gui_plate_batch_mask.png"
EXCLUSION_MASK_PATH = GUI_TEMP_DIR / "gui_mask_batch_exclusion.png"

# --- Helpers ---

def local_folder_picker(title="Select Folder"):
    """Open a native folder picker."""
    if sys.platform == "darwin":
        cmd = f"osascript -e 'POSIX path of (choose folder with prompt \"{title}\")'"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception: pass
    elif sys.platform == "win32":
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askdirectory(title=title)
            root.destroy()
            return path if path else None
        except Exception: pass
    return None

def compact_control(label, key, min_val, max_val, default_val, step, help_text):
    if key not in st.session_state:
        st.session_state[key] = default_val

    st.markdown(f"**{label}**", help=help_text)
    c1, c2, c3 = st.columns([1.2, 3, 0.8])

    def sync_num():
        st.session_state[key] = st.session_state[f"num_{key}"]
        st.session_state[f"slide_{key}"] = st.session_state[key]

    def sync_slide():
        st.session_state[key] = st.session_state[f"slide_{key}"]
        st.session_state[f"num_{key}"] = st.session_state[key]

    if c3.button("Reset", key=f"reset_{key}"):
        st.session_state[key] = default_val
        st.session_state[f"num_{key}"] = default_val
        st.session_state[f"slide_{key}"] = default_val
        st.rerun()

    if f"num_{key}" not in st.session_state: st.session_state[f"num_{key}"] = st.session_state[key]
    if f"slide_{key}" not in st.session_state: st.session_state[f"slide_{key}"] = st.session_state[key]

    c1.number_input(label, min_value=float(min_val), max_value=float(max_val), step=float(step),
                    key=f"num_{key}", label_visibility="collapsed", on_change=sync_num)
    c2.slider(label, min_value=float(min_val), max_value=float(max_val), step=float(step),
              key=f"slide_{key}", label_visibility="collapsed", on_change=sync_slide)
    return st.session_state[key]

# Monkey-patch for streamlit-drawable-canvas compatibility
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
    try:
        from streamlit.elements.lib.image_utils import image_to_url as _image_to_url
        st_image.image_to_url = _image_to_url
    except ImportError: pass

_orig_image_to_url = st_image.image_to_url
def _compat_image_to_url(image_data, width, *args, **kwargs):
    if image_data is None: return None
    if isinstance(width, int):
        from dataclasses import dataclass
        @dataclass
        class FakeLayoutConfig: width: int
        return _orig_image_to_url(image_data, FakeLayoutConfig(width=width), *args, **kwargs)
    return _orig_image_to_url(image_data, width, *args, **kwargs)
st_image.image_to_url = _compat_image_to_url

from streamlit_drawable_canvas import st_canvas

def generate_batch_summary(batch_data, output_dir):
    import csv
    if len(batch_data) <= 1: return
    summary_path = Path(output_dir) / "batch_summary.csv"
    rows = []
    expected_plate_cols = set()
    for data in batch_data:
        m = data.metadata
        row = {
            "input": data.source or "unknown",
            "stem": m.get("stem", "unknown"),
            "status": "ok",
            "colony_count": m.get("colony_count", 0),
            "n_quality_flags": len(data.quality_flags),
            "flag_codes": "|".join(f.code for f in data.quality_flags),
        }
        if "per_plate_counts" in m:
            for plabel, count in m["per_plate_counts"].items():
                col_name = f"plate_{plabel}_count"
                row[col_name] = count
                expected_plate_cols.add(col_name)
        rows.append(row)
    if not rows: return
    fieldnames = ["input", "stem", "status", "colony_count"] + sorted(list(expected_plate_cols))
    all_keys = set()
    for r in rows: all_keys.update(r.keys())
    for k in sorted(list(all_keys)):
        if k not in fieldnames: fieldnames.append(k)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def generate_batch_colonies(batch_data, output_dir):
    if not batch_data: return
    all_measurements = []
    for data in batch_data: all_measurements.extend(data.measurements)
    # Shared writer keeps every measurement column (the old fixed-column list
    # here silently dropped mean_h/s/v, colony_count_estimate, segment_label,
    # classification and the bbox columns while the CLI kept them).
    from blenny.batch import write_batch_colonies_csv
    write_batch_colonies_csv(Path(output_dir) / "batch_colonies.csv", all_measurements)

# --- App Layout ---
st.set_page_config(page_title="Blenny Plate Reader", layout="wide", initial_sidebar_state="expanded")

logo_path = Path(__file__).parent.parent / "screenshots" / "blenny_logo.png"
if logo_path.exists():
    with open(logo_path, "rb") as f: logo_base64 = base64.b64encode(f.read()).decode()
    st.markdown(f'<div style="display: flex; align-items: center; justify-content: center; gap: 20px;">'
                f'<h1 style="margin: 0;">Blenny Plate Reader</h1>'
                f'<img src="data:image/png;base64,{logo_base64}" width="200"></div>', unsafe_allow_html=True)
else:
    st.markdown("<h1 style='text-align: center;'>Blenny Plate Reader</h1>", unsafe_allow_html=True)

st.markdown("""<style>.main .block-container { min-width: 1000px; padding-top: 2rem !important; }
iframe[title="streamlit_drawable_canvas.st_canvas"] { max-width: 100%; overflow: hidden !important; border: 1px solid #eee; border-radius: 4px; }
div[data-testid="stCustomComponentV1"], div[data-testid="stIFrame"] { overflow-x: auto !important; padding: 5px; }</style>""", unsafe_allow_html=True)

cli_cmd = [sys.executable, "-m", "blenny"]

# --- Sidebar ---
with st.sidebar:
    st.title("Blenny Control Panel")
    mode = st.segmented_control(
        "Mode",
        ["Colony Counting", "ROI Mode"],
        default="Colony Counting",
        key="app_mode",
    )
    st.divider()

# ROI mode takes over both the sidebar and the main area; everything below
# (colony counter) is skipped via st.stop().
if mode == "ROI Mode":
    with st.sidebar:
        render_roi_sidebar()
    render_roi_main()
    st.stop()

# --- Colony-mode sidebar ---
with st.sidebar:
    st.header("1. Data Input")
    input_files = st.file_uploader("Upload Plate Images", type=["jpg", "jpeg", "png", "tif"], accept_multiple_files=True)
    
    c_f1, c_f2 = st.columns([3, 1])
    input_folder = c_f1.text_input("OR Folder Path", value=st.session_state.get("folder_path", ""))
    c_f2.markdown("<div style='height: 29px;'></div>", unsafe_allow_html=True)
    if c_f2.button("Browse", key="browse_input", width="stretch"):
        selected = local_folder_picker("Select Input Folder")
        if selected:
            st.session_state["folder_path"] = selected
            st.rerun()

    c_o1, c_o2 = st.columns([3, 1])
    output_folder_input = c_o1.text_input("Output Folder", value=st.session_state.get("output_folder_path", ""))
    c_o2.markdown("<div style='height: 29px;'></div>", unsafe_allow_html=True)
    if c_o2.button("Browse", key="browse_output", width="stretch"):
        selected = local_folder_picker("Select Output Folder")
        if selected:
            st.session_state["output_folder_path"] = selected
            st.rerun()

    st.divider()
    st.header("2. Analysis Mode")
    
    def on_mode_change():
        mode = st.session_state["manual_plate_mode"]
        if mode == "Auto":
            p = "pipeline_yolo_facile.yaml"
            if not Path(p).exists(): p = "pipeline_yolo.yaml"
            st.session_state["pipeline_path"] = p
        elif mode == "Multi-Plate Grid":
            p = "pipeline_yolo_facile_grid.yaml"
            if not Path(p).exists(): p = "pipeline_multi.yaml"
            st.session_state["pipeline_path"] = p
        st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1
        # Keep the keyed text input in sync: without this, its stale session
        # value re-syncs pipeline_path back to the old file on the same rerun.
        st.session_state["pipeline_path_input"] = st.session_state["pipeline_path"]

    plate_mode = st.radio("Plate Detection Mode", ["Auto", "Multi-Plate Grid", "Manual Circle", "Manual Shape"],
                          key="manual_plate_mode", on_change=on_mode_change)

    # Determine suggested pipeline based on mode
    if plate_mode == "Auto":
        suggested = "pipeline_yolo_facile.yaml"
        if not Path(suggested).exists(): suggested = "pipeline_yolo.yaml"
    elif plate_mode == "Multi-Plate Grid":
        suggested = "pipeline_yolo_facile_grid.yaml"
        if not Path(suggested).exists(): suggested = "pipeline_multi.yaml"
    else:
        suggested = "pipeline_yolo_facile.yaml"

    if "pipeline_path" not in st.session_state:
        st.session_state["pipeline_path"] = suggested

    c_p1, c_p2 = st.columns([3, 1])
    pipeline_path_ui = c_p1.text_input("Pipeline YAML Path", key="pipeline_path_input", value=st.session_state["pipeline_path"])
    st.session_state["pipeline_path"] = pipeline_path_ui
    
    if c_p2.button("Browse", key="browse_pipeline", width="stretch"):
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(title="Select Pipeline YAML", filetypes=[("YAML files", "*.yaml *.yml")])
        root.destroy()
        if selected:
            st.session_state["pipeline_path"] = selected
            st.session_state["pipeline_path_input"] = selected
            st.rerun()

    uploaded_pipeline = st.file_uploader("OR Upload Pipeline YAML", type=["yaml", "yml"])
    if uploaded_pipeline:
        temp_p = GUI_TEMP_DIR / "uploaded_pipeline.yaml"
        if st.session_state.get("last_uploaded_p") != uploaded_pipeline.name:
            temp_p.write_bytes(uploaded_pipeline.getvalue())
            st.session_state["pipeline_path"] = str(temp_p.resolve())
            st.session_state["pipeline_path_input"] = str(temp_p.resolve())
            st.session_state["last_uploaded_p"] = uploaded_pipeline.name
            st.rerun()

    pipeline_path = st.session_state["pipeline_path"]
    st.divider()
    st.header("3. Masking & Tools")
    
    selected_tool_label = st.radio("Select Drawing Tool", ["View", "Polygon Plate Area", "Exclusion Mask"], 
                                   index=0, horizontal=True, key="active_drawing_tool")
    if selected_tool_label == "Polygon Plate Area" and plate_mode != "Manual Shape":
        st.session_state["manual_plate_mode"] = "Manual Shape"; st.rerun()

    radius_scale = compact_control("Plate Radius Scale", "radius_scale", 0.5, 2.0, radius_scale_default, 0.01, "Multiplier applied to the detected plate radius. <1 shrinks the analysis area (margin around the rim), >1 expands it — useful for tilted or off-centre plates.") if plate_mode in ("Auto", "Multi-Plate Grid") else 1.0

    grid_rows, grid_cols = 1, 1
    if plate_mode == "Multi-Plate Grid":
        c_g1, c_g2 = st.columns(2)
        grid_rows = c_g1.number_input("Rows", 1, 10, 2)
        grid_cols = c_g2.number_input("Columns", 1, 10, 3)
        # Preserve any user-typed labels, but always match the current grid
        # shape. (Rebuilding only on a ROW change used to IndexError when the
        # user changed Columns, and left stale labels when shrinking.)
        labels = st.session_state.get("grid_labels")
        if labels is None or len(labels) != grid_rows or any(
            len(row) != grid_cols for row in labels
        ):
            if labels is None:
                labels = [[f"{chr(65 + r)}{c + 1}" for c in range(grid_cols)] for r in range(grid_rows)]
            else:
                while len(labels) < grid_rows:
                    labels.append([f"{chr(65 + len(labels))}{c + 1}" for c in range(grid_cols)])
                while len(labels) > grid_rows:
                    labels.pop()
                for r, row in enumerate(labels):
                    if len(row) < grid_cols:
                        labels[r] = row + [f"{chr(65 + r)}{c + 1}" for c in range(len(row), grid_cols)]
                    elif len(row) > grid_cols:
                        labels[r] = row[:grid_cols]
            st.session_state["grid_labels"] = labels
        with st.expander("Edit Labels"):
            for r in range(grid_rows):
                cols_ui = st.columns(grid_cols)
                for c in range(grid_cols):
                    st.session_state["grid_labels"][r][c] = cols_ui[c].text_input(f"L{r}{c}", st.session_state["grid_labels"][r][c], key=f"l_{r}_{c}", label_visibility="collapsed")

    manual_cy, manual_cx, manual_r = None, None, None
    if plate_mode == "Manual Circle":
        manual_cy = compact_control("Center Y", "m_cy", 0, 4000, 1000, 1, "Pixel Y (vertical) coordinate of the plate centre in the uploaded image, for Manual Circle mode.")
        manual_cx = compact_control("Center X", "m_cx", 0, 4000, 1000, 1, "Pixel X (horizontal) coordinate of the plate centre in the uploaded image, for Manual Circle mode.")
        manual_r = compact_control("Radius", "m_r", 0, 2000, 800, 1, "Plate radius in pixels for Manual Circle mode. Pick a value just inside the plate rim.")

    brush_size = st.slider("Brush Size", 1, 100, 20, key="mask_brush_size")
    c_cl1, c_cl2 = st.columns(2)
    if c_cl1.button("Clear Plate", width="stretch"): 
        if PLATE_MASK_PATH.exists(): PLATE_MASK_PATH.unlink()
        st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1; st.rerun()
    if c_cl2.button("Clear Mask", width="stretch"):
        if EXCLUSION_MASK_PATH.exists(): EXCLUSION_MASK_PATH.unlink()
        st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1; st.rerun()

    st.divider()
    st.header("4. Tuning")

    # Which tuning-relevant steps does the active pipeline actually contain?
    # Sliders are only shown when they will have an effect (the GUI only
    # ships YOLO pipelines, whose sub_pipeline contains classify_by_interior
    # but not threshold_segment).
    from blenny.config import extract_steps as _extract_steps
    from blenny.config import load_yaml as _load_yaml

    pipeline_step_names: set[str] = set()
    try:
        def _collect_step_names(steps):
            for s in steps:
                pipeline_step_names.add(s.get("name", ""))
                if s.get("name") == "sub_pipeline":
                    _collect_step_names(s.get("params", {}).get("steps", []))

        _collect_step_names(_extract_steps(_load_yaml(pipeline_path)))
    except Exception:
        pass  # unreadable pipeline -> assume nothing is tunable
    has_threshold = "threshold_segment" in pipeline_step_names
    has_interior = "classify_by_interior" in pipeline_step_names
    has_filter = "filter_colonies" in pipeline_step_names

    resize_enabled = st.checkbox("Resize scan for detection", key="resize_enabled", value=resize_default)
    max_dimension = st.number_input("Max scan dimension (px)", 100, 10000, max_dimension_default, key="max_dimension", disabled=not resize_enabled)
    
    resize_sub_enabled = False; max_sub_dimension = 1280
    if plate_mode == "Multi-Plate Grid":
        resize_sub_enabled = st.checkbox("Resize sub-plates for analysis", key="resize_sub_enabled")
        max_sub_dimension = st.number_input("Max sub-plate dimension (px)", 100, 4000, 1280, key="max_sub_dimension", disabled=not resize_sub_enabled)

    yolo_conf = compact_control(
        "YOLO Confidence", "yolo_conf", 0.0, 1.0, yolo_conf_default, 0.01,
        "Minimum confidence for YOLO colony detections (0-1). Lower = more colonies "
        "detected but more false positives; higher = fewer, higher-confidence "
        "detections. Only used by pipelines with a yolo_detector step.",
    )

    if has_threshold or has_filter:
        min_area_ppm = compact_control("Min Size (ppm)", "min_area_ppm", 0, 1000, min_area_ppm_default, 1, "Smallest colony kept, in parts-per-million (ppm) of the plate area — 1 ppm = one millionth of the plate. On a 90 mm plate, 15 ppm ≈ 0.1 mm². 0 (the default) = no minimum, i.e. no area filtering. Applies during segmentation (Classic) or as a post-YOLO filter. Raise to ignore fine debris.")
        min_circ = compact_control("Min Circularity", "min_circ", 0.0, 1.0, min_circ_default, 0.05, "Roundness filter (1.0 = perfect circle). Detections below this are rejected as artefacts — rim arcs and smudges score <0.5. Applies during segmentation (Classic) or as a post-YOLO filter. Set to 0 to disable.")
    else:
        min_area_ppm, min_circ = min_area_ppm_default, min_circ_default

    if has_interior:
        interior_radius = compact_control("Interior Radius", "int_r", 0.1, 1.0, interior_radius_default, 0.05, "Fraction of the plate radius treated as the trusted 'interior' reference for artifact rejection. Edge-zone detections are scored against interior colonies; raise toward 1.0 to keep more edge colonies, lower to reject more.")
        fallback_ecc = compact_control("Max Eccentricity", "f_ecc", 0.1, 1.0, fallback_ecc_default, 0.05, "Elongation cap for the strict fallback filter used when too few interior colonies exist to build a reference. Detections more elongated than this are marked as artefacts; 0.55 is a typical value, 1.0 disables the filter.")
    else:
        interior_radius, fallback_ecc = interior_radius_default, fallback_ecc_default

    enable_debug = st.checkbox("Save debug images", value=False)
    generate_annotated = st.checkbox("Generate annotated images", value=True, key="gen_ann_check")
    save_subfolders = st.checkbox("Save as Subfolders", value=True)
    enable_interactive = st.checkbox("Interactive Review", value=True)
    
    st.divider()
    run_btn = st.button("Run Analysis", type="primary", width="stretch")

# --- Input Change Detection ---
# Clear manual masks and results if the input images change
if "last_input_paths" not in st.session_state:
    st.session_state["last_input_paths"] = []

input_names = [f.name for f in input_files] if input_files else []
if input_folder and Path(input_folder).is_dir():
    from blenny.modules.load_image import IMAGE_EXTENSIONS
    input_names.extend([f.name for f in Path(input_folder).iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])

if input_names != st.session_state["last_input_paths"]:
    # Remove the temp copies we wrote for the previous input set so the
    # scratch dir doesn't hold every upload forever.
    for p in st.session_state.get("gui_written_uploads", []):
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass
    st.session_state["gui_written_uploads"] = []
    # Clear Masks
    for p in [PLATE_MASK_PATH, EXCLUSION_MASK_PATH]:
        if p.exists(): p.unlink()
    # Clear Results
    for key in ["all_results", "result_stems", "batch_runs", "current_view_idx", "gui_analysis_log"]:
        st.session_state.pop(key, None)
    st.session_state["last_input_paths"] = input_names

# --- Main area ----------------------------------------------------------------
if mode == "Colony Counting":
    # --- Main Logic ---
    input_paths = []
    if input_files:
        written = []
        for f in input_files:
            p = GUI_TEMP_DIR / f.name
            p.write_bytes(f.getvalue())
            input_paths.append(p)
            written.append(str(p))
        st.session_state["gui_written_uploads"] = written
    elif input_folder and Path(input_folder).is_dir():
        from blenny.modules.load_image import IMAGE_EXTENSIONS
        input_paths = sorted([f for f in Path(input_folder).iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])

    if input_paths:
        ref_img = Image.open(input_paths[0]); ref_img = ImageOps.exif_transpose(ref_img).convert("RGB")
        scale = min(1200 / ref_img.width, 1000 / ref_img.height, 1.0)
        canvas_w, canvas_height = int(ref_img.width * scale), int(ref_img.height * scale)
        canvas_bg_img = ref_img.copy()

        # Drawing context overlays
        if plate_mode == "Multi-Plate Grid":
            from PIL import ImageDraw
            d = ImageDraw.Draw(canvas_bg_img); h, w = canvas_bg_img.height, canvas_bg_img.width
            ch, cw = h / grid_rows, w / grid_cols
            for r in range(1, grid_rows): d.line([(0, r*ch), (w, r*ch)], fill=(0,255,255), width=2)
            for c in range(1, grid_cols): d.line([(c*cw, 0), (c*cw, h)], fill=(0,255,255), width=2)

        active_tool = {"View": None, "Polygon Plate Area": "Define Plate Area", "Exclusion Mask": "Paint Exclusion Mask"}[selected_tool_label]
        canvas_bg = canvas_bg_img.resize((canvas_w, canvas_height), Image.Resampling.LANCZOS)

        col1, col2 = st.columns(2)
        with col1:
            if active_tool:
                mode = "polygon" if active_tool == "Define Plate Area" else "freedraw"
                canvas = st_canvas(fill_color="rgba(0,0,255,0.3)", stroke_width=2 if mode=="polygon" else brush_size,
                                   stroke_color="#0000FF", background_image=canvas_bg, height=canvas_height, width=canvas_w,
                                   drawing_mode=mode, key=f"canvas_{st.session_state.get('canvas_version', 0)}")
                if canvas.image_data is not None and np.any(canvas.image_data[:,:,3] > 0):
                    mask = Image.fromarray((canvas.image_data[:,:,3] > 0).astype(np.uint8)*255).resize(ref_img.size, Image.Resampling.NEAREST)
                    mask.save(PLATE_MASK_PATH if active_tool == "Define Plate Area" else EXCLUSION_MASK_PATH)
            else:
                st.image(canvas_bg_img, width="stretch")

        with col2:
            if run_btn:
                if not output_folder_input.strip(): st.error("Specify output folder."); st.stop()
                output_dir = Path(output_folder_input); output_dir.mkdir(parents=True, exist_ok=True)
                st.session_state.update({"all_results": {}, "result_stems": [], "batch_runs": [], "current_view_idx": 0})
            
                from blenny.config import extract_steps, load_yaml, substitute_paths
                try:
                    raw_steps = extract_steps(load_yaml(pipeline_path))

                    if not generate_annotated:
                        # Mirror the CLI's --no-annotated-images: drop every
                        # export_annotated step, including inside sub_pipeline.
                        def strip_annotated(steps):
                            out = []
                            for s in steps:
                                if s["name"] == "export_annotated":
                                    continue
                                if s["name"] == "sub_pipeline":
                                    s["params"]["steps"] = strip_annotated(
                                        s.get("params", {}).get("steps", [])
                                    )
                                out.append(s)
                            return out

                        raw_steps = strip_annotated(raw_steps)

                    if not save_subfolders:
                        # Mirror the CLI's --flat: flatten {output_dir}/{stem}/ paths.
                        def flatten_paths(steps):
                            for s in steps:
                                if "params" in s:
                                    for k, v in s["params"].items():
                                        if isinstance(v, str) and "{output_dir}/{stem}/" in v:
                                            rest = v.split("{output_dir}/{stem}/", 1)[1]
                                            s["params"][k] = (
                                                "{output_dir}/" + rest
                                                if rest.startswith("{stem}_")
                                                else "{output_dir}/{stem}_" + rest
                                            )
                                if s["name"] == "sub_pipeline":
                                    flatten_paths(s.get("params", {}).get("steps", []))

                        flatten_paths(raw_steps)

                    overrides = {
                        "load_image": {"max_dimension": int(max_dimension) if resize_enabled else None},
                        "detect_plate": {"radius_scale": radius_scale},
                        "detect_facile": {"radius_scale": radius_scale},
                        "detect_multi_plate": {"radius_scale": radius_scale, "grid": [grid_rows, grid_cols], "labels": st.session_state.get("grid_labels")},
                        "threshold_segment": {"min_area_ppm": min_area_ppm, "min_circularity": min_circ},
                        "filter_colonies": {"min_area_ppm": min_area_ppm, "min_circularity": min_circ},
                        "classify_by_interior": {"interior_radius_frac": interior_radius, "strict_fallback_max_eccentricity": fallback_ecc},
                        "yolo_detector": {"conf_threshold": yolo_conf},
                        "sub_pipeline": {"max_subplate_dimension": int(max_sub_dimension) if resize_sub_enabled else None}
                    }
                    if plate_mode == "Manual Circle":
                        for k in ["detect_plate", "detect_facile"]: overrides[k].update({"force_cy": manual_cy, "force_cx": manual_cx, "force_r": manual_r})
                    if plate_mode == "Manual Shape" and PLATE_MASK_PATH.exists():
                        for k in ["detect_plate", "detect_facile"]: overrides[k].update({"force_mask_path": str(PLATE_MASK_PATH)})
                    if EXCLUSION_MASK_PATH.exists(): overrides["apply_exclusion_mask"] = {"mask_path": str(EXCLUSION_MASK_PATH)}

                    def apply_overrides(steps):
                        for s in steps:
                            if s["name"] in overrides: s.setdefault("params", {}).update(overrides[s["name"]])
                            if s["name"] == "sub_pipeline": apply_overrides(s["params"].get("steps", []))
                    apply_overrides(raw_steps)

                    t_batch_start = time.perf_counter()
                    with st.status("Analyzing...", expanded=True) as status:
                        # Clear log area for a fresh run
                        st.session_state["gui_analysis_log"] = ""
                        log_container = st.empty()
                    
                        for i, p in enumerate(input_paths):
                            status.update(label=f"Analyzing {p.name} ({i+1}/{len(input_paths)})...")
                            st.session_state["gui_analysis_log"] += f"### {p.name}\n"
                            log_container.markdown(st.session_state["gui_analysis_log"])
                        
                            def gui_progress(current, total, name):
                                # Append to persistent session log
                                st.session_state["gui_analysis_log"] += f"&nbsp;&nbsp;[{current}/{total}] `{name}`  \n"
                                log_container.markdown(st.session_state["gui_analysis_log"])
                        
                            t_p_start = time.perf_counter()
                            res = substitute_paths(raw_steps, input_path=p, output_dir=output_dir)
                            dbg_dir = Path(output_dir) / "debug" / p.stem if enable_debug else None
                            data = Pipeline.from_config(res).run(
                                p, output_dir=output_dir, debug_dir=dbg_dir,
                                progress_callback=gui_progress,
                            )
                            t_p_elapsed = time.perf_counter() - t_p_start
                            st.session_state["gui_analysis_log"] += f"&nbsp;&nbsp;**Plate analysis complete in {t_p_elapsed:.2f}s**  \n\n"
                            log_container.markdown(st.session_state["gui_analysis_log"])

                            st.session_state["batch_runs"].append((data, Pipeline.from_config(res)))
                            if enable_interactive:
                                if "multi_plate_results" in data.metadata:
                                    for sr in data.metadata["multi_plate_results"]:
                                        plabel = sr.metadata.get("plate_label", "unknown")
                                        # Inject plate_label into sub-result measurements for UI display
                                        for m in sr.measurements:
                                            m["plate_label"] = plabel
                                        n = f"{p.stem} [{plabel}]"
                                        st.session_state["all_results"][n] = sr; st.session_state["result_stems"].append(n)
                                else:
                                    st.session_state["all_results"][p.stem] = data; st.session_state["result_stems"].append(p.stem)
                        generate_batch_summary([d for d, _ in st.session_state["batch_runs"]], output_dir)
                        t_batch_elapsed = time.perf_counter() - t_batch_start
                        status.update(label=f"Complete! ({t_batch_elapsed:.2f}s)", state="complete", expanded=True)
                except Exception as e: st.error(f"Error: {e}")

            # --- Interactive Review ---
            if st.session_state.get("all_results"):
                stems = st.session_state["result_stems"]; idx = st.session_state["current_view_idx"]
                c_nav1, c_nav2, c_nav3 = st.columns([1, 2, 1])
                if c_nav1.button("←", disabled=idx==0, width="stretch"): st.session_state["current_view_idx"] -= 1; st.rerun()
                st.session_state["current_view_idx"] = stems.index(c_nav2.selectbox("Plate", stems, index=idx, label_visibility="collapsed"))
                if c_nav3.button("→", disabled=idx==len(stems)-1, width="stretch"): st.session_state["current_view_idx"] += 1; st.rerun()
            
                stem = stems[st.session_state["current_view_idx"]]; data = st.session_state["all_results"][stem]
                import pandas as pd
                InteriorColonyClassifier.update_count(data.measurements, data)
            
                # Use original pipeline exporters if possible
                annotator = AnnotatedImageExporter(output_path="d.png")
                summarizer = SummaryExporter(output_path="d.txt")
                csv_exporter = CSVExporter(output_path="d.csv")
            
                img = annotator.render(data)
                if img is not None:
                    st.image(img, width="stretch")
                else:
                    st.warning("Annotation failed. Displaying original/preprocessed image.")
                    display_img = data.image if data.image is not None else data.original_image
                    if display_img is not None:
                        st.image(display_img, width="stretch")
                    else:
                        st.error("No image available to display.")
            
                # --- Download Section ---
                c_dl1, c_dl2, c_dl3 = st.columns(3)
                c_dl1.download_button("Download CSV", csv_exporter.generate_csv(data), file_name=f"{stem}_colonies.csv", mime="text/csv", width="stretch")
                c_dl2.download_button("Download Summary", summarizer.generate_text(data), file_name=f"{stem}_run_summary.txt", mime="text/plain", width="stretch")
            
                if img is not None:
                    import io
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    c_dl3.download_button("Download Image", buf.getvalue(), file_name=f"{stem}_annotated.png", mime="image/png", width="stretch")
                else:
                    c_dl3.button("Download Image", disabled=True, width="stretch")

                df = pd.DataFrame(data.measurements)
            
                # Ensure required columns exist for display and editing
                for col in ["plate_label", "label", "is_artifact", "centroid_x", "centroid_y", "area_px", "colony_count_estimate", "circularity", "solidity"]:
                    if col not in df.columns:
                        df[col] = 0 if any(k in col for k in ["centroid", "area", "circ", "solid"]) else (False if col == "is_artifact" else 1 if col == "colony_count_estimate" else "N/A")

                if "Type" not in df.columns:
                    df["Type"] = df.apply(lambda r: f"Merged(x{int(r.get('colony_count_estimate', 1))})" if int(r.get('colony_count_estimate', 1)) > 1 else "Colony", axis=1)
                    df.loc[df['is_artifact'] == True, 'Type'] = "Artifact"

                # Filter to final display set
                display_cols = ["plate_label", "label", "is_artifact", "Type", "centroid_x", "centroid_y", "area_px", "circularity", "solidity"]
                cols = [c for c in display_cols if c in df.columns]
            
                edited = st.data_editor(df[cols], key=f"ed_{stem}", hide_index=True, width="stretch",
                                        column_config={
                                            "is_artifact": st.column_config.CheckboxColumn("Artifact?"),
                                            "label": st.column_config.TextColumn("ID", disabled=True),
                                            "Type": st.column_config.TextColumn("Class", disabled=True),
                                            "centroid_x": st.column_config.NumberColumn("X", disabled=True, format="%.0f"),
                                            "centroid_y": st.column_config.NumberColumn("Y", disabled=True, format="%.0f"),
                                            "area_px": st.column_config.NumberColumn("Area", disabled=True),
                                            "circularity": st.column_config.NumberColumn("Circ", disabled=True, format="%.2f"),
                                            "solidity": st.column_config.NumberColumn("Solid", disabled=True, format="%.2f"),
                                        })
            
                if st.session_state.get(f"ed_{stem}"):
                    edits = st.session_state[f"ed_{stem}"]["edited_rows"]
                    if edits:
                        for r_idx, changes in edits.items():
                            if "is_artifact" in changes:
                                data.measurements[r_idx]["is_artifact"] = changes["is_artifact"]
                                # Also update main parent data if this is a sub-plate
                                if " [" in stem:
                                    parent_stem = stem.split(" [")[0]
                                    parent_data = next((d for d, p in st.session_state["batch_runs"] if d.metadata.get("stem") == parent_stem), None)
                                    if parent_data:
                                        # Find matching row in parent data. Measurements in parent
                                        # include global IDs and plate_label.
                                        plabel = data.metadata.get("plate_label")
                                        # Local label is the index in sub-plate. 
                                        # SubPipeline maps measurements linearly.
                                        # However, it's safer to match by plate_label + local label.
                                        local_label = data.measurements[r_idx].get("label")
                                        for pm in parent_data.measurements:
                                            if pm.get("plate_label") == plabel and pm.get("label") == local_label:
                                                pm["is_artifact"] = changes["is_artifact"]
                                                break
                                        InteriorColonyClassifier.update_count(parent_data.measurements, parent_data)
                        # Consume the edit event so it is not re-applied on every
                        # subsequent rerun (the widget state otherwise keeps the
                        # stale edited_rows until the user edits again).
                        st.session_state[f"ed_{stem}"]["edited_rows"] = {}
                        st.rerun()

                # Batch Save/Update Button
                save_dir = Path(output_folder_input).resolve()
                if st.button(f"Save/Update All results to {save_dir.name}", type="primary", width="stretch"):
                    save_dir.mkdir(parents=True, exist_ok=True)
                    for d, p in st.session_state.get("batch_runs", []):
                        old_out = d.metadata.get("output_dir")
                        d.metadata["output_dir"] = str(save_dir)
                        for step in p.steps:
                            if hasattr(step, "export"): step.export(d)
                        if old_out: d.metadata["output_dir"] = old_out
                    generate_batch_summary([d for d, _ in st.session_state["batch_runs"]], save_dir)
                    generate_batch_colonies([d for d, _ in st.session_state["batch_runs"]], save_dir)
                    st.success(f"Saved results to {save_dir}")
    else:
        st.info("Upload images to begin.")

    st.divider()
    st.caption("Blenny GUI v0.2 • Engine: " + subprocess.run(cli_cmd + ["--version"], capture_output=True, text=True).stdout.strip())
