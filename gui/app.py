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

from blenny.modules.classify_interior import InteriorColonyClassifier
from blenny.modules.export_annotated import AnnotatedImageExporter
from blenny.modules.export_csv import CSVExporter
from blenny.modules.export_summary import SummaryExporter
from blenny.pipeline import Pipeline

# --- Globals & Defaults ---
radius_scale_default = 1.0
min_area_ppm_default = 0
min_circ_default = 0.8
min_solidity_default = 0.7
interior_radius_default = 1.0
fallback_ecc_default = 1.0
max_dimension_default = 3200
resize_default = False

# Paths for temporary GUI files
GUI_TEMP_DIR = Path("gui_uploads")
GUI_TEMP_DIR.mkdir(exist_ok=True)
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
    import csv
    if not batch_data: return
    all_measurements = []
    for data in batch_data: all_measurements.extend(data.measurements)
    if not all_measurements: return
    path = Path(output_dir) / "batch_colonies.csv"
    preferred_order = ["plate_label", "label", "centroid_x", "centroid_y", "centroid_x_global", "centroid_y_global", 
                       "area_px", "circularity", "solidity", "eccentricity", "mean_r", "mean_g", "mean_b", 
                       "is_artifact", "artifact_reason", "source"]
    fieldnames = [p for p in preferred_order if any(p in m for m in all_measurements)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_measurements)

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
            st.session_state["pipeline_path"] = p
        elif mode == "Multi-Plate Grid":
            p = "pipeline_yolo_facile_grid.yaml"
            st.session_state["pipeline_path"] = p
        st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1

    plate_mode = st.radio("Plate Detection Mode", ["Auto", "Multi-Plate Grid", "Manual Circle", "Manual Shape"],
                          key="manual_plate_mode", on_change=on_mode_change)

    # Determine suggested pipeline based on mode
    if plate_mode == "Auto":
        suggested = "pipeline_yolo_facile.yaml"
    elif plate_mode == "Multi-Plate Grid":
        suggested = "pipeline_yolo_facile_grid.yaml"
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
            st.rerun()

    uploaded_pipeline = st.file_uploader("OR Upload Pipeline YAML", type=["yaml", "yml"])
    if uploaded_pipeline:
        temp_p = GUI_TEMP_DIR / "uploaded_pipeline.yaml"
        if st.session_state.get("last_uploaded_p") != uploaded_pipeline.name:
            temp_p.write_bytes(uploaded_pipeline.getvalue())
            st.session_state["pipeline_path"] = str(temp_p.resolve())
            st.session_state["last_uploaded_p"] = uploaded_pipeline.name
            st.rerun()

    pipeline_path = st.session_state["pipeline_path"]
    st.divider()
    st.header("3. Masking & Tools")
    
    selected_tool_label = st.radio("Select Drawing Tool", ["View", "Polygon Plate Area", "Exclusion Mask"], 
                                   index=0, horizontal=True, key="active_drawing_tool")
    if selected_tool_label == "Polygon Plate Area" and plate_mode != "Manual Shape":
        st.session_state["manual_plate_mode"] = "Manual Shape"; st.rerun()

    radius_scale = compact_control("Plate Radius Scale", "radius_scale", 0.5, 2.0, radius_scale_default, 0.01, "Radius multiplier.") if plate_mode in ("Auto", "Multi-Plate Grid") else 1.0

    grid_rows, grid_cols = 1, 1
    if plate_mode == "Multi-Plate Grid":
        c_g1, c_g2 = st.columns(2)
        grid_rows = c_g1.number_input("Rows", 1, 10, 2)
        grid_cols = c_g2.number_input("Columns", 1, 10, 3)
        if "grid_labels" not in st.session_state or len(st.session_state["grid_labels"]) != grid_rows:
            st.session_state["grid_labels"] = [[f"{chr(65+r)}{c+1}" for c in range(grid_cols)] for r in range(grid_rows)]
        with st.expander("Edit Labels"):
            for r in range(grid_rows):
                cols_ui = st.columns(grid_cols)
                for c in range(grid_cols):
                    st.session_state["grid_labels"][r][c] = cols_ui[c].text_input(f"L{r}{c}", st.session_state["grid_labels"][r][c], key=f"l_{r}_{c}", label_visibility="collapsed")

    manual_cy, manual_cx, manual_r = None, None, None
    if plate_mode == "Manual Circle":
        manual_cy = compact_control("Center Y", "m_cy", 0, 4000, 1000, 1, "Y center")
        manual_cx = compact_control("Center X", "m_cx", 0, 4000, 1000, 1, "X center")
        manual_r = compact_control("Radius", "m_r", 0, 2000, 800, 1, "Radius")

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
    resize_enabled = st.checkbox("Resize scan for detection", key="resize_enabled")
    max_dimension = st.number_input("Max scan dimension (px)", 100, 10000, 3200, key="max_dimension", disabled=not resize_enabled)
    
    resize_sub_enabled = False; max_sub_dimension = 1280
    if plate_mode == "Multi-Plate Grid":
        resize_sub_enabled = st.checkbox("Resize sub-plates for analysis", key="resize_sub_enabled")
        max_sub_dimension = st.number_input("Max sub-plate dimension (px)", 100, 4000, 1280, key="max_sub_dimension", disabled=not resize_sub_enabled)

    min_area_ppm = compact_control("Min Size (ppm)", "min_area_ppm", 0, 1000, min_area_ppm_default, 1, "Min area.")
    min_circ = compact_control("Min Circularity", "min_circ", 0.0, 1.0, min_circ_default, 0.05, "Min roundness.")
    min_solidity = compact_control("Min Solidity", "min_solidity", 0.0, 1.0, min_solidity_default, 0.05, "Min solidity (area / convex hull area).")
    interior_radius = compact_control("Interior Radius", "int_r", 0.1, 1.0, interior_radius_default, 0.05, "Interior zone.")
    fallback_ecc = compact_control("Max Eccentricity", "f_ecc", 0.1, 1.0, fallback_ecc_default, 0.05, "Max elongation.")
    enable_debug = st.checkbox("Save debug images", value=False)
    generate_annotated = st.checkbox("Generate annotated images", value=True, key="gen_ann_check")
    save_subfolders = st.checkbox("Save as Subfolders", value=True)
    enable_interactive = st.checkbox("Interactive Review", value=True)
    
    if "manual_exclude_ids" not in st.session_state: st.session_state["manual_exclude_ids"] = []
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
    # Clear Masks
    for p in [PLATE_MASK_PATH, EXCLUSION_MASK_PATH]:
        if p.exists(): p.unlink()
    # Clear temp run output
    run_tmp = GUI_TEMP_DIR / "run_output"
    if run_tmp.exists():
        shutil.rmtree(run_tmp)
    # Clear Results
    for key in ["all_results", "result_stems", "batch_runs", "current_view_idx", "gui_analysis_log"]:
        st.session_state.pop(key, None)
    st.session_state["last_input_paths"] = input_names

# --- Main Logic ---
input_paths = []
if input_files:
    for f in input_files:
        p = GUI_TEMP_DIR / f.name; p.write_bytes(f.getvalue()); input_paths.append(p)
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
            if output_folder_input.strip():
                output_dir = Path(output_folder_input)
            else:
                run_tmp = GUI_TEMP_DIR / "run_output"
                if run_tmp.exists():
                    shutil.rmtree(run_tmp)
                output_dir = run_tmp
            output_dir.mkdir(parents=True, exist_ok=True)
            st.session_state.update({"all_results": {}, "result_stems": [], "batch_runs": [], "current_view_idx": 0})
            
            from blenny.config import extract_steps, load_yaml, substitute_paths
            try:
                raw_steps = extract_steps(load_yaml(pipeline_path))
                overrides = {
                    "load_image": {"max_dimension": int(max_dimension) if resize_enabled else None},
                    "detect_plate": {"radius_scale": radius_scale, "crop": False},
                    "detect_facile": {"radius_scale": radius_scale, "crop": False},
                    "detect_multi_plate": {"radius_scale": radius_scale, "grid": [grid_rows, grid_cols], "labels": st.session_state.get("grid_labels")},
                    "threshold_segment": {"min_area_ppm": min_area_ppm},
                    "filter_by_properties": {"min_area_ppm": min_area_ppm, "min_circularity": min_circ, "min_solidity": min_solidity},
                    "classify_by_interior": {"interior_radius_frac": interior_radius, "strict_fallback_max_eccentricity": fallback_ecc},
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
                        data = Pipeline.from_config(res).run(p, output_dir=output_dir, progress_callback=gui_progress)
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
                    st.rerun()

            # Batch Save/Update Button
            save_label = f"Save/Update All results to {Path(output_folder_input).resolve().name}" if output_folder_input.strip() else "Save/Update All results"
            if st.button(save_label, type="primary", width="stretch"):
                if not output_folder_input.strip():
                    st.error("Please specify an output folder to save results.")
                else:
                    save_dir = Path(output_folder_input).resolve()
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
