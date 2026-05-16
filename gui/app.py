import os
import shutil
import subprocess
import sys
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

# --- Helpers ---


def local_folder_picker(title="Select Folder"):
    """
    Open a native folder picker.
    """
    if sys.platform == "darwin":
        cmd = f"osascript -e 'POSIX path of (choose folder with prompt \"{title}\")'"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"macOS folder picker error: {e}")
    elif sys.platform == "win32":
        try:
            # Use tkinter as a more reliable fallback on Windows
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askdirectory(title=title)
            root.destroy()
            return path if path else None
        except Exception as e:
            print(f"Windows folder picker error: {e}")
    return None


# Monkey-patch for streamlit-drawable-canvas compatibility with newer Streamlit versions
import streamlit.elements.image as st_image  # noqa: E402

if not hasattr(st_image, "image_to_url"):
    try:
        from streamlit.elements.lib.image_utils import image_to_url as _image_to_url

        st_image.image_to_url = _image_to_url
    except ImportError:
        # Try another location if needed for even newer versions
        pass

# Fix for streamlit-drawable-canvas compatibility with Streamlit 1.34+
# where image_to_url expects a LayoutConfig object instead of a width integer.
_orig_image_to_url = st_image.image_to_url


def _compat_image_to_url(image_data, width, *args, **kwargs):
    if isinstance(width, int):
        from dataclasses import dataclass

        @dataclass
        class FakeLayoutConfig:
            width: int

        return _orig_image_to_url(image_data, FakeLayoutConfig(width=width), *args, **kwargs)
    return _orig_image_to_url(image_data, width, *args, **kwargs)


st_image.image_to_url = _compat_image_to_url

from streamlit_drawable_canvas import st_canvas  # noqa: E402

def generate_batch_summary(batch_data, output_dir):
    """Generate batch-level batch_summary.csv."""
    import csv

    if len(batch_data) <= 1:
        return

    summary_path = Path(output_dir) / "batch_summary.csv"

    rows = []
    expected_plate_cols = set()

    for data in batch_data:
        m = data.metadata
        row = {
            "input": data.source or "unknown",
            "stem": m.get("stem", "unknown"),
            "status": "ok",  # GUI only stores successful runs
            "colony_count": m.get("colony_count", 0),
            "n_quality_flags": len(data.quality_flags),
            "flag_codes": "|".join(f.code for f in data.quality_flags),
        }
        # Add per-plate counts if present
        if "per_plate_counts" in m:
            for plabel, count in m["per_plate_counts"].items():
                col_name = f"plate_{plabel}_count"
                row[col_name] = count
                expected_plate_cols.add(col_name)
        rows.append(row)

    if not rows:
        return

    # 1. Write batch_summary.csv
    # Build fieldnames: standard first, then plate counts, then others
    fieldnames = ["input", "stem", "status", "colony_count"]
    sorted_plate_cols = sorted(list(expected_plate_cols))
    fieldnames.extend(sorted_plate_cols)

    # Add any remaining keys
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    for k in sorted(list(all_keys)):
        if k not in fieldnames:
            fieldnames.append(k)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_batch_colonies(batch_data, output_dir):
    """Generate batch-level batch_colonies.csv."""
    import csv

    if not batch_data:
        return

    all_measurements = []
    for data in batch_data:
        all_measurements.extend(data.measurements)

    if not all_measurements:
        return

    path = Path(output_dir) / "batch_colonies.csv"

    # Define the exact preferred order from CSVExporter to match its format
    preferred_order = [
        "plate_label",
        "label",
        "centroid_x",
        "centroid_y",
        "centroid_x_global",
        "centroid_y_global",
        "area_px",
        "circularity",
        "solidity",
        "eccentricity",
        "mean_r",
        "mean_g",
        "mean_b",
        "mean_h",
        "mean_s",
        "mean_v",
        "is_artifact",
        "artifact_reason",
        "source",
    ]

    fieldnames = []
    for p in preferred_order:
        if any(p in m for m in all_measurements):
            fieldnames.append(p)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_measurements)


# Set page config for a clean, professional look
st.set_page_config(
    page_title="Blenny Plate Reader",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Blenny Plate Reader")

# Allow the drawable-canvas iframe (and its column) to scroll horizontally
# instead of clipping the right edge of the plate preview when the browser
# window is narrower than the canvas pixel width. Streamlit components live
# inside an <iframe> whose enclosing block has overflow:hidden by default.
st.markdown(
    """
    <style>
    /* The component iframe itself */
    iframe[title="streamlit_drawable_canvas.st_canvas"] {
        max-width: 100%;
        overflow: hidden !important;
        border: 1px solid #eee;
        border-radius: 4px;
    }
    /* The Streamlit-generated wrapper around components */
    div[data-testid="stCustomComponentV1"],
    div[data-testid="stIFrame"] {
        overflow-x: auto !important;
        padding: 5px; /* Added buffer to help with mouse event boundaries */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# CLI execution command
cli_cmd = [sys.executable, "-m", "blenny"]

# Paths for temporary GUI files
GUI_TEMP_DIR = Path("gui_uploads")
if "gui_initialized" not in st.session_state:
    if GUI_TEMP_DIR.exists():
        shutil.rmtree(GUI_TEMP_DIR)
    GUI_TEMP_DIR.mkdir(exist_ok=True)
    st.session_state["gui_initialized"] = True

PLATE_MASK_PATH = GUI_TEMP_DIR / "gui_plate_batch_mask.png"
EXCLUSION_MASK_PATH = GUI_TEMP_DIR / "gui_mask_batch_exclusion.png"

# --- Sidebar: Configuration ---
with st.sidebar:
    st.title("Blenny Control Panel")

    # 1. Data Input
    st.header("1. Data Input")
    input_files = st.file_uploader(
        "Upload Plate Images", type=["jpg", "jpeg", "png", "tif"], accept_multiple_files=True
    )

    # Input Folder Picker
    c_f1, c_f2 = st.columns([3, 1])
    input_folder = c_f1.text_input(
        "OR Folder Path",
        value=st.session_state.get("folder_path", ""),
        help="Path to a directory on your machine.",
    )
    c_f2.markdown("<div style='height: 29px;'></div>", unsafe_allow_html=True)
    if c_f2.button("Browse", key="browse_input", help="Browse for input folder", width="stretch"):
        selected = local_folder_picker("Select Input Folder")
        if selected:
            st.session_state["folder_path"] = selected
            st.rerun()

    # Output Folder Picker
    c_o1, c_o2 = st.columns([3, 1])
    output_folder_input = c_o1.text_input(
        "Output Folder",
        value=st.session_state.get("output_folder_path", ""),
        help="Directory where results will be saved. [Required]",
    )
    c_o2.markdown("<div style='height: 29px;'></div>", unsafe_allow_html=True)
    if c_o2.button("Browse", key="browse_output", help="Browse for output folder", width="stretch"):
        selected = local_folder_picker("Select Output Folder")
        if selected:
            st.session_state["output_folder_path"] = selected
            st.rerun()

    input_folder = st.session_state.get("folder_path", "")
    output_folder_input = st.session_state.get("output_folder_path", "")

    st.divider()

    # 2. Pipeline Selection
    st.header("2. Analysis Pipeline")

    # Pipeline choice: YOLO vs Classic
    pipeline_mode = st.radio(
        "Analysis Engine",
        ["YOLO ML", "Classic CV"],
        index=0,
        horizontal=True,
        help="Classic CV uses edge detection and thresholding. YOLO ML uses a trained neural network.",
    )

    repro_file = st.file_uploader(
        "Load Reproducible Config", type=["yaml", "yml"], help="Load settings from a previous run."
    )

    # Defaults
    radius_scale_default, min_ppm_default, min_circ_default = 1.0, 15, 0.7
    interior_radius_default = 0.85
    fallback_ecc_default = 0.55
    max_dimension_default = 2000
    resize_default = False

    if pipeline_mode == "YOLO ML":
        min_ppm_default = 5
        min_circ_default = 0.0
        max_dimension_default = 1280
        resize_default = False
        radius_scale_default = 1.0

    if repro_file:
        try:
            import yaml

            repro_data = yaml.safe_load(repro_file)
            st.success("Config loaded. Settings applied below.")
            steps = {s["name"]: s.get("params", {}) for s in repro_data.get("steps", [])}
            radius_scale_default = float(steps.get("detect_plate", {}).get("radius_scale", 1.0))
            min_ppm_default = int(steps.get("threshold_segment", {}).get("min_area_ppm", 15))
            min_circ_default = float(steps.get("threshold_segment", {}).get("min_circularity", 0.7))
            interior_radius_default = float(
                steps.get("classify_by_interior", {}).get("interior_radius_frac", 0.85)
            )
            fallback_ecc_default = float(
                steps.get("classify_by_interior", {}).get("strict_fallback_max_eccentricity", 0.55)
            )
            loaded_max_dim = steps.get("load_image", {}).get("max_dimension")
            if loaded_max_dim is not None:
                max_dimension_default = int(loaded_max_dim)
                resize_default = True
        except Exception as e:
            st.error(f"Error loading config: {e}")

    if (
        not Path("pipeline_classic.yaml").exists()
        and not Path("pipeline_yolo.yaml").exists()
        and st.button("Generate Default Pipelines")
    ):
        subprocess.run(cli_cmd + ["init"])
        st.success("Created pipeline_classic.yaml, pipeline_yolo.yaml, and pipeline_multi.yaml")

    if pipeline_mode == "YOLO ML":
        default_pipeline = "pipeline_yolo.yaml"
    else:
        default_pipeline = "pipeline_classic.yaml"

    if not Path(default_pipeline).exists():
        # Try finding it relative to the app root if it doesn't exist in CWD
        root_pipeline = Path(__file__).parent.parent / default_pipeline
        if root_pipeline.exists():
            default_pipeline = str(root_pipeline.resolve())

    pipeline_path = st.text_input("Pipeline Path", value=default_pipeline)

    st.divider()

    # 3. Plate Area & Masking
    st.header("3. Plate Area & Masking")

    # Compact Sliders with editable values and individual resets
    def compact_control(label, key, min_val, max_val, default_val, step, help_text):
        if key not in st.session_state:
            st.session_state[key] = default_val

        # Label Row - Attach help here so it's always visible next to the title
        st.markdown(f"**{label}**", help=help_text)

        # Control Row: [Input] [Slider] [Reset]
        c1, c2, c3 = st.columns([1.2, 3, 0.8])

        # Use on_change to sync the two widgets without lag
        def sync_num():
            new_val = st.session_state[f"num_{key}"]
            st.session_state[key] = new_val
            st.session_state[f"slide_{key}"] = new_val

        def sync_slide():
            new_val = st.session_state[f"slide_{key}"]
            st.session_state[key] = new_val
            st.session_state[f"num_{key}"] = new_val

        # Handle individual reset BEFORE widgets are instantiated
        if c3.button("Reset", key=f"reset_{key}", help=f"Reset {label}"):
            st.session_state[key] = default_val
            st.session_state[f"num_{key}"] = default_val
            st.session_state[f"slide_{key}"] = default_val
            st.rerun()

        # Ensure sub-keys are initialized
        if f"num_{key}" not in st.session_state:
            st.session_state[f"num_{key}"] = st.session_state[key]
        if f"slide_{key}" not in st.session_state:
            st.session_state[f"slide_{key}"] = st.session_state[key]

        c1.number_input(
            label,
            min_value=float(min_val) if isinstance(step, float) else int(min_val),
            max_value=float(max_val) if isinstance(step, float) else int(max_val),
            step=step,
            key=f"num_{key}",
            label_visibility="collapsed",
            on_change=sync_num,
        )

        c2.slider(
            label,
            min_value=float(min_val) if isinstance(step, float) else int(min_val),
            max_value=float(max_val) if isinstance(step, float) else int(max_val),
            step=step,
            key=f"slide_{key}",
            label_visibility="collapsed",
            help=help_text,
            on_change=sync_slide,
        )

        return st.session_state[key]

    # Tool labels for the radio button
    # We define this BEFORE the mode radio to determine if we should force a mode.
    # Note: We use a key for the tool so we can query it safely.
    selected_tool_label = st.radio(
        "Select Drawing Tool",
        options=["View", "Polygon Plate Area", "Exclusion Mask"],
        index=0,
        horizontal=True,
        key="active_drawing_tool",
        help="View: Preview image and overlays. Plate Area: Define the circular/polygonal analysis area. Exclusion Mask: Paint areas to ignore.",
    )

    # Determine if we should force "Manual Shape" because the Plate tool is active.
    # We do this logic BEFORE the mode radio widget is created to avoid the crash.
    is_plate_tool = selected_tool_label == "Polygon Plate Area"
    current_mode = st.session_state.get("manual_plate_mode", "Auto")

    if is_plate_tool and current_mode != "Manual Shape":
        st.session_state["manual_plate_mode"] = "Manual Shape"
        # No rerun needed yet, we haven't drawn the radio button below.

    # If the user switches mode to "Auto", we should probably reset the tool to "View"
    # to avoid confusion, and we definitely want to clear the custom mask.
    # We use a callback on the radio to handle these cleanups.
    def on_mode_change():
        new_mode = st.session_state["manual_plate_mode"]
        if new_mode == "Auto":
            # Clear manual shape mask file if it exists
            if PLATE_MASK_PATH.exists():
                PLATE_MASK_PATH.unlink()
            # Switch tool back to View if it was on Plate
            if st.session_state.get("active_drawing_tool") == "Polygon Plate Area":
                st.session_state["active_drawing_tool"] = "View"
        st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1

    # Mode Selector for the analysis engine
    plate_mode = st.radio(
        "Plate Detection Mode",
        ["Auto", "Multi-Plate Grid", "Manual Circle", "Manual Shape"],
        key="manual_plate_mode",
        on_change=on_mode_change,
        horizontal=False,
    )

    # Plate Radius Scale (available in Auto and Multi-Plate)
    if plate_mode in ("Auto", "Multi-Plate Grid"):
        radius_scale = compact_control(
            "Plate Radius Scale",
            "radius_scale",
            0.5,
            2.0,
            radius_scale_default,
            0.01,
            "Scale the detected plate radius. Values < 1.0 add a margin (shrink); values > 1.0 expand the area (useful for tilted cameras).",
        )
    else:
        radius_scale = 1.0

    if plate_mode == "Multi-Plate Grid":
        st.info("Specify the arrangement and labels for your plates.")
        c_g1, c_g2 = st.columns(2)

        grid_rows = c_g1.number_input(
            "Rows", min_value=1, max_value=10, value=2
        )
        grid_cols = c_g2.number_input(
            "Columns", min_value=1, max_value=10, value=3
        )

        if (
            "grid_labels" not in st.session_state
            or len(st.session_state["grid_labels"]) != grid_rows
            or len(st.session_state["grid_labels"][0]) != grid_cols
        ):
            new_labels = []
            for r in range(grid_rows):
                row_labels = []
                for c in range(grid_cols):
                    try:
                        old_val = st.session_state["grid_labels"][r][c]
                        row_labels.append(old_val)
                    except (KeyError, IndexError):
                        prefix = chr(65 + r) if r < 26 else str(r)
                        row_labels.append(f"{prefix}{c+1}")
                new_labels.append(row_labels)
            st.session_state["grid_labels"] = new_labels

        c_l1, c_l2 = st.columns(2)
        with c_l1.expander("Edit Plate Labels"):
            for r in range(grid_rows):
                cols_ui = st.columns(grid_cols)
                for c in range(grid_cols):
                    st.session_state["grid_labels"][r][c] = cols_ui[c].text_input(
                        f"Label {r},{c}",
                        value=st.session_state["grid_labels"][r][c],
                        key=f"label_input_{r}_{c}",
                        label_visibility="collapsed",
                    )

        if c_l2.button("Reset Labels", help="Reset all labels to default (A1, B1...)"):
            st.session_state.pop("grid_labels", None)
            st.rerun()

    manual_cy, manual_cx, manual_r = None, None, None
    if plate_mode == "Manual Circle":
        st.info("Tune center and radius with the controls below.")
        def_cy, def_cx, def_r = 1000, 1000, 800
        if input_files and len(input_files) == 1:
            try:
                img_peek = Image.open(input_files[0])
                w, h = img_peek.size
                def_cx, def_cy = w // 2, h // 2
                def_r = int(min(w, h) * 0.4)
            except Exception:
                pass
        manual_cy = compact_control(
            "Center Y", "manual_cy", 0, 4000, def_cy, 1, "Y coordinate of the plate center."
        )
        manual_cx = compact_control(
            "Center X", "manual_cx", 0, 4000, def_cx, 1, "X coordinate of the plate center."
        )
        manual_r = compact_control(
            "Radius", "manual_r", 0, 2000, def_r, 1, "Radius of the plate in pixels."
        )

    # Map label back to internal tool name
    active_tool = {
        "View": "View",
        "Polygon Plate Area": "Define Plate Area",
        "Exclusion Mask": "Paint Exclusion Mask",
    }[selected_tool_label]

    # Show/Hide relevant controls based on tool
    enable_mask = active_tool == "Paint Exclusion Mask"

    brush_size = st.slider(
        "Brush Size",
        1,
        100,
        20,
        disabled=(active_tool != "Paint Exclusion Mask"),
        key="mask_brush_size",
    )

    # Cleanup buttons
    c_cl1, c_cl2 = st.columns(2)
    if c_cl1.button("Clear Plate", help="Reset the manual shape plate area"):
        if PLATE_MASK_PATH.exists():
            PLATE_MASK_PATH.unlink()
        st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1
        st.rerun()
    if c_cl2.button("Clear Mask", help="Reset the exclusion mask"):
        if EXCLUSION_MASK_PATH.exists():
            EXCLUSION_MASK_PATH.unlink()
        st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1
        st.rerun()

    # Track mode changes for state cleanup (existing logic)
    if (
        "prev_plate_mode" in st.session_state
        and plate_mode != "Manual Shape"
        and st.session_state["prev_plate_mode"] == "Manual Shape"
        and PLATE_MASK_PATH.exists()
    ):
        PLATE_MASK_PATH.unlink()
        st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1
    st.session_state["prev_plate_mode"] = plate_mode

    st.divider()

    # 4. Tuning Parameters
    st.header("4. Tuning")

    # Image resize
    if "resize_enabled" not in st.session_state:
        st.session_state["resize_enabled"] = False
    if "max_dimension" not in st.session_state:
        st.session_state["max_dimension"] = max_dimension_default

    resize_enabled = st.checkbox(
        "Resize scan for detection",
        key="resize_enabled",
        help="Downscale the entire image before finding plates. Speeds up detection on large scanner files. "
        "Analysis will still use the full-resolution original unless 'Resize sub-plates' is also checked.",
    )

    # When the checkbox is first ticked, auto-populate max_dimension with the
    # actual longest side of the current image so the user has a meaningful
    # starting point rather than a hardcoded default.
    _prev_resize = st.session_state.get("_prev_resize_enabled", False)
    if resize_enabled and not _prev_resize:
        _src = None
        if input_files:
            _src = input_files[0]
        elif input_folder and Path(input_folder).is_dir():
            from blenny.modules.load_image import IMAGE_EXTENSIONS

            _candidates = sorted(
                p for p in Path(input_folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if _candidates:
                _src = _candidates[0]
        if _src is not None:
            try:
                _img = Image.open(_src)
                st.session_state["max_dimension"] = max(_img.size)
            except Exception:
                pass
    st.session_state["_prev_resize_enabled"] = resize_enabled

    max_dimension = st.number_input(
        "Max scan dimension (px)",
        min_value=100,
        max_value=10000,
        step=100,
        key="max_dimension",
        disabled=not resize_enabled,
        help="Longest side of the full image will be scaled to this many pixels for detection.",
    )

    if plate_mode == "Multi-Plate Grid":
        if "resize_sub_enabled" not in st.session_state:
            st.session_state["resize_sub_enabled"] = False
        if "max_sub_dimension" not in st.session_state:
            st.session_state["max_sub_dimension"] = 1280

        resize_sub_enabled = st.checkbox(
            "Resize sub-plates for analysis",
            key="resize_sub_enabled",
            help="Downscale each individual plate before counting colonies. Recommended for YOLO (e.g. 1280px) to save RAM and improve speed.",
        )
        max_sub_dimension = st.number_input(
            "Max sub-plate dimension (px)",
            min_value=100,
            max_value=4000,
            step=100,
            key="max_sub_dimension",
            disabled=not resize_sub_enabled,
            help="Longest side of each individual plate crop will be scaled to this many pixels for colony counting. 1280px is optimal for YOLO.",
        )
    else:
        resize_sub_enabled = False
        max_sub_dimension = None

    st.divider()

    if pipeline_mode == "Classic CV":
        if st.button("Reset All Tuning Defaults"):
            # Reset main keys and their synced widget counterparts
            for k, default in [
                ("margin", 0.04),
                ("min_area_ppm", 15),
                ("min_circ", 0.7),
                ("interior_radius", 0.85),
                ("fallback_ecc", 0.55),
            ]:
                st.session_state[k] = default
                st.session_state[f"num_{k}"] = default
                st.session_state[f"slide_{k}"] = default
            st.session_state["manual_plate_mode"] = "Auto"
            st.rerun()

        if st.session_state.get("manual_exclude_ids") and st.button(
            "Clear Manual Exclusions", help="Remove all manually excluded colonies."
        ):
            st.session_state["manual_exclude_ids"] = []
            st.rerun()

        min_area_ppm = compact_control(
            "Min Colony Size (ppm)",
            "min_area_ppm",
            0,
            1000,
            min_ppm_default,
            1,
            "Minimum area a colony must occupy, expressed in parts-per-million of the plate area. "
            "100 ppm is ~0.6mm2 on a 90mm plate.",
        )
        min_circ = compact_control(
            "Min Circularity",
            "min_circ",
            0.0,
            1.0,
            min_circ_default,
            0.05,
            "Filter objects by roundness (1.0 is a perfect circle). Low values catch rim artifacts.",
        )
        interior_radius = compact_control(
            "Interior Radius Frac",
            "interior_radius",
            0.1,
            1.0,
            interior_radius_default,
            0.05,
            "Fraction of the analysis area treated as the 'safe' interior zone for building the artifact reference profile.",
        )
        fallback_ecc = compact_control(
            "Fallback Max Eccentricity",
            "fallback_ecc",
            0.1,
            1.0,
            fallback_ecc_default,
            0.05,
            "Strict eccentricity (elongation) limit used when the plate is too sparse to build a reference profile.",
        )

        enable_multiplicity = st.checkbox(
            "Enable multiplicity estimation",
            value=True,
            key="enable_multiplicity",
            help="When enabled, detections that look like fused colonies (large area, low "
            "circularity, high solidity) are scored as multiple colonies. Uncheck to "
            "count every detection as exactly one colony.",
        )
    else:
        # Defaults for YOLO logic
        min_area_ppm = 0
        min_circ = 0.0
        interior_radius = 1.0
        fallback_ecc = 1.0
        enable_multiplicity = False

    enable_debug = st.checkbox(
        "Save debug step images",
        value=False,
        help="Write intermediate images to gui_debug. Slower.",
    )

    generate_annotated = st.checkbox(
        "Generate annotated images",
        value=True,
        help="Create and save PNG images with colony outlines. Uncheck for large batches to save time and space.",
        key="gen_ann_check",
    )

    save_subfolders = st.checkbox(
        "Save Plate Data as Subfolders",
        value=True,
        help="If checked, results for each plate are saved in a subfolder named after the image. If unchecked, all files are saved directly in the output folder with prefixes.",
    )

    enable_interactive = st.checkbox(
        "Enable Interactive Review",
        value=st.session_state.get("gen_ann_check", True),
        help="View and edit results in the GUI after analysis. Uncheck for large batches to save RAM.",
    )

    # Initialize session state for manual interventions
    if "manual_exclude_ids" not in st.session_state:
        st.session_state["manual_exclude_ids"] = []

    st.divider()

    run_btn = st.button("Run Analysis", type="primary", width="stretch")

# --- Main Area ---
input_source = None
input_paths = []
if input_files:
    input_source = "files"
    # Save the uploaded files temporarily
    temp_dir = Path("gui_uploads")
    temp_dir.mkdir(exist_ok=True)
    for f in input_files:
        p = temp_dir / f.name
        p.write_bytes(f.getvalue())
        input_paths.append(p)
elif input_folder and Path(input_folder).exists() and Path(input_folder).is_dir():
    input_source = "folder"
    folder_path = Path(input_folder)
    from blenny.modules.load_image import IMAGE_EXTENSIONS

    input_paths = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])

# --- Input Change Detection ---
# Clear manual masks if the input images change
if "last_input_paths" not in st.session_state:
    st.session_state["last_input_paths"] = []

current_input_names = [str(p) for p in input_paths]
if current_input_names != st.session_state["last_input_paths"]:
    for p in [PLATE_MASK_PATH, EXCLUSION_MASK_PATH]:
        if p.exists():
            p.unlink()
    st.session_state["last_input_paths"] = current_input_names

col1, col2 = st.columns(2)

if not input_source:
    for _key in (
        "analysis_data",
        "analysis_stem",
        "analysis_pipeline",
        "analysis_output_dir",
        "results_editor",
        "batch_results",
        "all_results",
        "result_stems",
        "current_view_idx",
        "stem_selector",
    ):
        st.session_state.pop(_key, None)

if input_source:
    if input_paths:
        # For masking, we use a 'reference image' (the first one)
        ref_image_path = input_paths[0]
        bg_image = Image.open(ref_image_path)
        bg_image = ImageOps.exif_transpose(bg_image)
        if bg_image.mode != "RGB":
            bg_image = bg_image.convert("RGB")

        # Determine canvas size
        max_ui_width = 800
        max_ui_height = 600
        scale = min(max_ui_width / bg_image.width, max_ui_height / bg_image.height, 1.0)
        canvas_width = int(bg_image.width * scale)
        canvas_height = int(bg_image.height * scale)

        # --- Visual Context for Canvas ---
        canvas_bg_img = bg_image.copy()

        if plate_mode == "Multi-Plate Grid":
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(canvas_bg_img)
            rows = grid_rows
            cols = grid_cols

            # Draw Grid Lines
            h, w = canvas_bg_img.height, canvas_bg_img.width
            cell_h, cell_w = h / rows, w / cols

            line_color = (0, 255, 255)  # Cyan
            line_w = max(2, int(4 / scale))

            for r in range(1, rows):
                y = int(r * cell_h)
                draw.line([(0, y), (w, y)], fill=line_color, width=line_w)
            for c in range(1, cols):
                x = int(c * cell_w)
                draw.line([(x, 0), (x, h)], fill=line_color, width=line_w)

            # Draw Expected or Detected Plate Circles
            font_size = int(min(cell_h, cell_w) * 0.2)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

            for r in range(rows):
                for c in range(cols):
                    # Draw Expected center
                    cy = int((r + 0.5) * cell_h)
                    cx = int((c + 0.5) * cell_w)
                    rad = int(min(cell_h, cell_w) * 0.4)

                    label = st.session_state["grid_labels"][r][c]
                    color = (0, 120, 255)  # Blue (Expected)

                    draw.ellipse(
                        [cx - rad, cy - rad, cx + rad, cy + rad], outline=color, width=line_w
                    )
                    # Draw label in center
                    draw.text((cx, cy), label, fill=(255, 255, 0), font=font, anchor="mm")

        if plate_mode == "Manual Circle" and manual_cy is not None:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(canvas_bg_img)
            r = manual_r
            # Raw Plate (Blue)
            draw.ellipse(
                [manual_cx - r, manual_cy - r, manual_cx + r, manual_cy + r],
                outline="blue",
                width=max(1, int(5 / scale)),
            )
            # Analysis Area (Green - Matches final output)
            r_eff = r * radius_scale
            draw.ellipse(
                [manual_cx - r_eff, manual_cy - r_eff, manual_cx + r_eff, manual_cy + r_eff],
                outline="green",
                width=max(1, int(3 / scale)),
            )
            # Interior Reference Zone (Yellow)
            r_int = r_eff * interior_radius
            draw.ellipse(
                [manual_cx - r_int, manual_cy - r_int, manual_cx + r_int, manual_cy + r_int],
                outline="yellow",
                width=max(1, int(2 / scale)),
            )

        # --- Integrated Drawing Area ---
        manual_shape_path = PLATE_MASK_PATH if PLATE_MASK_PATH.exists() else None
        mask_path = EXCLUSION_MASK_PATH if EXCLUSION_MASK_PATH.exists() else None

        # Tool Logic
        tool = active_tool if active_tool != "View" else None

        # Overlay existing masks as translucent context guides
        if manual_shape_path:
            mask_im = Image.open(manual_shape_path).convert("L")
            blue_img = Image.new("RGB", bg_image.size, (0, 0, 255))
            blended = Image.blend(canvas_bg_img, blue_img, 0.15)
            canvas_bg_img = Image.composite(blended, canvas_bg_img, mask_im)

        if mask_path:
            mask_im = Image.open(mask_path).convert("L")
            magenta_img = Image.new("RGB", bg_image.size, (255, 0, 255))
            blended = Image.blend(canvas_bg_img, magenta_img, 0.15)
            canvas_bg_img = Image.composite(blended, canvas_bg_img, mask_im)

        canvas_bg = canvas_bg_img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        with col1:
            if tool:
                st.subheader(f"Manual Drawing: {tool}" if tool != "View" else "Image Preview")

                if tool == "Define Plate Area":
                    st.info(
                        "**Left-click** to add vertices around the plate. **Right-click** to close and submit."
                    )
                    drawing_mode = "polygon"
                    fill_color = "rgba(0, 0, 255, 0.3)"
                    stroke_color = "#0000FF"
                elif tool == "Paint Exclusion Mask":
                    st.info("Paint over areas you want to EXCLUDE (contaminants, sharpie, etc.)")
                    drawing_mode = "freedraw"
                    fill_color = "rgba(255, 0, 255, 0.3)"
                    stroke_color = "#FF00FF"
                else:
                    # View Mode
                    drawing_mode = "transform"
                    fill_color = "rgba(0,0,0,0)"
                    stroke_color = "rgba(0,0,0,0)"

                canvas_key = f"{tool.lower().replace(' ', '_')}_canvas_v{st.session_state.get('canvas_version', 0)}"

                working_canvas = st_canvas(
                    fill_color=fill_color,
                    stroke_width=2
                    if drawing_mode == "polygon"
                    else (brush_size if tool != "View" else 0),
                    stroke_color=stroke_color,
                    background_image=canvas_bg,
                    update_streamlit=True,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode=drawing_mode,
                    display_toolbar=False,
                    key=canvas_key,
                )

                # Process results from whichever tool is active
                if working_canvas.image_data is not None:
                    alpha = working_canvas.image_data[:, :, 3]
                    if np.any(alpha > 0):
                        mask_canvas = Image.fromarray((alpha > 0).astype(np.uint8) * 255)
                        mask_im = mask_canvas.resize(bg_image.size, Image.Resampling.NEAREST)

                        if tool == "Define Plate Area":
                            manual_shape_path = PLATE_MASK_PATH
                            mask_im.save(manual_shape_path)
                        else:
                            mask_path = EXCLUSION_MASK_PATH
                            mask_im.save(mask_path)

            # Show input preview if in View mode and canvas isn't being used
            if active_tool == "View":
                st.subheader("Input Preview")
                display_img = canvas_bg_img
                st.image(display_img, width="stretch", caption=f"Reference: {ref_image_path.name}")

            if len(input_paths) > 1:
                st.write(
                    f"Batch processing {len(input_paths)} images starting with {ref_image_path.name}"
                )

    else:
        st.warning("No images found in the selected input.")
else:
    st.info("Please upload plate images or provide a folder path in the sidebar to begin.")

with col2:
    if run_btn and input_source:
        if not output_folder_input.strip():
            st.error("Please specify an Output Folder before running analysis.")
            st.stop()

        output_dir = Path(output_folder_input)
        debug_dir = output_dir / "debug" if enable_debug else None

        # Wipe previous results in the output dir if they exist
        # NOTE: We only wipe if it looks like a blenny output dir or if we want to be safe.
        # To avoid destroying user data, let's just ensure it exists.
        output_dir.mkdir(parents=True, exist_ok=True)
        if debug_dir and debug_dir.exists():
            shutil.rmtree(debug_dir)

        # Clear previous result state
        for _key in (
            "analysis_data",
            "analysis_stem",
            "analysis_pipeline",
            "analysis_output_dir",
            "results_editor",
            "batch_results",
            "batch_output_dir",
            "all_results",
            "result_stems",
            "current_view_idx",
            "stem_selector",
        ):
            st.session_state.pop(_key, None)

        with st.status("Analyzing plates...", expanded=True) as status_box:
            log_container = st.empty()

            def log_message(msg):
                if "gui_log" not in st.session_state:
                    st.session_state["gui_log"] = ""
                st.session_state["gui_log"] += msg + "\n"
                log_container.text(st.session_state["gui_log"])

            st.session_state["gui_log"] = ""
            input_imgs = input_paths

            st.session_state["all_results"] = {}
            st.session_state["result_stems"] = []
            st.session_state["batch_runs"] = []  # Store (ImageData, Pipeline) pairs
            st.session_state["current_view_idx"] = 0

            # Setup base pipeline config
            from blenny.config import extract_steps, load_yaml, substitute_paths

            try:
                if pipeline_mode == "YOLO ML":
                    # Priority: 1. User specified path (if it exists and looks like yolo)
                    #           2. pipeline_yolo.yaml in CWD
                    #           3. Built-in template
                    if Path(pipeline_path).exists() and "yolo" in pipeline_path:
                        raw_config = load_yaml(pipeline_path)
                    elif Path("pipeline_yolo.yaml").exists():
                        raw_config = load_yaml("pipeline_yolo.yaml")
                    else:
                        import yaml

                        raw_config = yaml.safe_load(templates.load_text("count-colonies-yolo"))
                else:
                    # Classic mode priority: 1. pipeline_path
                    #                        2. pipeline_classic.yaml
                    #                        3. Built-in template
                    if Path(pipeline_path).exists():
                        raw_config = load_yaml(pipeline_path)
                    elif Path("pipeline_classic.yaml").exists():
                        raw_config = load_yaml("pipeline_classic.yaml")
                    else:
                        import yaml

                        raw_config = yaml.safe_load(templates.load_text("count-colonies"))

                raw_steps = extract_steps(raw_config)
            except Exception as e:
                st.error(f"Failed to load pipeline: {e}")
                status_box.update(label="Analysis Failed", state="error")
                st.stop()

            # If YOLO, we want to replace threshold_segment with yolo_detector
            # and remove classify_by_interior
            if pipeline_mode == "YOLO ML" and not any(
                s["name"] == "yolo_detector" for s in raw_steps
            ):
                # if not (e.g. user loaded a classic config), we inject it
                new_steps = []
                for step in raw_steps:
                    if step["name"] == "threshold_segment":
                        new_steps.append(
                            {"name": "yolo_detector", "params": {"output_key": "objects"}}
                        )
                    elif step["name"] == "classify_by_interior":
                        continue
                    else:
                        new_steps.append(step)
                raw_steps = new_steps

            overrides = {
                "load_image": {"max_dimension": int(max_dimension) if resize_enabled else None},
                "detect_plate": {"radius_scale": radius_scale, "crop": False},
                "threshold_segment": {
                    "min_area": None,
                    "min_area_ppm": min_area_ppm,
                    "min_circularity": min_circ,
                },
                "yolo_detector": {"output_key": "objects"},
                "classify_by_interior": {
                    "interior_radius_frac": interior_radius,
                    "strict_fallback_max_eccentricity": fallback_ecc,
                },
                "estimate_multiplicity": {"enabled": enable_multiplicity},
            }
            if plate_mode == "Manual Circle":
                overrides["detect_plate"].update(
                    {
                        "crop": False,
                        "force_cy": manual_cy,
                        "force_cx": manual_cx,
                        "force_r": manual_r,
                    }
                )
            if plate_mode == "Manual Shape" and manual_shape_path:
                overrides["detect_plate"].update(
                    {"crop": False, "force_mask_path": str(manual_shape_path)}
                )

            # --- Handle Multi-Plate Mode Injection ---
            if plate_mode == "Multi-Plate Grid":
                # Multi-plate is a bit different: we replace detect_plate with detect_multi_plate
                # and wrap the core analysis in a sub_pipeline.

                if save_subfolders:
                    sub_path_prefix = "{output_dir}/{stem}/{stem}_{plate_label}"
                    main_path_prefix = "{output_dir}/{stem}/{stem}"
                else:
                    sub_path_prefix = "{output_dir}/{stem}_{plate_label}"
                    main_path_prefix = "{output_dir}/{stem}"

                # 1. Build the sub-pipeline steps
                # We extract the 'core' steps: detection/segmentation + measurement
                core_step_names = [
                    "apply_exclusion_mask",
                    "threshold_segment",
                    "yolo_detector",
                    "measure_colonies",
                    "classify_by_interior",
                    "estimate_multiplicity",
                ]
                sub_steps = []
                for s in raw_steps:
                    if s["name"] in core_step_names:
                        # Apply overrides to sub-steps too
                        if s["name"] in overrides:
                            s.setdefault("params", {}).update(overrides[s["name"]])
                        sub_steps.append(s)

                # Add individual exporters to sub-pipeline so they resolve {plate_label}
                if generate_annotated:
                    sub_steps.append(
                        {
                            "name": "export_annotated",
                            "params": {
                                "output_path": f"{sub_path_prefix}_annotated.png"
                            },
                        }
                    )

                # 2. Reconstruct raw_steps
                new_main_steps = [{"name": "load_image", "params": overrides["load_image"]}]
                new_main_steps.append({
                    "name": "detect_multi_plate",
                    "params": {
                        "grid": [grid_rows, grid_cols],
                        "labels": st.session_state["grid_labels"],
                        "min_confidence_score": 0.1,  # More lenient for GUI preview
                        "radius_scale": radius_scale
                    }
                })
                sub_pipeline_params = {"steps": sub_steps}
                if resize_sub_enabled:
                    sub_pipeline_params["max_subplate_dimension"] = int(max_sub_dimension)

                new_main_steps.append({
                    "name": "sub_pipeline",
                    "params": sub_pipeline_params
                })

                # Add exporters for the full multi-plate image
                if generate_annotated:
                    new_main_steps.append({
                        "name": "export_annotated",
                        "params": {"output_path": f"{main_path_prefix}_annotated.png"}
                    })
                
                new_main_steps.append({
                    "name": "export_csv",
                    "params": {"output_path": f"{main_path_prefix}_colonies.csv"}
                })
                new_main_steps.append({
                    "name": "export_summary",
                    "params": {"output_path": f"{main_path_prefix}_run_summary.txt"}
                })
                
                raw_steps = new_main_steps

            if mask_path and mask_path.exists():
                overrides["apply_exclusion_mask"] = {"mask_path": str(mask_path)}

        # Standard exporters for non-multi-plate mode
        if plate_mode != "Multi-Plate Grid":
            # Filter out any existing exporters to avoid duplicates
            raw_steps = [s for s in raw_steps if not s["name"].startswith("export_")]
            
            if save_subfolders:
                path_prefix = "{output_dir}/{stem}/{stem}"
            else:
                path_prefix = "{output_dir}/{stem}"

            if generate_annotated:
                raw_steps.append({
                    "name": "export_annotated",
                    "params": {"output_path": f"{path_prefix}_annotated.png"}
                })
            
            raw_steps.append({
                "name": "export_csv",
                "params": {"output_path": f"{path_prefix}_colonies.csv"}
            })
            raw_steps.append({
                "name": "export_summary",
                "params": {"output_path": f"{path_prefix}_run_summary.txt"}
            })

            for step in raw_steps:
                if step["name"] in overrides:
                    step.setdefault("params", {}).update(overrides[step["name"]])

            # Process every image in the batch
            for i, img_path in enumerate(input_imgs):
                log_message(f"Processing ({i + 1}/{len(input_imgs)}): {img_path.name}")

                resolved = substitute_paths(raw_steps, input_path=img_path, output_dir=output_dir)
                pipe = Pipeline.from_config(resolved)

                img_debug_dir = debug_dir / img_path.stem if debug_dir else None

                def gui_progress(current, total, name):
                    log_message(f"  [{current}/{total}] {name}...")

                data = pipe.run(img_path, output_dir=output_dir, debug_dir=img_debug_dir, progress_callback=gui_progress)
                st.session_state["batch_runs"].append((data, pipe))

                if enable_interactive:
                    # If this was a multi-plate scan, we flatten the results for the GUI
                    if "multi_plate_results" in data.metadata:
                        for sub_res in data.metadata["multi_plate_results"]:
                            plabel = sub_res.metadata.get("plate_label", "unknown")
                            display_name = f"{img_path.stem} [{plabel}]"
                            st.session_state["all_results"][display_name] = sub_res
                            st.session_state["result_stems"].append(display_name)
                        
                        # Add the main scan to the results at the END
                        display_name = f"{img_path.stem} [Full Scan]"
                        st.session_state["all_results"][display_name] = data
                        st.session_state["result_stems"].append(display_name)
                    else:
                        st.session_state["all_results"][img_path.stem] = data
                        st.session_state["result_stems"].append(img_path.stem)

            # Global completion message
            total_time = sum(
                sum(p.duration_s for p in d.provenance)
                for d, p in st.session_state["batch_runs"]
            )
            log_message(f"Batch complete: {len(input_imgs)} images in {total_time:.2f}s.")
            
            # Generate batch summary files in the temporary results dir
            generate_batch_summary([d for d, p in st.session_state["batch_runs"]], output_dir)
            generate_batch_colonies([d for d, p in st.session_state["batch_runs"]], output_dir)

            # Stash pipeline for later rendering/saving
            st.session_state["analysis_pipeline"] = pipe
            st.session_state["analysis_output_dir"] = output_dir

            status_box.update(label="Analysis Complete!", state="complete", expanded=False)

# --- Render Results (Interactive Review) ---

if st.session_state.get("all_results"):
    all_data = st.session_state["all_results"]
    stems = st.session_state["result_stems"]
    pipe = st.session_state["analysis_pipeline"]
    output_dir = st.session_state["analysis_output_dir"]

    with col2:
        st.subheader("Interactive Review")

        # --- Navigation Header ---
        c_nav1, c_nav2, c_nav3 = st.columns([1.2, 3, 1.2])

        # Initialize selection state if missing or invalid
        if "current_view_idx" not in st.session_state or st.session_state["current_view_idx"] >= len(
            stems
        ):
            st.session_state["current_view_idx"] = 0
            st.session_state["stem_selector"] = stems[0]

        curr_idx = st.session_state["current_view_idx"]

        def go_prev():
            st.session_state["current_view_idx"] -= 1
            st.session_state["stem_selector"] = stems[st.session_state["current_view_idx"]]

        def go_next():
            st.session_state["current_view_idx"] += 1
            st.session_state["stem_selector"] = stems[st.session_state["current_view_idx"]]

        def on_stem_change():
            st.session_state["current_view_idx"] = stems.index(st.session_state["stem_selector"])

        c_nav1.button(
            "← Previous",
            disabled=(curr_idx == 0),
            width="stretch",
            on_click=go_prev,
        )

        # Dropdown for direct selection
        c_nav2.selectbox(
            "Select Plate",
            options=stems,
            label_visibility="collapsed",
            key="stem_selector",
            on_change=on_stem_change,
        )

        c_nav3.button(
            "Next →",
            disabled=(curr_idx == len(stems) - 1),
            width="stretch",
            on_click=go_next,
        )

        # Get data for CURRENT selection
        stem = stems[st.session_state["current_view_idx"]]
        data = all_data[stem]

        # --- Standard Interactive Review (Table, Images, Exporters) ---
        import pandas as pd

        # Ensure ID order and counts are correct
        InteriorColonyClassifier.update_count(data.measurements, data)
        df = pd.DataFrame(data.measurements)

        # Rendering tools
        annotator = next(
            (s for s in pipe.steps if isinstance(s, AnnotatedImageExporter)),
            AnnotatedImageExporter(output_path="dummy.png", outline_color=(255, 64, 64)),
        )
        summarizer = next(
            (s for s in pipe.steps if isinstance(s, SummaryExporter)),
            SummaryExporter(output_path="dummy.txt"),
        )
        csv_exporter = next(
            (s for s in pipe.steps if isinstance(s, CSVExporter)),
            CSVExporter(output_path="dummy.csv"),
        )

        # Live render
        img = annotator.render(data)
        if img is not None:
            st.image(
                img,
                caption=f"{stem} — Reviewed Colonies: {data.metadata['colony_count']}",
                width="stretch",
            )
        else:
            # Fallback to original image if annotation fails or was skipped
            display_img = data.image if data.image is not None else data.original_image
            if display_img is not None:
                st.image(
                    display_img,
                    caption=f"{stem} (Original) — Reviewed Colonies: {data.metadata['colony_count']}",
                    width="stretch",
                )
            else:
                st.warning("No image available for preview.")

        # --- Download Section ---
        c_dl1, c_dl2, c_dl3 = st.columns(3)
        c_dl1.download_button(
            "Download CSV",
            csv_exporter.generate_csv(data),
            file_name=f"{stem}_colonies.csv",
            mime="text/csv",
            width="stretch",
        )
        c_dl2.download_button(
            "Download Summary",
            summarizer.generate_text(data),
            file_name=f"{stem}_run_summary.txt",
            mime="text/plain",
            width="stretch",
        )
        if img is not None:
            import io

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            c_dl3.download_button(
                "Download Image",
                buf.getvalue(),
                file_name=f"{stem}_annotated.png",
                mime="image/png",
                width="stretch",
            )

        st.write("Check boxes to mark artifacts. Counts update instantly.")

        display_cols = [
            "plate_label",
            "label",
            "is_artifact",
            "is_manual_review",
            "centroid_x",
            "centroid_y",
            "centroid_x_global",
            "centroid_y_global",
            "area_px",
            "area_ppm",
            "Type",
        ]

        if "is_manual_review" not in df.columns:
            df["is_manual_review"] = False
        if "is_artifact" not in df.columns:
            df["is_artifact"] = False
        if "colony_count_estimate" not in df.columns:
            df["colony_count_estimate"] = 1

        # Filter display columns based on what's actually in the data
        display_cols = [c for c in display_cols if c in df.columns]

        if "Type" not in df.columns:

            def get_type(row):
                if row.get("is_artifact"):
                    return "Artifact"
                est = int(row.get("colony_count_estimate", 1))
                if est >= 2:
                    return f"Merged(x{est})"
                return "Colony"

            df["Type"] = df.apply(get_type, axis=1)

        # Editor unique key includes stem to reset scroll/state between images
        edited_df = st.data_editor(
            df[display_cols],
            column_config={
                "is_artifact": st.column_config.CheckboxColumn("Artifact?", default=False),
                "label": st.column_config.TextColumn("ID", disabled=True),
                "is_manual_review": st.column_config.CheckboxColumn("Manual?", disabled=True),
                "Type": st.column_config.TextColumn("Class", disabled=True),
            },
            disabled=[
                "label",
                "is_manual_review",
                "centroid_x",
                "centroid_y",
                "area_px",
                "area_ppm",
                "Type",
            ],
            hide_index=True,
            width="stretch",
            key=f"editor_{stem}",
        )

        if st.session_state.get(f"editor_{stem}"):
            edits = st.session_state[f"editor_{stem}"]["edited_rows"]
            if edits:
                changed = False
                for idx, changes in edits.items():
                    if "is_artifact" in changes:
                        data.measurements[idx]["is_artifact"] = changes["is_artifact"]
                        data.measurements[idx]["is_manual_review"] = True
                        changed = True
                if changed:
                    # Update current view
                    InteriorColonyClassifier.update_count(data.measurements, data)
                    
                    # Also update main parent data if this is a sub-plate
                    if " [" in stem:
                        parent_stem = stem.split(" [")[0]
                        parent_data = next((d for d, p in st.session_state["batch_runs"] if d.metadata.get("stem") == parent_stem), None)
                        if parent_data:
                            InteriorColonyClassifier.update_count(parent_data.measurements, parent_data)
                            
                            # Re-calculate per_plate_counts
                            per_plate_counts = {}
                            for m in parent_data.measurements:
                                if not m.get("is_artifact"):
                                    pl = m.get("plate_label", "unknown")
                                    per_plate_counts[pl] = per_plate_counts.get(pl, 0) + 1
                            
                            # Preserve order/presence of plates
                            if "per_plate_counts" in parent_data.metadata:
                                for pl in parent_data.metadata["per_plate_counts"]:
                                    if pl in per_plate_counts:
                                        parent_data.metadata["per_plate_counts"][pl] = per_plate_counts[pl]
                                    else:
                                        parent_data.metadata["per_plate_counts"][pl] = 0

                    st.rerun()

        # Batch Save/Update Button
        save_dir = Path(output_folder_input).resolve()
        if st.button(f"Save/Update All results to {save_dir.name}", type="primary", width="stretch", help="Write or update all result files to the specified output folder."):
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Use the stored batch_runs (main ImageData + Pipeline)
            batch_runs = st.session_state.get("batch_runs", [])
            
            for d, p in batch_runs:
                # Update output_dir in metadata so exporters use the new destination
                old_output_dir = d.metadata.get("output_dir")
                d.metadata["output_dir"] = str(save_dir)
                
                for step in p.steps:
                    if hasattr(step, "export"):
                        step.export(d)
                
                # Restore original metadata
                if old_output_dir:
                    d.metadata["output_dir"] = old_output_dir

            # Also generate the batch summary in the destination dir
            generate_batch_summary([d for d, p in batch_runs], save_dir)
            generate_batch_colonies([d for d, p in batch_runs], save_dir)
            
            st.success(f"All {len(batch_runs)} image results (including sub-plates) saved to {save_dir}")

        with st.expander("View Live Summary"):
            st.text(summarizer.generate_text(data))

elif input_source:
    st.info("Please run the analysis to see results.")

# --- Footer ---
st.divider()
st.caption(
    "Blenny GUI v0.2 • YOLO ML Engine • Engine: "
    + subprocess.run(
        cli_cmd + ["--version"], capture_output=True, text=True
    ).stdout.strip()
)
