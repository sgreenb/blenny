import os
import shutil
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

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
        cmd = f'osascript -e \'POSIX path of (choose folder with prompt "{title}")\''
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
            root.attributes('-topmost', True)
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

# Find the CLI script relative to this app.py file
cli_script = str((Path(__file__).parent.parent / "blenny-cli.py").resolve())

# --- Sidebar: Configuration ---
with st.sidebar:
    st.title("Blenny Control Panel")

    # 1. Input Selection (Moved to top)
    st.header("1. Data Input")
    input_files = st.file_uploader("Upload Plate Images", type=["jpg", "jpeg", "png", "tif"], accept_multiple_files=True)

    # Input Folder Picker
    c_f1, c_f2 = st.columns([3, 1])
    input_folder = c_f1.text_input(
        "OR Folder Path",
        value=st.session_state.get("folder_path", ""),
        help="Path to a directory on your machine."
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
        help="Directory where results will be saved. Defaults to gui_results/<image_name>/ if left blank.",
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

    # Pipeline choice: Classic vs YOLO
    pipeline_mode = st.radio(
        "Analysis Engine",
        ["Classic CV", "YOLO ML"],
        index=1,
        horizontal=True,
        help="Classic CV uses edge detection and thresholding. YOLO ML uses a trained neural network."
    )

    repro_file = st.file_uploader("Load Reproducible Config", type=["yaml", "yml"], help="Load settings from a previous run.")

    # Defaults
    margin_default, min_ppm_default, min_circ_default = 0.04, 15, 0.7
    interior_radius_default = 0.85
    fallback_ecc_default = 0.55
    max_dimension_default = 2000
    resize_default = False

    if pipeline_mode == "YOLO ML":
        min_ppm_default = 5
        min_circ_default = 0.0
        max_dimension_default = 1280
        resize_default = True
        margin_default = 0.02

    if repro_file:
        try:
            import yaml
            repro_data = yaml.safe_load(repro_file)
            st.success("Config loaded. Settings applied below.")
            steps = {s['name']: s.get('params', {}) for s in repro_data.get('steps', [])}
            margin_default = float(steps.get('detect_plate', {}).get('margin_frac', 0.04))
            min_ppm_default = int(steps.get('threshold_segment', {}).get('min_area_ppm', 15))
            min_circ_default = float(steps.get('threshold_segment', {}).get('min_circularity', 0.7))
            interior_radius_default = float(steps.get('classify_by_interior', {}).get('interior_radius_frac', 0.85))
            fallback_ecc_default = float(steps.get('classify_by_interior', {}).get('strict_fallback_max_eccentricity', 0.55))
            loaded_max_dim = steps.get('load_image', {}).get('max_dimension')
            if loaded_max_dim is not None:
                max_dimension_default = int(loaded_max_dim)
                resize_default = True
        except Exception as e:
            st.error(f"Error loading config: {e}")

    if not Path("pipeline_classic.yaml").exists() and not Path("pipeline_yolo.yaml").exists() and st.button("Generate Default Pipelines"):
        subprocess.run([sys.executable, cli_script, "init"])
        st.success("Created pipeline_classic.yaml and pipeline_yolo.yaml")

    default_pipeline = "pipeline_yolo.yaml"
    if not Path(default_pipeline).exists():
        # Fallback to classic if yolo is missing
        if Path("pipeline_classic.yaml").exists():
            default_pipeline = "pipeline_classic.yaml"
        else:
            # Try finding it relative to the app root if it doesn't exist in CWD
            root_pipeline = Path(__file__).parent.parent / "pipeline_yolo.yaml"
            if root_pipeline.exists():
                default_pipeline = str(root_pipeline.resolve())

    pipeline_path = st.text_input("Pipeline Path", value=default_pipeline)

    st.divider()

    # 3. Tuning Parameters
    st.header("3. Tuning")

    # Image resize
    if "resize_enabled" not in st.session_state:
        st.session_state["resize_enabled"] = resize_default
    if "max_dimension" not in st.session_state:
        st.session_state["max_dimension"] = max_dimension_default

    resize_enabled = st.checkbox(
        "Resize images before analysis",
        key="resize_enabled",
        help="Downscale each image so its longest side is at most the value below. "
             "Preserves aspect ratio. Useful for speeding up large phone photos."
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
                p for p in Path(input_folder).iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
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
        "Max dimension (px)",
        min_value=100,
        max_value=10000,
        step=100,
        key="max_dimension",
        disabled=not resize_enabled,
        help="Longest side of the image will be scaled down to this many pixels. "
             "Non-square images are scaled proportionally — no stretching.",
        label_visibility="visible",
    )

    st.divider()

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
            on_change=sync_num
        )

        c2.slider(
            label,
            min_value=float(min_val) if isinstance(step, float) else int(min_val),
            max_value=float(max_val) if isinstance(step, float) else int(max_val),
            step=step,
            key=f"slide_{key}",
            label_visibility="collapsed",
            help=help_text,
            on_change=sync_slide
        )

        return st.session_state[key]

    if pipeline_mode == "Classic CV":
        if st.button("Reset All Tuning Defaults"):
            # Reset main keys and their synced widget counterparts
            for k, default in [("margin", 0.04), ("min_area_ppm", 15), ("min_circ", 0.7), ("interior_radius", 0.85), ("fallback_ecc", 0.55)]:
                st.session_state[k] = default
                st.session_state[f"num_{k}"] = default
                st.session_state[f"slide_{k}"] = default
            st.session_state["manual_plate_mode"] = "Auto"
            st.rerun()

        if st.session_state.get("manual_exclude_ids") and st.button("Clear Manual Exclusions", help="Remove all manually excluded colonies."):
            st.session_state["manual_exclude_ids"] = []
            st.rerun()

        margin = compact_control(
            "Plate Rim Margin", "margin", 0.0, 0.2, margin_default, 0.01,
            "The fraction of the plate radius to exclude from the edge to avoid rim reflections."
        )
        min_area_ppm = compact_control(
            "Min Colony Size (ppm)", "min_area_ppm", 0, 1000, min_ppm_default, 1,
            "Minimum area a colony must occupy, expressed in parts-per-million of the plate area. "
            "100 ppm is ~0.6mm2 on a 90mm plate."
        )
        min_circ = compact_control(
            "Min Circularity", "min_circ", 0.0, 1.0, min_circ_default, 0.05,
            "Filter objects by roundness (1.0 is a perfect circle). Low values catch rim artifacts."
        )
        interior_radius = compact_control(
            "Interior Radius Frac", "interior_radius", 0.1, 1.0, interior_radius_default, 0.05,
            "Fraction of the analysis area treated as the 'safe' interior zone for building the artifact reference profile."
        )
        fallback_ecc = compact_control(
            "Fallback Max Eccentricity", "fallback_ecc", 0.1, 1.0, fallback_ecc_default, 0.05,
            "Strict eccentricity (elongation) limit used when the plate is too sparse to build a reference profile."
        )

        enable_multiplicity = st.checkbox(
            "Enable multiplicity estimation",
            value=True,
            key="enable_multiplicity",
            help="When enabled, detections that look like fused colonies (large area, low "
                 "circularity, high solidity) are scored as multiple colonies. Uncheck to "
                 "count every detection as exactly one colony."
        )
    else:
        # Defaults for YOLO logic
        margin = 0.02
        min_area_ppm = 0
        min_circ = 0.0
        interior_radius = 1.0
        fallback_ecc = 1.0
        enable_multiplicity = False

    st.divider()

    # 4. Plate Area & Masking
    st.header("4. Plate Area & Masking")

    # Tool labels for the radio button
    # We define this BEFORE the mode radio to determine if we should force a mode.
    # Note: We use a key for the tool so we can query it safely.
    selected_tool_label = st.radio(
        "Select Drawing Tool",
        options=["🔍 View", "Polygon Plate Area", "🖌️ Mask"],
        index=0,
        horizontal=True,
        key="active_drawing_tool",
        help="View: Preview image and overlays. Plate Area: Define the circular/polygonal analysis area. Mask: Paint areas to ignore."
    )

    # Determine if we should force "Manual Shape" because the Plate tool is active.
    # We do this logic BEFORE the mode radio widget is created to avoid the crash.
    is_plate_tool = (selected_tool_label == "Polygon Plate Area")
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
            if os.path.exists("gui_plate_batch_mask.png"):
                os.remove("gui_plate_batch_mask.png")
            # Switch tool back to View if it was on Plate
            if st.session_state.get("active_drawing_tool") == "Polygon Plate Area":
                st.session_state["active_drawing_tool"] = "🔍 View"
        st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1

    # Mode Selector for the analysis engine
    plate_mode = st.radio(
        "Plate Detection Mode",
        ["Auto", "Manual Circle", "Manual Shape"],
        key="manual_plate_mode",
        on_change=on_mode_change,
        horizontal=False
    )

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
            except Exception: pass
        manual_cy = compact_control("Center Y", "manual_cy", 0, 4000, def_cy, 1, "Y coordinate of the plate center.")
        manual_cx = compact_control("Center X", "manual_cx", 0, 4000, def_cx, 1, "X coordinate of the plate center.")
        manual_r = compact_control("Radius", "manual_r", 0, 2000, def_r, 1, "Radius of the plate in pixels.")

    # Map label back to internal tool name
    active_tool = {
        "🔍 View": "View",
        "Polygon Plate Area": "Define Plate Area",
        "🖌️ Mask": "Paint Exclusion Mask"
    }[selected_tool_label]

    # Show/Hide relevant controls based on tool
    enable_mask = (active_tool == "Paint Exclusion Mask")

    brush_size = st.slider("Brush Size", 1, 100, 20, disabled=(active_tool != "Paint Exclusion Mask"))

    # Cleanup buttons
    c_cl1, c_cl2 = st.columns(2)
    if c_cl1.button("Clear Plate", help="Reset the manual shape plate area"):
        if os.path.exists("gui_plate_batch_mask.png"): os.remove("gui_plate_batch_mask.png")
        st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1
        st.rerun()
    if c_cl2.button("Clear Mask", help="Reset the exclusion mask"):
        if os.path.exists("gui_mask_batch_exclusion.png"): os.remove("gui_mask_batch_exclusion.png")
        st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1
        st.rerun()

    # Track mode changes for state cleanup (existing logic)
    if "prev_plate_mode" in st.session_state and plate_mode != "Manual Shape" and st.session_state["prev_plate_mode"] == "Manual Shape" and os.path.exists("gui_plate_batch_mask.png"):
        os.remove("gui_plate_batch_mask.png")
        st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1
    st.session_state["prev_plate_mode"] = plate_mode

    enable_debug = st.checkbox("Save debug step images", value=False, help="Write intermediate images to gui_debug. Slower.")

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
    for p in ["gui_plate_batch_mask.png", "gui_mask_batch_exclusion.png"]:
        if os.path.exists(p):
            os.remove(p)
    st.session_state["last_input_paths"] = current_input_names

col1, col2 = st.columns(2)

if not input_source:
    for _key in ("analysis_data", "analysis_stem", "analysis_pipeline",
                 "analysis_output_dir", "results_editor", "batch_results", 
                 "all_results", "result_stems", "current_view_idx"):
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
        if plate_mode == "Manual Circle" and manual_cy is not None:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(canvas_bg_img)
            r = manual_r
            # Raw Plate (Blue)
            draw.ellipse([manual_cx - r, manual_cy - r, manual_cx + r, manual_cy + r], outline="blue", width=max(1, int(5/scale)))
            # Analysis Area (Green - Matches final output)
            r_eff = r * (1.0 - margin)
            draw.ellipse([manual_cx - r_eff, manual_cy - r_eff, manual_cx + r_eff, manual_cy + r_eff], outline="green", width=max(1, int(3/scale)))
            # Interior Reference Zone (Yellow)
            r_int = r_eff * interior_radius
            draw.ellipse([manual_cx - r_int, manual_cy - r_int, manual_cx + r_int, manual_cy + r_int], outline="yellow", width=max(1, int(2/scale)))

        # --- Integrated Drawing Area ---
        manual_shape_path = Path("gui_plate_batch_mask.png") if Path("gui_plate_batch_mask.png").exists() else None
        mask_path = Path("gui_mask_batch_exclusion.png") if Path("gui_mask_batch_exclusion.png").exists() else None

        # Tool Logic
        tool = active_tool # We now always use the canvas to keep scaling consistent

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
                    st.info("**Left-click** to add vertices around the plate. **Right-click** to close and submit.")
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
                    stroke_width=2 if drawing_mode == "polygon" else (brush_size if tool != "View" else 0),
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
                            manual_shape_path = Path("gui_plate_batch_mask.png")
                            mask_im.save(manual_shape_path)
                        else:
                            mask_path = Path("gui_mask_batch_exclusion.png")
                            mask_im.save(mask_path)

            # Show input preview if in View mode and canvas isn't being used
            if active_tool == "View" and (not 'working_canvas' in locals() or not working_canvas):
                st.subheader("Input Preview")
                display_img = bg_image
                if plate_mode == "Manual Circle" and manual_cy is not None:
                    from PIL import ImageDraw
                    draw_img = display_img.copy()
                    draw = ImageDraw.Draw(draw_img)
                    r = manual_r
                    # Raw Plate (Blue)
                    draw.ellipse([manual_cx - r, manual_cy - r, manual_cx + r, manual_cy + r], outline="blue", width=5)
                    # Analysis Area (Green)
                    r_eff = r * (1.0 - margin)
                    draw.ellipse([manual_cx - r_eff, manual_cy - r_eff, manual_cx + r_eff, manual_cy + r_eff], outline="green", width=3)
                    # Interior Zone (Yellow)
                    r_int = r_eff * interior_radius
                    draw.ellipse([manual_cx - r_int, manual_cy - r_int, manual_cx + r_int, manual_cy + r_int], outline="yellow", width=2)
                    draw.line([manual_cx-20, manual_cy, manual_cx+20, manual_cy], fill="blue", width=3)
                    draw.line([manual_cx, manual_cy-20, manual_cx, manual_cy+20], fill="blue", width=3)
                    display_img = draw_img
                st.image(display_img, width="stretch", caption=f"Reference: {ref_image_path.name}")

            if len(input_paths) > 1:
                st.write(f"Batch processing {len(input_paths)} images starting with {ref_image_path.name}")

    else:
        st.warning("No images found in the selected input.")
else:
    st.info("Please upload plate images or provide a folder path in the sidebar to begin.")

with col2:
    if run_btn and input_source:
        output_dir = Path("gui_results")
        debug_dir = Path("gui_debug") if enable_debug else None

        # Wipe the temp working directories
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if debug_dir and debug_dir.exists():
            shutil.rmtree(debug_dir)

        # Clear previous result state
        for _key in ("analysis_data", "analysis_stem", "analysis_pipeline",
                     "analysis_output_dir", "results_editor", "batch_results", 
                     "batch_output_dir", "all_results", "result_stems", "current_view_idx"):
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
            st.session_state["current_view_idx"] = 0
            
            # Setup base pipeline config
            from blenny.config import extract_steps, load_yaml, substitute_paths
            
            if pipeline_mode == "YOLO ML":
                pipeline_path_to_load = "pipeline_yolo.yaml" if Path("pipeline_yolo.yaml").exists() else pipeline_path
                raw_config = load_yaml(pipeline_path_to_load)
                raw_steps = extract_steps(raw_config)
                
                # Check if yolo_detector is actually in the loaded config
                # if not (e.g. user loaded a classic config), we inject it
                if not any(s["name"] == "yolo_detector" for s in raw_steps):
                    new_steps = []
                    for step in raw_steps:
                        if step["name"] == "threshold_segment":
                            new_steps.append({"name": "yolo_detector", "params": {"output_key": "objects"}})
                        elif step["name"] == "classify_by_interior":
                            continue
                        else:
                            new_steps.append(step)
                    raw_steps = new_steps
            else:
                raw_config = load_yaml(pipeline_path)
                raw_steps = extract_steps(raw_config)

            overrides = {
                "load_image": {"max_dimension": int(max_dimension) if resize_enabled else None},
                "detect_plate": {"margin_frac": margin},
                "threshold_segment": {"min_area": None, "min_area_ppm": min_area_ppm, "min_circularity": min_circ},
                "yolo_detector": {
                    "output_key": "objects"
                },
                "classify_by_interior": {
                    "interior_radius_frac": interior_radius,
                    "strict_fallback_max_eccentricity": fallback_ecc
                },
                "estimate_multiplicity": {"enabled": enable_multiplicity},
            }
            if plate_mode == "Manual Circle":
                overrides["detect_plate"].update({
                    "crop": False,
                    "force_cy": manual_cy,
                    "force_cx": manual_cx,
                    "force_r": manual_r
                })
            if plate_mode == "Manual Shape" and manual_shape_path:
                overrides["detect_plate"].update({
                    "crop": False,
                    "force_mask_path": str(manual_shape_path)
                })
            if mask_path and mask_path.exists():
                overrides["apply_exclusion_mask"] = {"mask_path": str(mask_path)}

            for step in raw_steps:
                if step["name"] in overrides:
                    step.setdefault("params", {}).update(overrides[step["name"]])

            # Process every image in the batch
            for i, img_path in enumerate(input_imgs):
                log_message(f"Processing ({i+1}/{len(input_imgs)}): {img_path.name}")
                
                resolved = substitute_paths(raw_steps, input_path=img_path, output_dir=output_dir)
                pipe = Pipeline.from_config(resolved)
                
                img_debug_dir = debug_dir / img_path.stem if debug_dir else None
                def gui_progress(current, total, name):
                    log_message(f"  [{current}/{total}] {name}...")

                data = pipe.run(img_path, debug_dir=img_debug_dir, progress_callback=gui_progress)
                
                st.session_state["all_results"][img_path.stem] = data
                st.session_state["result_stems"].append(img_path.stem)
            
            # Global completion message
            total_time = sum(sum(p.duration_s for p in d.provenance) for d in st.session_state["all_results"].values())
            log_message(f"Batch complete: {len(input_imgs)} images in {total_time:.2f}s.")
            
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
        
        curr_idx = st.session_state.get("current_view_idx", 0)
        
        def update_idx(new_val):
            st.session_state["current_view_idx"] = new_val
            # Sync the selectbox key immediately to prevent it being sticky
            st.session_state["stem_selector"] = stems[new_val]

        if c_nav1.button("← Previous", disabled=(curr_idx == 0), use_container_width=True):
            update_idx(curr_idx - 1)
            st.rerun()
            
        # Dropdown for direct selection
        selected_stem = c_nav2.selectbox(
            "Select Plate",
            options=stems,
            index=curr_idx,
            label_visibility="collapsed",
            key="stem_selector"
        )
        # Update index if dropdown changed manually
        if selected_stem != stems[curr_idx]:
            st.session_state["current_view_idx"] = stems.index(selected_stem)
            st.rerun()
            
        if c_nav3.button("Next →", disabled=(curr_idx == len(stems) - 1), use_container_width=True):
            update_idx(curr_idx + 1)
            st.rerun()
            
        # Get data for CURRENT selection
        stem = stems[st.session_state["current_view_idx"]]
        data = all_data[stem]

        # --- Standard Interactive Review (Table, Images, Exporters) ---
        import pandas as pd
        
        # Ensure ID order and counts are correct
        InteriorColonyClassifier.update_count(data.measurements, data)
        df = pd.DataFrame(data.measurements)

        # Rendering tools
        annotator = next((s for s in pipe.steps if isinstance(s, AnnotatedImageExporter)),
                         AnnotatedImageExporter(output_path="dummy.png", outline_color=(255, 64, 64)))
        summarizer = next((s for s in pipe.steps if isinstance(s, SummaryExporter)),
                          SummaryExporter(output_path="dummy.txt"))
        csv_exporter = next((s for s in pipe.steps if isinstance(s, CSVExporter)),
                            CSVExporter(output_path="dummy.csv"))

        # Live render
        img = annotator.render(data)
        st.image(img, caption=f"{stem} — Reviewed Colonies: {data.metadata['colony_count']}", width="stretch")

        # --- Download Section ---
        c_dl1, c_dl2, c_dl3 = st.columns(3)
        c_dl1.download_button("Download CSV", csv_exporter.generate_csv(data), file_name=f"{stem}_colonies.csv", mime="text/csv", use_container_width=True)
        c_dl2.download_button("Download Log", summarizer.generate_text(data), file_name=f"{stem}_log.txt", mime="text/plain", use_container_width=True)
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        c_dl3.download_button("Download Image", buf.getvalue(), file_name=f"{stem}_annotated.png", mime="image/png", use_container_width=True)

        st.write("Check boxes to mark artifacts. Counts update instantly.")

        display_cols = ["label", "is_artifact", "is_manual_review", "centroid_x", "centroid_y", "area_px", "area_ppm", "Type"]
        if "is_manual_review" not in df.columns: df["is_manual_review"] = False
        if "Type" not in df.columns:
            def get_type(row):
                if row["is_artifact"]: return "Artifact"
                if int(row["colony_count_estimate"]) >= 2: return f"Merged(x{int(row['colony_count_estimate'])})"
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
            disabled=["label", "is_manual_review", "centroid_x", "centroid_y", "area_px", "area_ppm", "Type"],
            hide_index=True,
            width="stretch",
            key=f"editor_{stem}"
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
                    InteriorColonyClassifier.update_count(data.measurements, data)
                    st.rerun()

        # Batch Save Button
        save_dir = Path(output_folder_input).resolve() if output_folder_input.strip() else output_dir
        if st.button(f"Save All results to {save_dir.name}", type="primary", width="stretch"):
            save_dir.mkdir(parents=True, exist_ok=True)
            for s, d in all_data.items():
                cur_save_dir = save_dir / s
                cur_save_dir.mkdir(parents=True, exist_ok=True)
                for step in pipe.steps:
                    if not hasattr(step, "export"): continue
                    orig_path = getattr(step.params, "output_path", None)
                    if orig_path:
                        filename = Path(orig_path).name
                        step.params.output_path = str(cur_save_dir / filename)
                    step.export(d)
                    if orig_path: step.params.output_path = orig_path
            st.success(f"All {len(all_data)} plate results saved to {save_dir}")

        with st.expander("View Live Summary"):
            st.text(summarizer.generate_text(data))

elif input_source:
    st.info("Please run the analysis to see results.")

# --- Footer ---
st.divider()
st.caption("Blenny GUI v0.2 • YOLO ML Engine • Engine: " + subprocess.run([sys.executable, cli_script, "--version"], capture_output=True, text=True).stdout.strip())
