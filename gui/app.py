import tkinter as tk
from tkinter import filedialog
import streamlit as st
import subprocess
import shutil
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

from blenny.pipeline import Pipeline
from blenny.modules.export_annotated import AnnotatedImageExporter
from blenny.modules.export_summary import SummaryExporter
from blenny.modules.export_csv import CSVExporter
from blenny.modules.classify_interior import InteriorColonyClassifier

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
import streamlit.elements.image as st_image
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

from streamlit_drawable_canvas import st_canvas

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
        overflow: auto !important;
    }
    /* The Streamlit-generated wrapper around components */
    div[data-testid="stCustomComponentV1"],
    div[data-testid="stIFrame"] {
        overflow-x: auto !important;
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
    repro_file = st.file_uploader("Load Reproducible Config", type=["yaml", "yml"], help="Load settings from a previous run.")
    
    # Defaults
    margin_default, min_area_default, min_circ_default = 0.08, 10, 0.7
    interior_radius_default = 0.85
    max_dimension_default = 2000
    resize_default = False

    if repro_file:
        try:
            import yaml
            repro_data = yaml.safe_load(repro_file)
            st.success("Config loaded. Settings applied below.")
            steps = {s['name']: s.get('params', {}) for s in repro_data.get('steps', [])}
            margin_default = float(steps.get('detect_plate', {}).get('margin_frac', 0.08))
            min_area_default = int(steps.get('threshold_segment', {}).get('min_area', 10))
            min_circ_default = float(steps.get('threshold_segment', {}).get('min_circularity', 0.7))
            interior_radius_default = float(steps.get('classify_by_interior', {}).get('interior_radius_frac', 0.85))
            loaded_max_dim = steps.get('load_image', {}).get('max_dimension')
            if loaded_max_dim is not None:
                max_dimension_default = int(loaded_max_dim)
                resize_default = True
        except Exception as e:
            st.error(f"Error loading config: {e}")

    if not Path("pipeline.yaml").exists():
        if st.button("Generate Default Pipeline"):
            subprocess.run(["python3", cli_script, "init"])
            st.success("Created pipeline.yaml")

    default_pipeline = "pipeline.yaml"
    if not Path(default_pipeline).exists():
        # Try finding it relative to the app root if it doesn't exist in CWD
        root_pipeline = Path(__file__).parent.parent / "pipeline.yaml"
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
        
        # Label Row
        st.markdown(f"**{label}**")
        
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

        val = c1.number_input(
            label,
            min_value=float(min_val) if isinstance(step, float) else int(min_val),
            max_value=float(max_val) if isinstance(step, float) else int(max_val),
            step=step,
            key=f"num_{key}",
            label_visibility="collapsed",
            on_change=sync_num
        )

        val = c2.slider(
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

    if st.button("Reset All Tuning Defaults"):
        # Reset main keys and their synced widget counterparts
        for k, default in [("margin", 0.08), ("min_area", 10), ("min_circ", 0.7), ("interior_radius", 0.85)]:
            st.session_state[k] = default
            st.session_state[f"num_{k}"] = default
            st.session_state[f"slide_{k}"] = default
        st.session_state["manual_plate_mode"] = "Auto"
        st.rerun()
    
    if st.session_state.get("manual_exclude_ids"):
        if st.button("Clear Manual Exclusions", help="Remove all manually excluded colonies."):
            st.session_state["manual_exclude_ids"] = []
            st.rerun()

    margin = compact_control(
        "Plate Rim Margin", "margin", 0.0, 0.2, margin_default, 0.01, 
        "The fraction of the plate radius to exclude from the edge to avoid rim reflections."
    )
    min_area = compact_control(
        "Min Colony Area (px)", "min_area", 0, 1000, min_area_default, 1, 
        "Minimum number of pixels a group must occupy to be counted as a colony."
    )
    min_circ = compact_control(
        "Min Circularity", "min_circ", 0.0, 1.0, min_circ_default, 0.05, 
        "Filter objects by roundness (1.0 is a perfect circle)."
    )
    interior_radius = compact_control(
        "Interior Radius Frac", "interior_radius", 0.1, 1.0, interior_radius_default, 0.05,
        "Fraction of the plate radius treated as the 'safe' interior zone for artifact rejection."
    )

    enable_multiplicity = st.checkbox(
        "Enable multiplicity estimation",
        value=True,
        key="enable_multiplicity",
        help="When enabled, detections that look like fused colonies (large area, low "
             "circularity, high solidity) are scored as multiple colonies. Uncheck to "
             "count every detection as exactly one colony."
    )

    st.divider()

    # 4. Plate Area
    st.header("4. Plate Area")
    plate_mode = st.radio("Detection Mode", ["Auto", "Manual Circle", "Manual Shape"], index=0, key="manual_plate_mode", horizontal=True)
    
    manual_cy, manual_cx, manual_r = None, None, None
    manual_shape_path = None
    
    if plate_mode == "Manual Circle":
        st.info("Tune center and radius with the controls below.")
        
        # Determine smart defaults if an image is loaded
        def_cy, def_cx, def_r = 1000, 1000, 800
        if input_files and len(input_files) == 1:
            try:
                # We need to peek at the image size
                from PIL import Image
                img_peek = Image.open(input_files[0])
                w, h = img_peek.size
                def_cx, def_cy = w // 2, h // 2
                def_r = int(min(w, h) * 0.4)
            except:
                pass

        manual_cy = compact_control("Center Y", "manual_cy", 0, 4000, def_cy, 1, "Y coordinate of the plate center.")
        manual_cx = compact_control("Center X", "manual_cx", 0, 4000, def_cx, 1, "X coordinate of the plate center.")
        manual_r = compact_control("Radius", "manual_r", 0, 2000, def_r, 1, "Radius of the plate in pixels.")
    
    elif plate_mode == "Manual Shape":
        st.info("Use the canvas on the right to define the plate boundary. **Left-click** to add points and **Right-click** to close the shape.")

    st.divider()

    # 5. Masking
    st.header("5. Masking")
    enable_mask = st.checkbox("Enable Paint-to-Exclude", value=False)
    enable_debug = st.checkbox("Save debug step images", value=False,
                               help="Write intermediate images for every pipeline step to gui_debug/. Slower.")
    brush_size = st.slider("Brush Size", 1, 50, 20)
    
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

col1, col2 = st.columns(2)

if not input_source:
    for _key in ("analysis_data", "analysis_stem", "analysis_pipeline",
                 "analysis_output_dir", "results_editor", "batch_results"):
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
        # If we are in Manual Circle mode, we want to see the blue circle 
        # while we are painting the exclusion mask.
        canvas_bg_img = bg_image.copy()
        if plate_mode == "Manual Circle" and manual_cy is not None:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(canvas_bg_img)
            # Draw at original scale, then we resize for canvas
            r = manual_r
            draw.ellipse([manual_cx - r, manual_cy - r, manual_cx + r, manual_cy + r], outline="blue", width=max(1, int(5/scale)))
            r_eff = r * (1.0 - margin)
            draw.ellipse([manual_cx - r_eff, manual_cy - r_eff, manual_cx + r_eff, manual_cy + r_eff], outline="cyan", width=max(1, int(3/scale)))
        
        canvas_bg = canvas_bg_img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        if len(input_paths) > 1:
            st.info(f"Batch Mode: Drawing on **{ref_image_path.name}** as reference. Mask will apply to all {len(input_paths)} images.")

        # --- Integrated Drawing Area ---
        manual_shape_path = None
        mask_path = None
        
        # If both Manual Shape and Masking are enabled, we need a toggle to switch tool
        canvas_mode = "Inclusion"
        if plate_mode == "Manual Shape" and enable_mask:
            canvas_mode = st.radio("Canvas Tool", ["1. Define Plate Shape", "2. Paint Exclusion Mask"], horizontal=True)
        elif enable_mask:
            canvas_mode = "Exclusion"
        
        if plate_mode == "Manual Shape" or enable_mask:
            st.subheader(f"Manual Drawing: {canvas_mode}")
            
            if "Inclusion" in canvas_mode or "Define Plate Shape" in canvas_mode:
                st.info("**Left-click** to add vertices around the plate. **Right-click** to close and submit.")
                drawing_mode = "polygon"
                fill_color = "rgba(0, 0, 255, 0.3)"
                stroke_color = "#0000FF"
                canvas_key = "shape_canvas"
            else:
                st.info("Paint over areas you want to EXCLUDE (contaminants, sharpie, etc.)")
                drawing_mode = "freedraw"
                fill_color = "rgba(255, 0, 255, 0.3)"
                stroke_color = "#FF00FF"
                canvas_key = "exclusion_canvas"

            working_canvas = st_canvas(
                fill_color=fill_color,
                stroke_width=2 if drawing_mode == "polygon" else brush_size,
                stroke_color=stroke_color,
                background_image=canvas_bg,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode=drawing_mode,
                display_toolbar=True,
                key=canvas_key,
            )
            
            # Process results from whichever canvas is active
            if working_canvas.image_data is not None:
                alpha = working_canvas.image_data[:, :, 3]
                if np.any(alpha > 0):
                    mask_canvas = Image.fromarray((alpha > 0).astype(np.uint8) * 255)
                    mask_im = mask_canvas.resize(bg_image.size, Image.Resampling.NEAREST)
                    
                    if "Inclusion" in canvas_mode or "Shape" in canvas_mode:
                        manual_shape_path = Path("gui_plate_batch_mask.png")
                        mask_im.save(manual_shape_path)
                    else:
                        mask_path = Path("gui_mask_batch_exclusion.png")
                        mask_im.save(mask_path)

        with col1:
            # We only show the "Input Preview" if no drawing canvas is visible
            # to avoid cluttering the screen.
            if not (plate_mode == "Manual Shape" or enable_mask):
                st.subheader("Input Preview")
                display_img = bg_image
                if plate_mode == "Manual Circle" and manual_cy is not None:
                    from PIL import ImageDraw
                    draw_img = display_img.copy()
                    draw = ImageDraw.Draw(draw_img)
                    r = manual_r
                    draw.ellipse([manual_cx - r, manual_cy - r, manual_cx + r, manual_cy + r], outline="blue", width=5)
                    r_eff = r * (1.0 - margin)
                    draw.ellipse([manual_cx - r_eff, manual_cy - r_eff, manual_cx + r_eff, manual_cy + r_eff], outline="cyan", width=3)
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

if run_btn and input_source:
    output_dir = Path("gui_results")
    debug_dir = Path("gui_debug") if enable_debug else None

    # Wipe the temp working directories so stale results from previous runs
    # never bleed into the current one.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir and debug_dir.exists():
        shutil.rmtree(debug_dir)

    # Clear previous result state when starting a new run
    for _key in ("analysis_data", "analysis_stem", "analysis_pipeline",
                 "analysis_output_dir", "results_editor", "batch_results", "batch_output_dir"):
        st.session_state.pop(_key, None)

    with st.status("Analyzing plates...", expanded=True) as status:
        st.write("Executing Blenny engine...")
        
        # Determine input for Pipeline
        if input_source == "files":
            input_imgs = input_paths
        else:
            input_imgs = input_paths

        # We'll use the Python API for single images to allow interactive review.
        # For batch, we still use the CLI for now as it's more robust for large sets.
        if len(input_imgs) == 1:
            img_path = input_imgs[0]
            
            # 1. Build Pipeline manually so we can inject params
            # We mimic the logic from CLI main.py
            from blenny.config import extract_steps, load_yaml, substitute_paths
            raw_config = load_yaml(pipeline_path)
            raw_steps = extract_steps(raw_config)
            
            # Apply GUI overrides
            overrides = {
                "load_image": {"max_dimension": int(max_dimension) if resize_enabled else None},
                "detect_plate": {"margin_frac": margin},
                "threshold_segment": {"min_area": min_area, "min_circularity": min_circ},
                "classify_by_interior": {"interior_radius_frac": interior_radius},
                "estimate_multiplicity": {"enabled": enable_multiplicity},
            }
            if plate_mode == "Manual Circle":
                overrides["detect_plate"].update({
                    "crop": False,
                    "radius_expand_frac": 0.0,
                    "force_cy": manual_cy,
                    "force_cx": manual_cx,
                    "force_r": manual_r
                })
            if plate_mode == "Manual Shape" and manual_shape_path:
                overrides["detect_plate"].update({
                    "crop": False,
                    "force_mask_path": str(manual_shape_path)
                })
            if enable_mask and 'mask_path' in locals() and mask_path:
                overrides["apply_exclusion_mask"] = {"mask_path": str(mask_path)}
            
            for step in raw_steps:
                if step["name"] in overrides:
                    step.setdefault("params", {}).update(overrides[step["name"]])
            
            resolved = substitute_paths(raw_steps, input_path=img_path, output_dir=output_dir)
            pipe = Pipeline.from_config(resolved)
            
            # 2. Run
            img_debug_dir = debug_dir / img_path.stem if debug_dir else None
            data = pipe.run(img_path, debug_dir=img_debug_dir)
            
            # 3. Store in session state
            st.session_state["analysis_data"] = data
            st.session_state["analysis_stem"] = img_path.stem
            st.session_state["analysis_output_dir"] = output_dir
            st.session_state["analysis_pipeline"] = pipe
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)
        else:
            # BATCH MODE (existing CLI path)
            cli_input = str(input_folder) if input_source == "folder" else str(temp_dir)
            cmd = [
                "python3", cli_script, "run", 
                pipeline_path, 
                "--input", cli_input,
                "--output", str(output_dir),
                "--json",
                "--override", f"detect_plate.margin_frac={margin}",
                "--override", f"threshold_segment.min_area={min_area}",
                "--override", f"threshold_segment.min_circularity={min_circ}",
                "--override", f"classify_by_interior.interior_radius_frac={interior_radius}",
                *([] if enable_multiplicity else ["--no-multiplicity"]),
                *((["--override", f"load_image.max_dimension={int(max_dimension)}"] if resize_enabled else [])),
            ]
            if plate_mode == "Manual Circle":
                cmd.extend([
                    "--override", f"detect_plate.force_cy={manual_cy}",
                    "--override", f"detect_plate.force_cx={manual_cx}",
                    "--override", f"detect_plate.force_r={manual_r}"
                ])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                st.session_state["batch_results"] = result.stdout
                st.session_state["batch_output_dir"] = str(output_dir.resolve())
                status.update(label="Batch Analysis Complete!", state="complete", expanded=False)
            else:
                st.error(result.stderr)

# --- Render Results (Live or Batch) ---

if "analysis_data" in st.session_state:
    data = st.session_state["analysis_data"]
    stem = st.session_state["analysis_stem"]
    output_dir = st.session_state["analysis_output_dir"]
    pipe = st.session_state["analysis_pipeline"]
    
    with col2:
        st.subheader("Interactive Review")
        
        # 1. Get current measurements
        import pandas as pd
        df = pd.DataFrame(data.measurements)
        
        # Ensure ID order and counts are correct based on current status
        # Note: We do NOT reassign IDs here during the review loop, as the user
        # wants IDs to remain stable once the analysis has run.
        InteriorColonyClassifier.update_count(data.measurements, data)
        df = pd.DataFrame(data.measurements)

        # 2. Setup rendering tools (reusing pipeline exporters)
        # We find them in the pipeline or create defaults with dummy paths
        annotator = next((s for s in pipe.steps if isinstance(s, AnnotatedImageExporter)), 
                         AnnotatedImageExporter(output_path="dummy.png", outline_color=(255, 64, 64)))
        summarizer = next((s for s in pipe.steps if isinstance(s, SummaryExporter)), 
                          SummaryExporter(output_path="dummy.txt"))
        csv_exporter = next((s for s in pipe.steps if isinstance(s, CSVExporter)),
                            CSVExporter(output_path="dummy.csv"))

        # 3. Live render the image and summary
        img = annotator.render(data)
        st.image(img, caption=f"Reviewed Colonies: {data.metadata['colony_count']}", width="stretch")

        # --- Download Section ---
        c_dl1, c_dl2, c_dl3 = st.columns(3)
        c_dl1.download_button(
            "Download CSV",
            csv_exporter.generate_csv(data),
            file_name=f"{stem}_colonies.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Log download
        c_dl2.download_button(
            "Download Log",
            summarizer.generate_text(data),
            file_name=f"{stem}_log.txt",
            mime="text/plain",
            use_container_width=True
        )

        # Annotated image download
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        c_dl3.download_button(
            "Download Image",
            buf.getvalue(),
            file_name=f"{stem}_annotated.png",
            mime="image/png",
            use_container_width=True
        )
        # --- End Download Section ---

        # 4. Handle data editor changes
        # We want to toggle 'is_artifact'
        st.write("Check boxes below to mark objects as artifacts. Counts and images will update instantly.")
        
        # We define which columns to show and make 'is_artifact' editable
        display_cols = ["label", "is_artifact", "is_manual_review", "centroid_x", "centroid_y", "area_px", "Type"]
        if "is_manual_review" not in df.columns:
            df["is_manual_review"] = False
        if "Type" not in df.columns:
            def get_type(row):
                if row["is_artifact"]: return "Artifact"
                if int(row["colony_count_estimate"]) >= 2: return f"Merged(x{int(row['colony_count_estimate'])})"
                return "Colony"
            df["Type"] = df.apply(get_type, axis=1)

        edited_df = st.data_editor(
            df[display_cols],
            column_config={
                "is_artifact": st.column_config.CheckboxColumn("Artifact?", default=False),
                "label": st.column_config.TextColumn("ID", disabled=True),
                "is_manual_review": st.column_config.CheckboxColumn("Manual?", disabled=True),
                "Type": st.column_config.TextColumn("Class", disabled=True),
            },
            disabled=["label", "is_manual_review", "centroid_x", "centroid_y", "area_px", "Type"],
            hide_index=True,
            width="stretch",
            key="results_editor"
        )
        
        # Apply changes from editor back to the ImageData
        if st.session_state.get("results_editor"):
            edits = st.session_state["results_editor"]["edited_rows"]
            if edits:
                changed = False
                for idx, changes in edits.items():
                    if "is_artifact" in changes:
                        # Map index back to the measurements list
                        data.measurements[idx]["is_artifact"] = changes["is_artifact"]
                        data.measurements[idx]["is_manual_review"] = True
                        changed = True
                if changed:
                    # Update counts but DO NOT reassign IDs during review
                    InteriorColonyClassifier.update_count(data.measurements, data)
                    st.rerun()

        save_dir = Path(output_folder_input).resolve() if output_folder_input.strip() else output_dir / stem
        if st.button("Save Results", type="primary", width="stretch"):
            save_dir.mkdir(parents=True, exist_ok=True)
            # Re-point every exporter to the chosen directory, preserving filenames.
            for step in pipe.steps:
                if not hasattr(step, "export"):
                    continue
                orig_path = getattr(step.params, "output_path", None)
                if orig_path is not None:
                    filename = Path(orig_path).name
                    step.params.output_path = str(save_dir / filename)
                step.export(data)
                if orig_path is not None:
                    step.params.output_path = orig_path
            st.success(f"Results saved to {save_dir}")

        # Summary Log
        with st.expander("View Live Summary"):
            st.text(summarizer.generate_text(data))

elif "batch_results" in st.session_state:
    st.subheader("Batch Results")
    st.info("Interactive review is available for single-image runs. Batch results are saved to disk.")

    # Show the JSON summary
    with st.expander("📊 View Batch Summary", expanded=True):
        st.code(st.session_state["batch_results"], language="json")

    # Save / copy results
    default_batch_out = st.session_state.get("batch_output_dir", "gui_results")
    save_dest = Path(output_folder_input).resolve() if output_folder_input.strip() else None

    if save_dest and str(save_dest) != default_batch_out:
        if st.button("Copy Results to Output Folder", type="primary", width="stretch"):
            src = Path(default_batch_out)
            save_dest.mkdir(parents=True, exist_ok=True)
            for item in src.iterdir():
                dest_item = save_dest / item.name
                if item.is_dir():
                    shutil.copytree(item, dest_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest_item)
            st.success(f"Results copied to {save_dest}")
    else:
        st.success(f"Results already saved to **{default_batch_out}**")
        st.caption("To save to a different location, set the Output Folder in the sidebar before running, "
                   "or set it now and click the button that will appear above.")

elif input_source:
    st.info("Please run the analysis to see results.")

# --- Footer ---
st.divider()
st.caption("Blenny GUI Skeleton v0.1 • Local Engine: " + subprocess.run(["python3", cli_script, "--version"], capture_output=True, text=True).stdout.strip())
