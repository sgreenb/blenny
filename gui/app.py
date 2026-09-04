import base64
import contextlib
import os
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
import streamlit as st
from PIL import Image, ImageFile, ImageOps

# Allow loading of slightly truncated images (common in some scanner/camera outputs)
ImageFile.LOAD_TRUNCATED_IMAGES = True

from blenny import __version__
from blenny.batch import count_plates
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
    # holds them), so drop it. Stale .pid_* markers from dead processes are
    # removed too -- the current process re-creates its own below.
    for p in GUI_TEMP_DIR.rglob("*"):
        if p.is_file():
            with contextlib.suppress(OSError):
                p.unlink()
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
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askdirectory(title=title)
            root.destroy()
            return path if path else None
        except Exception:
            pass
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

    if f"num_{key}" not in st.session_state:
        st.session_state[f"num_{key}"] = st.session_state[key]
    if f"slide_{key}" not in st.session_state:
        st.session_state[f"slide_{key}"] = st.session_state[key]

    c1.number_input(
        label,
        min_value=float(min_val),
        max_value=float(max_val),
        step=float(step),
        key=f"num_{key}",
        label_visibility="collapsed",
        on_change=sync_num,
    )
    c2.slider(
        label,
        min_value=float(min_val),
        max_value=float(max_val),
        step=float(step),
        key=f"slide_{key}",
        label_visibility="collapsed",
        on_change=sync_slide,
    )
    return st.session_state[key]


def _first_input_image_size(input_files, input_folder):
    """Return (width, height) of the first input image without decoding
    pixels (header + EXIF orientation only), or None when no image is
    available yet. Used to auto-centre the Manual Circle preview and to cap
    the centre coordinates at the image size."""
    import io as _io

    from blenny.modules.load_image import IMAGE_EXTENSIONS

    try:
        if input_files:
            src: Path | _io.BytesIO = _io.BytesIO(input_files[0].getvalue())
        elif input_folder and Path(input_folder).is_dir():
            imgs = sorted(
                f for f in Path(input_folder).iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not imgs:
                return None
            src = imgs[0]
        else:
            return None
        with Image.open(src) as im:
            w, h = im.size
            if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):
                w, h = h, w  # rotated 90/270 degrees
            return w, h
    except Exception:
        return None


# Monkey-patch for streamlit-drawable-canvas compatibility
import streamlit.elements.image as st_image

if not hasattr(st_image, "image_to_url"):
    try:
        from streamlit.elements.lib.image_utils import image_to_url as _image_to_url

        st_image.image_to_url = _image_to_url
    except ImportError:
        pass

_orig_image_to_url = st_image.image_to_url


def _compat_image_to_url(image_data, width, *args, **kwargs):
    if image_data is None:
        return None
    if isinstance(width, int):
        from dataclasses import dataclass

        @dataclass
        class FakeLayoutConfig:
            width: int

        return _orig_image_to_url(image_data, FakeLayoutConfig(width=width), *args, **kwargs)
    return _orig_image_to_url(image_data, width, *args, **kwargs)


st_image.image_to_url = _compat_image_to_url

from streamlit_drawable_canvas import st_canvas


def _total_plates(batch_data):
    """Count every plate analysed across a run's results.

    A batch is emitted whenever more than one plate is analysed (so a single
    scan image holding several plates still produces a batch), not merely when
    more than one input image was processed.
    """
    return sum(count_plates(d) for d in batch_data)


def batch_summary_text(batch_data):
    """Return the batch_summary.csv content as a string (None when there is
    nothing worth summarising into a batch, i.e. fewer than two plates
    analysed across the whole run)."""
    import csv
    import io

    if _total_plates(batch_data) <= 1:
        return None
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
        # Only add per-plate count columns when this image genuinely holds
        # several plates; a single-plate image's count is already colony_count.
        if count_plates(data) > 1 and "per_plate_counts" in m:
            for plabel, count in m["per_plate_counts"].items():
                col_name = f"plate_{plabel}_count"
                row[col_name] = count
                expected_plate_cols.add(col_name)
        rows.append(row)
    if not rows:
        return None
    fieldnames = ["input", "stem", "status", "colony_count", *sorted(expected_plate_cols)]
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    for k in sorted(list(all_keys)):
        if k not in fieldnames:
            fieldnames.append(k)
    fh = io.StringIO()
    writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return fh.getvalue()


def generate_batch_summary(batch_data, output_dir):
    text = batch_summary_text(batch_data)
    if text is None:
        return
    # Bytes avoid newline translation on Windows (the CSV text already ends
    # rows with \r\n).
    (Path(output_dir) / "batch_summary.csv").write_bytes(text.encode("utf-8"))


def batch_colonies_text(batch_data):
    """Return the batch_colonies.csv content as a string, or ``None`` when the
    run analysed fewer than two plates (a single plate has nothing to batch)."""
    if not batch_data or _total_plates(batch_data) <= 1:
        return None
    all_measurements = []
    for data in batch_data:
        all_measurements.extend(data.measurements)
    # Shared writer keeps every measurement column (the old fixed-column list
    # here silently dropped mean_h/s/v, colony_count_estimate, segment_label,
    # classification and the bbox columns while the CLI kept them).
    from blenny.batch import batch_colonies_csv_text

    return batch_colonies_csv_text(all_measurements)


def generate_batch_colonies(batch_data, output_dir):
    text = batch_colonies_text(batch_data)
    if text is None:
        return
    (Path(output_dir) / "batch_colonies.csv").write_bytes(text.encode("utf-8"))


def restamp_source(data, display_source):
    """Point an analysed ImageData at the user-facing source path.

    The pipeline runs against the scratch copy in GUI_TEMP_DIR (the loader
    reads bytes from disk), so ``data.source`` ends up pointing at the temp
    folder. That path is meaningless to the user, so re-stamp it everywhere
    exporters look: ``data.source``, ``metadata["source_path"]``, every
    measurement row's ``source`` column, and any multi-plate sub-results.
    """
    data.source = display_source
    data.metadata["source_path"] = display_source
    for row in data.measurements:
        row["source"] = display_source
    for sr in data.metadata.get("multi_plate_results", []):
        label = sr.metadata.get("plate_label", "unknown")
        sr.source = f"{display_source} [{label}]"
        sr.metadata["source_path"] = display_source
        for row in sr.measurements:
            row["source"] = sr.source


@st.dialog("Export Results", width="medium")
def export_results_dialog(
    plate_exports: list,
    batch_csv_text: str | None,
    batch_summary_text: str | None,
) -> None:
    """Modal dialog letting the user pick which files to export, rename them,
    then download them individually or write them to an output folder.

    ``plate_exports`` is a list of dicts -- one per analysed plate, each with
    ``stem``, ``csv_text``, ``log_text`` and ``annot_bytes``. Every plate's
    generated files are offered by default, so a batch / multi-plate / batch-of-
    multiplates run can export everything in one go. The aggregate batch CSV
    files are offered only when more than one input image was processed. Widget
    keys are reset by the caller before opening, so the dialog always starts
    from the default names/selection."""
    # Group per plate (in order), plus a separate batch group. When there is
    # more than one plate we render a nested list (one expander per plate) so
    # the dialog stays compact instead of listing every file flat.
    plate_items: dict[str, list] = {}
    order: list[str] = []
    for pe in plate_exports:
        s = pe["stem"]
        if s not in plate_items:
            plate_items[s] = []
            order.append(s)
        for kind, label, default_name, content, mime in [
            (
                "annot",
                "Annotated image (PNG)",
                f"{s}_annotated.png",
                pe.get("annot_bytes"),
                "image/png",
            ),
            ("csv", "Colonies CSV", f"{s}_colonies.csv", pe.get("csv_text"), "text/csv"),
            ("log", "Processing log (TXT)", f"{s}_log.txt", pe.get("log_text"), "text/plain"),
        ]:
            if content is not None:
                plate_items[s].append((kind, label, default_name, content, mime, False))
    batch_items: list[tuple] = []
    for kind, label, default_name, content, mime in [
        ("batch_csv", "Batch colonies CSV", "batch_colonies.csv", batch_csv_text, "text/csv"),
        ("batch_summary", "Batch summary CSV", "batch_summary.csv", batch_summary_text, "text/csv"),
    ]:
        if content is not None:
            batch_items.append((kind, label, default_name, content, mime, True))

    def _payload(content) -> bytes:
        return content if isinstance(content, bytes) else content.encode("utf-8")

    selected = []  # (filename, payload, mime, is_batch, stem)

    def _render_items(items: list[tuple], stem: str | None) -> None:
        for kind, label, default_name, content, mime, is_batch in items:
            key_suffix = f"batch_{kind}" if is_batch else f"{kind}_{stem}"
            c1, c2, c3 = st.columns([0.5, 2.1, 0.8], vertical_alignment="center")
            include = c1.checkbox(label, value=True, key=f"export_include_{key_suffix}")
            name = c2.text_input(
                "File name",
                value=default_name,
                key=f"export_name_{key_suffix}",
                label_visibility="collapsed",
            )
            c3.download_button(
                "Download",
                _payload(content),
                file_name=name or default_name,
                mime=mime,
                key=f"export_dl_{key_suffix}",
                disabled=not include,
                width="stretch",
            )
            if include:
                selected.append((name or default_name, _payload(content), mime, is_batch, stem))

    # Nested list when multiple plates; flat layout for a single plate.
    if len(order) > 1:
        for s in order:
            with st.expander(f"Plate {s}", expanded=False):
                _render_items(plate_items[s], s)
    else:
        for s in order:
            _render_items(plate_items[s], s)
    if batch_items:
        with st.expander("Batch files", expanded=False):
            _render_items(batch_items, None)

    st.divider()
    if not selected:
        st.info("No files available to export for this run.")
        return

    # Destination folder is only needed for the save action, so it lives here
    # in the dialog (persisted across opens via its session key).
    def _pick_output_folder() -> None:
        selected_dir = local_folder_picker("Select Output Folder")
        if selected_dir:
            st.session_state["last_export_folder"] = selected_dir

    c_f1, c_f2 = st.columns([4, 1])
    c_f1.text_input(
        "Output folder",
        key="last_export_folder",
        placeholder="e.g. C:/Users/you/Documents/exports",
        label_visibility="collapsed",
    )
    c_f2.button("Browse", on_click=_pick_output_folder, width="stretch")

    use_subfolders = st.checkbox(
        "Save each plate in its own subfolder",
        value=False,
        key="export_subfolders",
        help="Per-plate files go into output/<plate>/; batch files always stay in the output root.",
    )

    if st.button("Save to Output Folder", type="primary", width="stretch"):
        out = (st.session_state.get("last_export_folder") or "").strip()
        if not out:
            st.error("Enter an output folder path above (or use Browse).")
        else:
            save_dir = Path(out).resolve()
            save_dir.mkdir(parents=True, exist_ok=True)
            for fname, payload, _mime, is_batch, pk_stem in selected:
                dest_dir = (
                    save_dir
                    if (is_batch or not use_subfolders or not pk_stem)
                    else save_dir / pk_stem
                )
                dest_dir.mkdir(parents=True, exist_ok=True)
                (dest_dir / fname).write_bytes(payload)
            st.success(f"Saved {len(selected)} file(s) to {save_dir}")


# --- App Layout ---
st.set_page_config(
    page_title="Blenny Plate Reader", layout="wide", initial_sidebar_state="expanded"
)

logo_path = Path(__file__).parent.parent / "screenshots" / "blenny_logo.png"
if logo_path.exists():
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<div style="display: flex; align-items: center; justify-content: center; gap: 20px;">'
        f'<h1 style="margin: 0;">Blenny Plate Reader</h1>'
        f'<img src="data:image/png;base64,{logo_base64}" width="200"></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown("<h1 style='text-align: center;'>Blenny Plate Reader</h1>", unsafe_allow_html=True)

st.markdown(
    """<style>.main .block-container { min-width: 1000px; padding-top: 2rem !important; }
iframe[title="streamlit_drawable_canvas.st_canvas"] { max-width: 100%; overflow: hidden !important; border: 1px solid #eee; border-radius: 4px; }
div[data-testid="stCustomComponentV1"], div[data-testid="stIFrame"] { overflow-x: auto !important; padding: 5px; }</style>""",
    unsafe_allow_html=True,
)

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
    input_files = st.file_uploader(
        "Upload Plate Images", type=["jpg", "jpeg", "png", "tif"], accept_multiple_files=True
    )

    c_f1, c_f2 = st.columns([3, 1])
    input_folder = c_f1.text_input("OR Folder Path", value=st.session_state.get("folder_path", ""))
    c_f2.markdown("<div style='height: 29px;'></div>", unsafe_allow_html=True)
    if c_f2.button("Browse", key="browse_input", width="stretch"):
        selected = local_folder_picker("Select Input Folder")
        if selected:
            st.session_state["folder_path"] = selected
            st.rerun()

    st.divider()
    st.header("2. Analysis Mode")

    def on_mode_change():
        mode = st.session_state["manual_plate_mode"]
        if mode == "Auto":
            st.session_state["pipeline_path"] = "pipeline_yolo.yaml"
        elif mode == "Multi-Plate Grid":
            st.session_state["pipeline_path"] = "pipeline_multi.yaml"
        st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1
        # Keep the keyed text input in sync: without this, its stale session
        # value re-syncs pipeline_path back to the old file on the same rerun.
        st.session_state["pipeline_path_input"] = st.session_state["pipeline_path"]
        # The mode change resets the drawing canvas (new key), so drop any
        # per-channel clear replay left over from a previous canvas.
        st.session_state.pop("canvas_initial_drawing", None)
        # Entering Manual Polygon starts on the plate-area tool (the natural
        # first action for a hand-drawn plate); leaving it, drop the plate
        # tool so a stale selection can't force a switch back or crash later
        # (and the preview reverts to a plain image).
        if mode == "Manual Polygon":
            st.session_state["active_drawing_tool"] = "Plate Area"
        elif st.session_state.get("active_drawing_tool") == "Plate Area":
            st.session_state["active_drawing_tool"] = "View"

    plate_mode = st.radio(
        "Plate Detection Mode",
        ["Auto", "Multi-Plate Grid", "Manual Circle", "Manual Polygon"],
        key="manual_plate_mode",
        on_change=on_mode_change,
    )

    # Determine suggested pipeline based on mode
    suggested = "pipeline_multi.yaml" if plate_mode == "Multi-Plate Grid" else "pipeline_yolo.yaml"

    if "pipeline_path" not in st.session_state:
        st.session_state["pipeline_path"] = suggested

    c_p1, c_p2 = st.columns([3, 1])
    # Seed the keyed input once; on_mode_change / Browse / upload keep it in
    # sync. Don't pass value= here: Streamlit warns when a keyed widget is
    # created with a default value while its session value was set via API.
    st.session_state.setdefault("pipeline_path_input", st.session_state["pipeline_path"])
    pipeline_path_ui = c_p1.text_input("Pipeline YAML Path", key="pipeline_path_input")
    st.session_state["pipeline_path"] = pipeline_path_ui

    if c_p2.button("Browse", key="browse_pipeline", width="stretch"):
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Select Pipeline YAML", filetypes=[("YAML files", "*.yaml *.yml")]
        )
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

    def on_drawing_tool_change():
        # Selecting the plate-area tool implies a manual-polygon analysis:
        # switch the mode to match. Runs as a widget callback, i.e. BEFORE any
        # widget is instantiated, so writing manual_plate_mode is allowed here
        # -- the script body cannot (the radio already exists by the time it
        # runs, which used to raise StreamlitAPIException).
        #
        # Deliberately does NOT reset the canvas: the drawing tools share one
        # canvas (see the main area), so the plate polygon persists when you
        # switch to the exclusion tool and back -- matching main.
        tool = st.session_state.get("active_drawing_tool")
        if tool == "Plate Area" and st.session_state.get("manual_plate_mode") != "Manual Polygon":
            st.session_state["manual_plate_mode"] = "Manual Polygon"

    selected_tool_label = st.radio(
        "Select Drawing Tool",
        ["Plate Area", "Exclusion Mask", "View"],
        index=0 if plate_mode == "Manual Polygon" else 2,
        horizontal=True,
        key="active_drawing_tool",
        on_change=on_drawing_tool_change,
    )

    radius_scale = (
        compact_control(
            "Plate Radius Scale",
            "radius_scale",
            0.5,
            2.0,
            radius_scale_default,
            0.01,
            "Multiplier applied to the detected plate radius. <1 shrinks the analysis area (margin around the rim), >1 expands it — useful for tilted or off-centre plates.",
        )
        if plate_mode in ("Auto", "Multi-Plate Grid")
        else 1.0
    )

    grid_rows, grid_cols = 1, 1
    if plate_mode == "Multi-Plate Grid":
        c_g1, c_g2 = st.columns(2)
        grid_rows = c_g1.number_input("Rows", 1, 10, 2)
        grid_cols = c_g2.number_input("Columns", 1, 10, 3)
        # Preserve any user-typed labels, but always match the current grid
        # shape. (Rebuilding only on a ROW change used to IndexError when the
        # user changed Columns, and left stale labels when shrinking.)
        labels = st.session_state.get("grid_labels")
        if (
            labels is None
            or len(labels) != grid_rows
            or any(len(row) != grid_cols for row in labels)
        ):
            if labels is None:
                labels = [
                    [f"{chr(65 + r)}{c + 1}" for c in range(grid_cols)] for r in range(grid_rows)
                ]
            else:
                while len(labels) < grid_rows:
                    labels.append([f"{chr(65 + len(labels))}{c + 1}" for c in range(grid_cols)])
                while len(labels) > grid_rows:
                    labels.pop()
                for r, row in enumerate(labels):
                    if len(row) < grid_cols:
                        labels[r] = row + [
                            f"{chr(65 + r)}{c + 1}" for c in range(len(row), grid_cols)
                        ]
                    elif len(row) > grid_cols:
                        labels[r] = row[:grid_cols]
            st.session_state["grid_labels"] = labels
        with st.expander("Edit Labels"):
            for r in range(grid_rows):
                cols_ui = st.columns(grid_cols)
                for c in range(grid_cols):
                    st.session_state["grid_labels"][r][c] = cols_ui[c].text_input(
                        f"L{r}{c}",
                        st.session_state["grid_labels"][r][c],
                        key=f"l_{r}_{c}",
                        label_visibility="collapsed",
                    )

    manual_cy, manual_cx, manual_r = None, None, None
    if plate_mode == "Manual Circle":
        # Manual Circle defaults/maxima derive from the first input image:
        # the circle starts centred at ~70% of the image, and the centre
        # cannot leave the frame (x/y capped at the image size). Stale
        # values from a previous image are dropped when the size changes so
        # nothing sits outside the new bounds.
        img_size = _first_input_image_size(input_files, input_folder)
        if img_size is not None:
            img_w, img_h = img_size
            d_cx, d_cy, d_r = img_w // 2, img_h // 2, int(0.35 * min(img_w, img_h))
            cx_max, cy_max, r_max = img_w, img_h, max(img_w, img_h)
            if st.session_state.get("manual_circle_img_size") != img_size:
                for _k in ("m_cy", "m_cx", "m_r"):
                    for _sk in (_k, f"num_{_k}", f"slide_{_k}"):
                        st.session_state.pop(_sk, None)
                st.session_state["manual_circle_img_size"] = img_size
        else:
            d_cx, d_cy, d_r = 1000, 1000, 800
            cx_max, cy_max, r_max = 4000, 4000, 2000
            st.session_state["manual_circle_img_size"] = None

        manual_cy = compact_control(
            "Center Y",
            "m_cy",
            0,
            cy_max,
            d_cy,
            1,
            "Pixel Y (vertical) coordinate of the plate centre in the uploaded image, for Manual Circle mode.",
        )
        manual_cx = compact_control(
            "Center X",
            "m_cx",
            0,
            cx_max,
            d_cx,
            1,
            "Pixel X (horizontal) coordinate of the plate centre in the uploaded image, for Manual Circle mode.",
        )
        manual_r = compact_control(
            "Radius",
            "m_r",
            0,
            r_max,
            d_r,
            1,
            "Plate radius in pixels for Manual Circle mode. Pick a value just inside the plate rim.",
        )

    def _clear_drawing_channel(target: str) -> None:
        """Clear one aspect of the drawn plate without touching the other.

        ``target`` is 'plate' or 'exclusion'. The derived mask file is
        deleted and only that colour's strokes are removed from the canvas
        layer (blue = plate area, magenta = exclusion); the surviving
        strokes are replayed via ``initial_drawing`` on the fresh canvas.
        """
        mask_path = PLATE_MASK_PATH if target == "plate" else EXCLUSION_MASK_PATH
        if mask_path.exists():
            mask_path.unlink()

        # The canvas keeps every stroke in a single layer; the widget's
        # json_data lists the drawing objects, so we can filter by stroke
        # colour to drop only the cleared aspect.
        canvas_key = f"canvas_{st.session_state.get('canvas_version', 0)}"
        prev = st.session_state.get(canvas_key)
        raw = getattr(prev, "json_data", None) if prev is not None else None
        if raw is None and isinstance(prev, dict):
            raw = prev.get("raw") or prev.get("json_data")

        def _color_str(o: dict) -> str:
            return f"{o.get('stroke', '')} {o.get('fill', '')}".lower().replace(" ", "")

        survivors = None
        if isinstance(raw, dict) and raw.get("objects"):
            kept = []
            for o in raw["objects"]:
                s = _color_str(o)
                is_plate = "0000ff" in s or "0,0,255" in s
                is_excl = "ff00ff" in s or "255,0,255" in s
                if not (is_plate if target == "plate" else is_excl):
                    kept.append(o)
            # Always replay the survivors (even if nothing was dropped) so a
            # clear of an already-empty aspect can't wipe the other one.
            survivors = {**raw, "objects": kept}
        if survivors is not None:
            st.session_state["canvas_initial_drawing"] = survivors
        else:
            st.session_state.pop("canvas_initial_drawing", None)
        st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1
        st.rerun()

    brush_size = st.slider("Brush Size", 1, 100, 20, key="mask_brush_size")
    c_cl1, c_cl2 = st.columns(2)
    if c_cl1.button("Clear Plate Area", width="stretch"):
        _clear_drawing_channel("plate")
    if c_cl2.button("Clear Exclusion Mask", width="stretch"):
        _clear_drawing_channel("exclusion")

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

    resize_enabled = st.checkbox(
        "Resize scan for detection", key="resize_enabled", value=resize_default
    )
    max_dimension = st.number_input(
        "Max scan dimension (px)",
        100,
        10000,
        max_dimension_default,
        key="max_dimension",
        disabled=not resize_enabled,
    )

    resize_sub_enabled = False
    max_sub_dimension = 1280
    if plate_mode == "Multi-Plate Grid":
        resize_sub_enabled = st.checkbox("Resize sub-plates for analysis", key="resize_sub_enabled")
        max_sub_dimension = st.number_input(
            "Max sub-plate dimension (px)",
            100,
            4000,
            1280,
            key="max_sub_dimension",
            disabled=not resize_sub_enabled,
        )

    yolo_conf = compact_control(
        "YOLO Confidence",
        "yolo_conf",
        0.0,
        1.0,
        yolo_conf_default,
        0.01,
        "Minimum confidence for YOLO colony detections (0-1). Lower = more colonies "
        "detected but more false positives; higher = fewer, higher-confidence "
        "detections. Only used by pipelines with a yolo_detector step.",
    )

    if has_threshold or has_filter:
        min_area_ppm = compact_control(
            "Min Size (ppm)",
            "min_area_ppm",
            0,
            1000,
            min_area_ppm_default,
            1,
            "Smallest colony kept, in parts-per-million (ppm) of the plate area — 1 ppm = one millionth of the plate. On a 90 mm plate, 15 ppm ≈ 0.1 mm². 0 (the default) = no minimum, i.e. no area filtering. Applies during segmentation (Classic) or as a post-YOLO filter. Raise to ignore fine debris.",
        )
        min_circ = compact_control(
            "Min Circularity",
            "min_circ",
            0.0,
            1.0,
            min_circ_default,
            0.05,
            "Roundness filter (1.0 = perfect circle). Detections below this are rejected as artefacts — rim arcs and smudges score <0.5. Applies during segmentation (Classic) or as a post-YOLO filter. Set to 0 to disable.",
        )
    else:
        min_area_ppm, min_circ = min_area_ppm_default, min_circ_default

    # Interior/edge artifact rejection is circle-based (radial zones from the
    # plate centre), so it is meaningless for a drawn polygon: the whole
    # polygon is ground truth. Hide the sliders in Manual Polygon mode instead
    # of showing controls that silently do nothing.
    if has_interior and plate_mode != "Manual Polygon":
        interior_radius = compact_control(
            "Interior Radius",
            "int_r",
            0.1,
            1.0,
            interior_radius_default,
            0.05,
            "Fraction of the plate radius treated as the trusted 'interior' reference for artifact rejection. Edge-zone detections are scored against interior colonies; raise toward 1.0 to keep more edge colonies, lower to reject more.",
        )
        fallback_ecc = compact_control(
            "Max Eccentricity",
            "f_ecc",
            0.1,
            1.0,
            fallback_ecc_default,
            0.05,
            "Elongation cap for the strict fallback filter used when too few interior colonies exist to build a reference. Detections more elongated than this are marked as artefacts; 0.55 is a typical value, 1.0 disables the filter.",
        )
    elif has_interior:
        interior_radius, fallback_ecc = interior_radius_default, fallback_ecc_default
        st.caption(
            "Manual Polygon treats the whole drawn polygon as ground truth: "
            "edge-zone artifact testing (Interior Radius / Max Eccentricity) "
            "is disabled."
        )
    else:
        interior_radius, fallback_ecc = interior_radius_default, fallback_ecc_default

    generate_annotated = st.checkbox("Generate annotated images", value=True, key="gen_ann_check")
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

    input_names.extend(
        [f.name for f in Path(input_folder).iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    )

if input_names != st.session_state["last_input_paths"]:
    # Remove the temp copies we wrote for the previous input set so the
    # scratch dir doesn't hold every upload forever.
    for p in st.session_state.get("gui_written_uploads", []):
        with contextlib.suppress(OSError):
            Path(p).unlink(missing_ok=True)
    st.session_state["gui_written_uploads"] = []
    # Clear Masks
    for p in [PLATE_MASK_PATH, EXCLUSION_MASK_PATH]:
        if p.exists():
            p.unlink()
    # Clear Results
    for key in [
        "all_results",
        "result_stems",
        "batch_runs",
        "current_view_idx",
        "review_plate",
        "gui_analysis_log",
    ]:
        st.session_state.pop(key, None)
    st.session_state["last_input_paths"] = input_names

# --- Main area ----------------------------------------------------------------
if mode == "Colony Counting":
    # --- Main Logic ---
    input_paths = []
    display_sources = []
    if input_files:
        written = []
        for f in input_files:
            p = GUI_TEMP_DIR / f.name
            p.write_bytes(f.getvalue())
            input_paths.append(p)
            # Uploads live only in the scratch dir; the original filename is
            # the real user-facing source (the client-side path is never sent
            # to the server, so that's the best identifier available).
            display_sources.append(f.name)
            written.append(str(p))
        st.session_state["gui_written_uploads"] = written
    elif input_folder and Path(input_folder).is_dir():
        from blenny.modules.load_image import IMAGE_EXTENSIONS

        input_paths = sorted(
            [f for f in Path(input_folder).iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        )
        display_sources = [str(p) for p in input_paths]

    if input_paths:
        ref_img = Image.open(input_paths[0])
        ref_img = ImageOps.exif_transpose(ref_img).convert("RGB")
        scale = min(1200 / ref_img.width, 1000 / ref_img.height, 1.0)
        canvas_w, canvas_height = int(ref_img.width * scale), int(ref_img.height * scale)
        # The st_canvas iframe is clipped to its column (overflow hidden in
        # the app CSS), so an over-wide canvas used to cut off the right side
        # of the image. Cap the display size to the left column's width in
        # the 2-col layout below, preserving the aspect ratio (the mask is
        # still resized back to full resolution afterwards).
        fit = min(1.0, 560 / canvas_w) if canvas_w else 1.0
        canvas_w, canvas_height = int(canvas_w * fit), int(canvas_height * fit)
        canvas_bg_img = ref_img.copy()

        # Drawing context overlays
        if plate_mode == "Multi-Plate Grid":
            from PIL import ImageDraw

            d = ImageDraw.Draw(canvas_bg_img)
            h, w = canvas_bg_img.height, canvas_bg_img.width
            ch, cw = h / grid_rows, w / grid_cols
            for r in range(1, grid_rows):
                d.line([(0, r * ch), (w, r * ch)], fill=(0, 255, 255), width=2)
            for c in range(1, grid_cols):
                d.line([(c * cw, 0), (c * cw, h)], fill=(0, 255, 255), width=2)
            # Label each grid cell (A1, A2, ...) in its top-left corner so the
            # slot-to-label mapping is visible on the preview.
            labels = st.session_state.get("grid_labels") or []
            for r in range(grid_rows):
                for c in range(grid_cols):
                    lab = (
                        labels[r][c]
                        if r < len(labels) and c < len(labels[r])
                        else str(r * grid_cols + c + 1)
                    )
                    d.text((int(c * cw) + 8, int(r * ch) + 8), lab, fill=(255, 0, 0))
        if plate_mode == "Manual Circle" and None not in (manual_cy, manual_cx, manual_r):
            from PIL import ImageDraw

            d = ImageDraw.Draw(canvas_bg_img)
            cy, cx, r = int(manual_cy), int(manual_cx), int(manual_r)
            # Live preview of the plate circle (full-resolution coordinates,
            # drawn before the display resize) so the sidebar sliders give
            # immediate visual feedback before the analysis runs.
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 255, 0), width=3)
            d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(0, 255, 0))

        active_tool = {
            "Plate Area": "Define Plate Area",
            "Exclusion Mask": "Paint Exclusion Mask",
            "View": None,
        }[selected_tool_label]
        canvas_bg = canvas_bg_img.resize((canvas_w, canvas_height), Image.Resampling.LANCZOS)

        col1, col2 = st.columns([3, 2])
        with col1:
            if active_tool:
                mode = "polygon" if active_tool == "Define Plate Area" else "freedraw"
                # The stroke colors double as mask keys: plate polygons are
                # blue, exclusion strokes are magenta at ~50% opacity. The
                # canvas keeps every stroke (so the polygon survives tool
                # switches), and the two masks are separated by colour below --
                # otherwise the polygon strokes would leak into the exclusion
                # mask (leaving no analysis area at all).
                if active_tool == "Paint Exclusion Mask":
                    stroke_color, fill_color = (
                        "rgba(255, 0, 255, 0.5)",
                        "rgba(255, 0, 255, 0.5)",
                    )
                else:
                    stroke_color, fill_color = "#0000FF", "rgba(0, 0, 255, 0.3)"
                # Single, stable canvas (matching main): no per-tool keys, so
                # strokes persist across tool switches and never flash/reload
                # while drawing. The only exception is the per-channel clear
                # (see _clear_drawing_channel), which replays the surviving
                # strokes once onto a freshly keyed canvas.
                canvas = st_canvas(
                    fill_color=fill_color,
                    stroke_width=2 if mode == "polygon" else brush_size,
                    stroke_color=stroke_color,
                    background_image=canvas_bg,
                    height=canvas_height,
                    width=canvas_w,
                    drawing_mode=mode,
                    key=f"canvas_{st.session_state.get('canvas_version', 0)}",
                    # Replays strokes kept by a per-channel clear (see
                    # _clear_drawing_channel); stable across reruns so the
                    # canvas doesn't reload while drawing.
                    initial_drawing=st.session_state.get("canvas_initial_drawing"),
                )
                if canvas.image_data is not None and np.any(canvas.image_data[:, :, 3] > 0):
                    # image_data is the drawing layer only (transparent
                    # background), so stroke colour cleanly identifies the tool.
                    arr = canvas.image_data
                    r = arr[:, :, 0].astype(int)
                    b = arr[:, :, 2].astype(int)
                    alpha = arr[:, :, 3] > 0
                    blue = alpha & (b > 100) & (r <= 100)
                    magenta = alpha & (b > 100) & (r > 100)
                    if blue.any():
                        Image.fromarray((blue * 255).astype(np.uint8)).resize(
                            ref_img.size, Image.Resampling.NEAREST
                        ).save(PLATE_MASK_PATH)
                    if magenta.any():
                        Image.fromarray((magenta * 255).astype(np.uint8)).resize(
                            ref_img.size, Image.Resampling.NEAREST
                        ).save(EXCLUSION_MASK_PATH)
            else:
                st.image(canvas_bg_img, width="stretch")

        with col2:
            if run_btn:
                # Analysis outputs always go to the gitignored scratch dir
                # (cleared each run, and on the next server launch); results
                # stay fully interactive. The user only chooses a destination
                # folder when exporting via the export dialog.
                output_dir = GUI_TEMP_DIR / "output"
                if output_dir.exists():
                    for _f in output_dir.rglob("*"):
                        if _f.is_file():
                            with contextlib.suppress(OSError):
                                _f.unlink()
                output_dir.mkdir(parents=True, exist_ok=True)
                st.session_state.update(
                    {"all_results": {}, "result_stems": [], "batch_runs": [], "current_view_idx": 0}
                )
                # Drop cached review-table state/snapshots and annotated-image
                # caches from the previous run so they rebuild from the new
                # measurements.
                for _k in list(st.session_state):
                    if _k.startswith("ed_") or _k.startswith("annot_img_"):
                        del st.session_state[_k]

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

                    overrides = {
                        "load_image": {
                            "max_dimension": int(max_dimension) if resize_enabled else None
                        },
                        "detect_plate": {"radius_scale": radius_scale},
                        "detect_facile": {"radius_scale": radius_scale},
                        "detect_multi_plate": {
                            "radius_scale": radius_scale,
                            "grid": [grid_rows, grid_cols],
                            "labels": st.session_state.get("grid_labels"),
                        },
                        "threshold_segment": {
                            "min_area_ppm": min_area_ppm,
                            "min_circularity": min_circ,
                        },
                        "filter_colonies": {
                            "min_area_ppm": min_area_ppm,
                            "min_circularity": min_circ,
                        },
                        "classify_by_interior": {
                            "interior_radius_frac": interior_radius,
                            "strict_fallback_max_eccentricity": fallback_ecc,
                        },
                        "yolo_detector": {"conf_threshold": yolo_conf},
                        "sub_pipeline": {
                            "max_subplate_dimension": int(max_sub_dimension)
                            if resize_sub_enabled
                            else None
                        },
                    }
                    # Manual modes don't detect anything: the user's circle or
                    # polygon IS the plate, so swap the detector step out for
                    # force_plate, which builds the plate mask + ROI metadata
                    # from the drawn geometry (no circle finding). (elif: a
                    # stale plate polygon from a previous Manual Polygon
                    # session must not override the manual circle.)
                    force_plate_params = None
                    forced_label = None
                    if plate_mode == "Manual Circle":
                        force_plate_params = {
                            "radius_scale": radius_scale,
                            "force_cy": manual_cy,
                            "force_cx": manual_cx,
                            "force_r": manual_r,
                        }
                        forced_label = "manual_circle"
                    elif plate_mode == "Manual Polygon" and PLATE_MASK_PATH.exists():
                        force_plate_params = {
                            "radius_scale": radius_scale,
                            "force_mask_path": str(PLATE_MASK_PATH),
                        }
                        forced_label = "manual_polygon"
                    if EXCLUSION_MASK_PATH.exists():
                        overrides["apply_exclusion_mask"] = {"mask_path": str(EXCLUSION_MASK_PATH)}

                    def apply_overrides(steps):
                        for i, s in enumerate(steps):
                            if force_plate_params is not None and s["name"] in (
                                "detect_plate",
                                "detect_facile",
                            ):
                                steps[i] = {
                                    "name": "force_plate",
                                    "params": dict(force_plate_params),
                                    "instance_name": forced_label,
                                }
                            else:
                                if s["name"] in overrides:
                                    s.setdefault("params", {}).update(overrides[s["name"]])
                                if s["name"] == "sub_pipeline":
                                    apply_overrides(s["params"].get("steps", []))

                    apply_overrides(raw_steps)

                    t_batch_start = time.perf_counter()
                    with st.status("Analyzing...", expanded=True) as status:
                        # Clear log area for a fresh run
                        st.session_state["gui_analysis_log"] = ""
                        log_container = st.empty()

                        for i, (p, display_source) in enumerate(
                            zip(input_paths, display_sources, strict=True)
                        ):
                            status.update(
                                label=f"Analyzing {p.name} ({i + 1}/{len(input_paths)})..."
                            )
                            st.session_state["gui_analysis_log"] += f"### {p.name}\n"
                            log_container.markdown(st.session_state["gui_analysis_log"])

                            def gui_progress(current, total, name, depth=0):
                                # Append to persistent session log. Steps that
                                # run inside a sub_pipeline report depth > 0;
                                # indent them so the parent/child structure
                                # reads clearly.
                                indent = "&nbsp;&nbsp;" * depth
                                st.session_state["gui_analysis_log"] += (
                                    f"&nbsp;&nbsp;{indent}[{current}/{total}] `{name}`  \n"
                                )
                                log_container.markdown(st.session_state["gui_analysis_log"])

                            t_p_start = time.perf_counter()
                            res = substitute_paths(raw_steps, input_path=p, output_dir=output_dir)
                            data = Pipeline.from_config(res).run(
                                p,
                                output_dir=output_dir,
                                progress_callback=gui_progress,
                            )
                            # The pipeline read from the scratch copy; re-stamp
                            # the source so logs/CSVs report where the image
                            # actually came from (see restamp_source).
                            restamp_source(data, display_source)
                            t_p_elapsed = time.perf_counter() - t_p_start
                            st.session_state["gui_analysis_log"] += (
                                f"&nbsp;&nbsp;**Plate analysis complete in {t_p_elapsed:.2f}s**  \n\n"
                            )
                            log_container.markdown(st.session_state["gui_analysis_log"])

                            st.session_state["batch_runs"].append((data, raw_steps))
                            if enable_interactive:
                                if "multi_plate_results" in data.metadata:
                                    for sr in data.metadata["multi_plate_results"]:
                                        plabel = sr.metadata.get("plate_label", "unknown")
                                        # Inject plate_label into sub-result measurements for UI display
                                        for m in sr.measurements:
                                            m["plate_label"] = plabel
                                        n = f"{p.stem} [{plabel}]"
                                        st.session_state["all_results"][n] = sr
                                        st.session_state["result_stems"].append(n)
                                else:
                                    st.session_state["all_results"][p.stem] = data
                                    st.session_state["result_stems"].append(p.stem)
                        batch_data = [d for d, _ in st.session_state["batch_runs"]]
                        generate_batch_summary(batch_data, output_dir)
                        generate_batch_colonies(batch_data, output_dir)
                        # Re-sync the stable plate-selector key so the dropdown
                        # and current_view_idx start in agreement after a run.
                        if enable_interactive and st.session_state.get("result_stems"):
                            st.session_state["review_plate"] = st.session_state["result_stems"][0]
                        t_batch_elapsed = time.perf_counter() - t_batch_start
                        status.update(
                            label=f"Complete! ({t_batch_elapsed:.2f}s)",
                            state="complete",
                            expanded=True,
                        )
                except Exception as e:
                    st.error(f"Error: {e}")

            # --- Interactive Review ---
            if st.session_state.get("all_results"):
                stems = st.session_state["result_stems"]
                idx = st.session_state["current_view_idx"]
                c_nav1, c_nav2, c_nav3 = st.columns([1, 2, 1])

                def _nav_step(step: int) -> None:
                    # Runs as an on_click callback, i.e. BEFORE the script body
                    # executes, so writing the keyed widget's state here is
                    # allowed (writing it in the body would raise after the
                    # selectbox below has been instantiated).
                    new_idx = st.session_state["current_view_idx"] + step
                    st.session_state["current_view_idx"] = new_idx
                    st.session_state["review_plate"] = st.session_state["result_stems"][new_idx]

                c_nav1.button(
                    "←",
                    disabled=idx == 0,
                    width="stretch",
                    on_click=_nav_step,
                    args=(-1,),
                )
                # The plate selector needs a stable key: with an auto-generated
                # key its widget identity includes `index=idx`, so any selection
                # that changed the index churned the identity and Streamlit
                # discarded the just-made selection as a brand-new widget (the
                # first click on a plate appeared lost and needed a second one).
                # The arrows keep the keyed value in sync via the callback above.
                st.session_state["current_view_idx"] = stems.index(
                    c_nav2.selectbox(
                        "Plate",
                        stems,
                        index=0,
                        key="review_plate",
                        label_visibility="collapsed",
                    )
                )
                c_nav3.button(
                    "→",
                    disabled=idx == len(stems) - 1,
                    width="stretch",
                    on_click=_nav_step,
                    args=(1,),
                )

                stem = stems[st.session_state["current_view_idx"]]
                data = st.session_state["all_results"][stem]
                import pandas as pd

                InteriorColonyClassifier.update_count(data.measurements, data)

                # Use original pipeline exporters if possible
                annotator = AnnotatedImageExporter(output_path="d.png")
                summarizer = SummaryExporter(output_path="d.txt")
                csv_exporter = CSVExporter(output_path="d.csv")

                # View selector: mark artifacts (default), the raw colonies.csv
                # output as a table, or the log.txt output.
                view = st.selectbox(
                    "View",
                    ["Mark Artifacts", "colonies.csv", "log.txt"],
                    help="Mark Artifacts: editable table for reviewing colonies. "
                    "colonies.csv: the raw CSV output rendered as a table. "
                    "log.txt: the processing log as written by export_summary.",
                )

                def _ensure_annot_cache():
                    """Build (once) and return the cached annotated-image PNG bytes
                    for the current plate, so the download row works in every view."""
                    key = f"annot_img_{stem}"
                    if key not in st.session_state:
                        import io

                        _img = annotator.render(data)
                        if _img is not None:
                            _buf = io.BytesIO()
                            _img.save(_buf, format="PNG")
                            st.session_state[key] = _buf.getvalue()
                        else:
                            _disp = data.image if data.image is not None else data.original_image
                            if _disp is not None:
                                _buf = io.BytesIO()
                                _disp.save(_buf, format="PNG")
                                st.session_state[key] = _buf.getvalue()
                            else:
                                st.session_state[key] = None
                    return st.session_state.get(key)

                def _render_export_button() -> None:
                    """Single export entry point, shared by every view: opens the
                    export dialog (file checkboxes + editable names).

                    Every plate generated by this run is offered -- not just the
                    one currently shown -- so batch / multi-plate / batch-of-
                    multiplates runs can export all images at once."""
                    if st.button("Export Results", type="primary", width="stretch"):
                        # Start the dialog from a clean slate: reset the
                        # per-export widget keys so names/selection use the
                        # defaults again on every open.
                        for _k in list(st.session_state):
                            if _k.startswith("export_"):
                                del st.session_state[_k]

                        def _png(image):
                            import io

                            _buf = io.BytesIO()
                            image.save(_buf, format="PNG")
                            return _buf.getvalue()

                        # Build one export payload per generated plate.
                        plate_exports = []
                        for pstem in st.session_state.get("result_stems", []):
                            pdata = st.session_state["all_results"][pstem]
                            _pi = annotator.render(pdata)
                            plate_exports.append(
                                {
                                    "stem": pstem,
                                    "csv_text": CSVExporter(output_path="d.csv").generate_csv(
                                        pdata
                                    ),
                                    "log_text": SummaryExporter(output_path="d.txt").generate_text(
                                        pdata
                                    ),
                                    "annot_bytes": _png(_pi) if _pi is not None else None,
                                }
                            )

                        # Batch files are offered whenever the run analysed more
                        # than one plate (a single scan image can hold several
                        # plates), matching the CLI's default.
                        batch_data = [d for d, _ in st.session_state.get("batch_runs", [])]
                        has_batch = _total_plates(batch_data) > 1
                        export_results_dialog(
                            plate_exports=plate_exports,
                            batch_csv_text=(batch_colonies_text(batch_data) if has_batch else None),
                            batch_summary_text=(
                                batch_summary_text(batch_data) if has_batch else None
                            ),
                        )

                if view == "colonies.csv":
                    # The exact content of colonies.csv, rendered as a table.
                    import io

                    csv_text = csv_exporter.generate_csv(data)
                    if csv_text.lstrip().startswith("# (no measurements)"):
                        st.info("No measurements to display.")
                    else:
                        try:
                            csv_df = pd.read_csv(io.StringIO(csv_text))
                            # Boolean-ish columns with missing cells (e.g.
                            # touches_edge) render as empty checkboxes; show
                            # them as explicit True/False text instead
                            # (missing -> False).
                            for _col in csv_df.columns:
                                if csv_df[_col].isna().any():
                                    _vals = set(csv_df[_col].dropna().astype(str).unique())
                                    if _vals <= {"True", "False"}:
                                        csv_df[_col] = csv_df[_col].fillna(False).astype(str)
                            st.dataframe(csv_df, hide_index=True, width="stretch")
                        except Exception:
                            st.code(csv_text)
                    # Export must be available here too (no st.stop()).
                    _render_export_button()
                    st.stop()

                if view == "log.txt":
                    # The exact content of log.txt, as written by export_summary.
                    st.code(summarizer.generate_text(data), language=None)
                    _render_export_button()
                    st.stop()

                # Placeholders keep the annotated image and the download row
                # visually on top of the review section. The image slot is
                # filled immediately from a per-plate cache (see below) and
                # swapped in place after the edit handler, so the layout never
                # collapses while the editor re-renders; the downloads render
                # after the handler so they include the latest toggles.
                img_ph = st.empty()
                dl_ph = st.container()

                # Fill the image slot from the per-plate cache immediately, so
                # the slot is never left empty while the editor re-renders (an
                # empty slot collapses the layout and jerks the table). The
                # cache is refreshed further down only when artifacts were
                # toggled this rerun (or on first render).
                _annot_cache_key = f"annot_img_{stem}"
                _cached_annot = st.session_state.get(_annot_cache_key)
                if _cached_annot is not None:
                    img_ph.image(_cached_annot, width="stretch")

                def _build_editor_df() -> "pd.DataFrame":
                    """Snapshot of the review table for the selected plate.

                    Built from the current measurements and kept stable across
                    reruns so Streamlit does not remount the editor grid and
                    lose the scroll position (streamlit#10181). It is refreshed
                    only when the widget state is missing (first render,
                    view/plate switch, or a fresh run).

                    The derived "Type" column is restored (it was removed on
                    dev to avoid feeding data back into the editor, which
                    resets scroll; streamlit#10181). It is computed at snapshot
                    build time, so it refreshes when the snapshot is rebuilt
                    (view/plate switch or a fresh run).
                    """
                    df = pd.DataFrame(data.measurements)
                    for col in [
                        "plate_label",
                        "label",
                        "is_artifact",
                        "centroid_x",
                        "centroid_y",
                        "area_px",
                        "colony_count_estimate",
                        "circularity",
                        "solidity",
                    ]:
                        if col not in df.columns:
                            df[col] = (
                                0
                                if any(k in col for k in ["centroid", "area", "circ", "solid"])
                                else (
                                    False
                                    if col == "is_artifact"
                                    else 1
                                    if col == "colony_count_estimate"
                                    else "N/A"
                                )
                            )
                    # Derived per-detection classification label (Type). Mirrors
                    # the origin/main review grid: Artifact / Merged(xN) / Colony.
                    df["Type"] = df.apply(
                        lambda r: (
                            "Artifact"
                            if bool(r.get("is_artifact", False))
                            else (
                                f"Merged(x{int(r.get('colony_count_estimate', 1))})"
                                if int(r.get("colony_count_estimate", 1)) > 1
                                else "Colony"
                            )
                        ),
                        axis=1,
                    )
                    display_cols = [
                        "plate_label",
                        "label",
                        "is_artifact",
                        "Type",
                        "centroid_x",
                        "centroid_y",
                        "area_px",
                        "circularity",
                        "solidity",
                    ]
                    return df[[c for c in display_cols if c in df.columns]]

                if f"ed_{stem}" not in st.session_state:
                    st.session_state[f"ed_snapshot_{stem}"] = _build_editor_df()

                edited = st.data_editor(
                    st.session_state[f"ed_snapshot_{stem}"],
                    key=f"ed_{stem}",
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "is_artifact": st.column_config.CheckboxColumn("Artifact?"),
                        "Type": st.column_config.TextColumn("Class", disabled=True),
                        "label": st.column_config.TextColumn("ID", disabled=True),
                        "centroid_x": st.column_config.NumberColumn(
                            "X", disabled=True, format="%.0f"
                        ),
                        "centroid_y": st.column_config.NumberColumn(
                            "Y", disabled=True, format="%.0f"
                        ),
                        "area_px": st.column_config.NumberColumn("Area", disabled=True),
                        "circularity": st.column_config.NumberColumn(
                            "Circ", disabled=True, format="%.2f"
                        ),
                        "solidity": st.column_config.NumberColumn(
                            "Solid", disabled=True, format="%.2f"
                        ),
                    },
                )

                changed_any = False
                if st.session_state.get(f"ed_{stem}"):
                    edits = st.session_state[f"ed_{stem}"]["edited_rows"]
                    if edits:
                        for r_idx, changes in edits.items():
                            if "is_artifact" in changes:
                                new_val = bool(changes["is_artifact"])
                                if data.measurements[r_idx].get("is_artifact") != new_val:
                                    data.measurements[r_idx]["is_artifact"] = new_val
                                    changed_any = True
                                    # Also update main parent data if this is a sub-plate
                                    if " [" in stem:
                                        parent_stem = stem.split(" [")[0]
                                        parent_data = next(
                                            (
                                                d
                                                for d, p in st.session_state["batch_runs"]
                                                if d.metadata.get("stem") == parent_stem
                                            ),
                                            None,
                                        )
                                        if parent_data:
                                            # Find matching row in parent data. Measurements in parent
                                            # include global IDs and plate_label.
                                            plabel = data.metadata.get("plate_label")
                                            # Local label is the index in sub-plate.
                                            # SubPipeline maps measurements linearly.
                                            # However, it's safer to match by plate_label + local label.
                                            local_label = data.measurements[r_idx].get("label")
                                            for pm in parent_data.measurements:
                                                if (
                                                    pm.get("plate_label") == plabel
                                                    and pm.get("label") == local_label
                                                ):
                                                    if pm.get("is_artifact") != new_val:
                                                        pm["is_artifact"] = new_val
                                                        InteriorColonyClassifier.update_count(
                                                            parent_data.measurements, parent_data
                                                        )
                                                    break
                        if changed_any:
                            InteriorColonyClassifier.update_count(data.measurements, data)
                        # Consume the edit event in place. Deliberately no
                        # st.rerun(): re-running (or feeding the edited data
                        # back into the editor) makes Streamlit remount the
                        # grid and reset the scroll position (streamlit#10181).
                        # Any re-emitted edits are idempotent via the guards
                        # above.
                        st.session_state[f"ed_{stem}"]["edited_rows"] = {}

                # Refresh the cached annotated image only when artifacts were
                # toggled this rerun (or on first render), so a colony just
                # marked as an artifact shows up magenta immediately.
                if changed_any or _annot_cache_key not in st.session_state:
                    import io

                    _img = annotator.render(data)
                    if _img is not None:
                        _buf = io.BytesIO()
                        _img.save(_buf, format="PNG")
                        st.session_state[_annot_cache_key] = _buf.getvalue()
                    else:
                        _disp = data.image if data.image is not None else data.original_image
                        if _disp is not None:
                            _buf = io.BytesIO()
                            _disp.save(_buf, format="PNG")
                            st.session_state[_annot_cache_key] = _buf.getvalue()
                        else:
                            st.session_state[_annot_cache_key] = None

                # Swap the image in place: the early fill above already holds
                # the previous frame, so the slot keeps its size and the table
                # below never shifts while the new annotations load.
                _cached_annot = st.session_state.get(_annot_cache_key)
                if _cached_annot is not None:
                    img_ph.image(_cached_annot, width="stretch")
                else:
                    img_ph.error("No image available to display.")

                with dl_ph:
                    _render_export_button()
    else:
        st.info("Upload images to begin.")

    st.divider()
    st.caption(f"Blenny GUI v{__version__}")
