"""Streamlit UI for ROI mode: a sidebar section + the main-area canvas.

Kept out of ``gui/app.py`` so the ROI feature can grow (figures, R export,
folder-batch input, new canvas interactions, ...) without bloating the main
GUI. ``gui/app.py`` only calls :func:`render_roi_sidebar` and
:func:`render_roi_main` when ROI mode is selected.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from blenny.roi import analyze_rois, run_roi_analysis, write_analysis_outputs
from blenny.roi.analyze import (
    PARAM_RANGES,
    PARAMS,
    build_histogram_figure,
    count_in_range,
    pooled_counts,
    pooled_exact_stats,
    stats_from_hist,
)
from blenny.roi.canvas import PALETTE, roi_canvas
from blenny.roi.template import map_to_display, parse_template

#: Where uploaded ROI images are stashed so the analysis can read them from disk.
ROI_TEMP_DIR = Path("gui_uploads") / "roi_uploads"

_DISPLAY_MAX_W = 1000
_DISPLAY_MAX_H = 750


def _local_folder_picker(title: str) -> str | None:
    """Open a native folder picker (macOS / Windows). Mirrors the main GUI."""
    if sys.platform == "darwin":
        cmd = f"osascript -e 'POSIX path of (choose folder with prompt \"{title}\")'"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askdirectory(title=title)
            root.destroy()
            return path if path else None
        except Exception:
            pass
    return None


def render_roi_sidebar() -> None:
    """Sidebar settings for ROI mode: image upload + optional ROI template.

    Default is one image at a time; folder/batch input is a future feature.
    A JSON template (full-resolution coordinates) is parsed here and applied
    by :func:`render_roi_main` once the image is loaded.
    """
    st.markdown("#### ROI Mode")
    st.caption("Draw regions of interest; get per-region area + colour stats.")

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "tif"],
        key="roi_uploaded_image",
    )
    if uploaded is not None:
        target = ROI_TEMP_DIR / uploaded.name
        # Only re-write when the file actually changed (the widget re-delivers
        # the same file on every rerun, e.g. after each canvas interaction).
        if st.session_state.get("roi_uploaded_name") != uploaded.name or not target.exists():
            ROI_TEMP_DIR.mkdir(parents=True, exist_ok=True)
            target.write_bytes(uploaded.getvalue())
            st.session_state["roi_uploaded_name"] = uploaded.name
            st.session_state.pop("roi_img_b64_for", None)  # force re-encode
        st.session_state["roi_image_path"] = str(target.resolve())
    else:
        # The user removed the uploaded file — clear the image and everything
        # derived from it, mirroring the colony counter's input-change reset.
        for key in [
            "roi_image_path",
            "roi_uploaded_name",
            "roi_last_image",
            "roi_img_b64",
            "roi_img_b64_for",
            # Re-uploading the same image later re-applies the template.
            "roi_template_applied_for",
            "roi_template_map_error",
        ]:
            st.session_state.pop(key, None)

    # Optional ROI template: a JSON describing ROIs in *full-resolution* image
    # coordinates (e.g. a saved *_roi_geometry.json). It is parsed here and
    # mapped onto the preview by render_roi_main, exactly as if the user had
    # drawn the ROIs by hand. Parsing happens only when the file content
    # changes (the widget re-delivers the same file on every rerun).
    tmpl_file = st.file_uploader(
        "ROI Template (JSON, optional)",
        type=["json"],
        key="roi_template_file",
        help=(
            "JSON defining ROIs in full-resolution image coordinates "
            "(e.g. a saved *_roi_geometry.json). Points are mapped onto the "
            "preview automatically; any point beyond the image is rejected "
            "and nothing is drawn."
        ),
    )
    if tmpl_file is not None:
        raw = tmpl_file.getvalue()
        fp = hashlib.md5(raw).hexdigest()
        if st.session_state.get("roi_template_fp") != fp:
            # File changed — (re)parse and force a re-apply in the main area.
            st.session_state["roi_template_fp"] = fp
            st.session_state.pop("roi_template_applied_for", None)
            try:
                st.session_state["roi_template"] = parse_template(raw)
                st.session_state["roi_template_parse_error"] = None
            except ValueError as exc:
                st.session_state["roi_template"] = None
                st.session_state["roi_template_parse_error"] = str(exc)
        if st.session_state.get("roi_template_parse_error"):
            st.error(f"Invalid ROI template: {st.session_state['roi_template_parse_error']}")
    else:
        # Template removed — stop applying it, but keep any ROIs already drawn.
        for key in [
            "roi_template",
            "roi_template_fp",
            "roi_template_parse_error",
            "roi_template_applied_for",
            "roi_template_map_error",
        ]:
            st.session_state.pop(key, None)

    # NOTE: deliberately keyless, mirroring the colony sidebar's folder field.
    # A keyed widget would make ``st.session_state["roi_output_folder"] = ...``
    # (the Browse button) raise after the widget is instantiated.
    typed_output = st.text_input(
        "Output Folder",
        value=st.session_state.get("roi_output_folder", ""),
    )
    st.session_state["roi_output_folder"] = typed_output
    if st.button("Browse", key="browse_roi_output", width="stretch"):
        selected = _local_folder_picker("Select ROI Output Folder")
        if selected:
            st.session_state["roi_output_folder"] = selected
            st.rerun()


def render_roi_main() -> None:
    """Main-area ROI canvas, ROI list, run button and results."""
    img_path = st.session_state.get("roi_image_path")
    if not img_path or not Path(img_path).exists():
        st.info("Upload an image in the sidebar to begin.")
        return

    # Switching images resets the drawn ROIs and any results. ``revision`` is
    # bumped (never reset to an old value) so the canvas re-initialises; the
    # widget's stale value is deliberately NOT popped so the stale-value guard
    # below keeps ignoring it until the next real interaction.
    if st.session_state.get("roi_last_image") != img_path:
        for key in [
            "roi_rois",
            "roi_draft",
            "roi_analysis",
            "roi_analysis_rows",
            "roi_analysis_selection",
            "roi_thresholds",
            "roi_save_result",
        ]:
            st.session_state.pop(key, None)
        st.session_state["roi_last_image"] = img_path
        st.session_state["roi_next_id"] = 1
        st.session_state["roi_revision"] = st.session_state.get("roi_revision", 0) + 1

    # --- Canvas (display space) --------------------------------------------
    with Image.open(img_path) as im0:
        im0 = ImageOps.exif_transpose(im0).convert("RGB")
    full_w, full_h = im0.size
    disp = min(_DISPLAY_MAX_W / full_w, _DISPLAY_MAX_H / full_h, 1.0)
    disp_w, disp_h = max(1, int(full_w * disp)), max(1, int(full_h * disp))
    disp_img = im0.resize((disp_w, disp_h), Image.Resampling.LANCZOS) if disp < 1.0 else im0

    # --- Optional ROI template: apply once per (template, image) -----------
    # The sidebar stores the parsed template; here it is bounds-checked
    # against the full-resolution image, mapped into display space (the exact
    # inverse of scale_points) and loaded into the canvas exactly as if the
    # user had drawn the ROIs by hand. The marker prevents re-applying on
    # every rerun (which would clobber subsequent edits); it is invalidated
    # when the file content or the image changes.
    tmpl = st.session_state.get("roi_template")
    if tmpl:
        applied_for = st.session_state.get("roi_template_applied_for")
        if applied_for != (st.session_state.get("roi_template_fp"), str(img_path)):
            st.session_state["roi_template_applied_for"] = (
                st.session_state.get("roi_template_fp"),
                str(img_path),
            )
            try:
                mapped = map_to_display(tmpl, full_w, full_h, disp_w, disp_h)
            except ValueError as exc:
                st.session_state["roi_template_map_error"] = str(exc)
            else:
                if not mapped:
                    st.session_state["roi_template_map_error"] = (
                        "the template contains no ROIs"
                    )
                else:
                    st.session_state["roi_template_map_error"] = None
                    st.session_state["roi_rois"] = [
                        {**r, "color": r.get("color") or PALETTE[i % len(PALETTE)]}
                        for i, r in enumerate(mapped)
                    ]
                    st.session_state["roi_draft"] = []
                    st.session_state["roi_next_id"] = (
                        max((r["id"] for r in mapped), default=0) + 1
                    )
                    # Any previous analysis is stale — mirror the image-switch reset.
                    for key in [
                        "roi_analysis",
                        "roi_analysis_rows",
                        "roi_analysis_selection",
                        "roi_thresholds",
                        "roi_save_result",
                    ]:
                        st.session_state.pop(key, None)
                    # Python-side change: bump revision so the canvas re-inits.
                    st.session_state["roi_revision"] = (
                        st.session_state.get("roi_revision", 0) + 1
                    )
    if st.session_state.get("roi_template_map_error"):
        st.error(
            "ROI template not applied — "
            f"{st.session_state['roi_template_map_error']}. "
            "No ROIs were changed."
        )

    # Encode the display image once per image; the base64 is unchanged across
    # reruns, so caching it keeps every canvas interaction fast (the canvas
    # already skips reloading it when the data URL is identical).
    if st.session_state.get("roi_img_b64_for") != str(img_path):
        buf = io.BytesIO()
        disp_img.save(buf, format="JPEG", quality=85)
        st.session_state["roi_img_b64"] = base64.b64encode(buf.getvalue()).decode()
        st.session_state["roi_img_b64_for"] = str(img_path)
    image_b64 = st.session_state["roi_img_b64"]

    rois = st.session_state.get("roi_rois", [])
    draft = st.session_state.get("roi_draft", [])
    next_id = st.session_state.get("roi_next_id", 1)
    revision = st.session_state.get("roi_revision", 0)

    result = roi_canvas(
        disp_img,
        rois,
        draft,
        next_id=next_id,
        revision=revision,
        image_b64=image_b64,
        height=disp_h + 34,
    )
    state = getattr(result, "state", None)
    last_applied = st.session_state.get("roi_last_applied")
    if state and state != last_applied:
        # Fresh canvas interaction — apply it (coordinates are display-space).
        # The widget re-delivers its last value on unrelated reruns; comparing
        # with what we already applied makes those no-ops.
        st.session_state["roi_last_applied"] = state
        st.session_state["roi_rois"] = state.get("rois", [])
        st.session_state["roi_draft"] = state.get("draft", [])
        st.session_state["roi_next_id"] = int(state.get("nextId", next_id))
    rois = st.session_state.get("roi_rois", [])

    # --- ROI list: rename / delete -----------------------------------------
    if rois:
        st.markdown("##### Regions of Interest")
        names_changed = False
        for roi in rois:
            c1, c2, c3 = st.columns([4, 1, 1])
            new_name = c1.text_input(
                "Name",
                value=roi.get("name", "ROI"),
                key=f"roi_name_{roi.get('id')}",
                label_visibility="collapsed",
            )
            if new_name != roi.get("name"):
                roi["name"] = new_name
                names_changed = True
            c2.markdown(
                f"<div style='display:flex;align-items:center;gap:6px;height:34px;'>"
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"background:{roi.get('color', '#888')};border-radius:2px;'></span>"
                f"{len(roi.get('points', []))} pts</div>",
                unsafe_allow_html=True,
            )
            if c3.button("Delete", key=f"roi_delete_{roi.get('id')}", width="stretch"):
                st.session_state["roi_rois"] = [r for r in rois if r.get("id") != roi.get("id")]
                # Python-side change: bump revision so the canvas re-inits.
                st.session_state["roi_revision"] = st.session_state.get("roi_revision", 0) + 1
                st.rerun()
        # Name edits mutate the shared list; persist them explicitly.
        st.session_state["roi_rois"] = rois
        if names_changed:
            # Python-side change: bump revision + rerun so the canvas labels
            # update immediately.
            st.session_state["roi_revision"] = st.session_state.get("roi_revision", 0) + 1
            st.rerun()

    # --- Analyze / Save -----------------------------------------------------
    c_an, c_sv = st.columns(2)
    analyze_btn = c_an.button("Analyze", type="primary", width="stretch")
    save_btn = c_sv.button("Save Results", width="stretch")
    scale = (full_w / disp_w, full_h / disp_h)
    stem = Path(img_path).stem

    if analyze_btn:
        if not rois:
            st.warning("Draw at least one ROI before analyzing.")
        else:
            with st.spinner("Analyzing..."):
                rows_a, hist_data = analyze_rois(str(img_path), rois, scale=scale)
            st.session_state["roi_analysis"] = {
                **hist_data,
                "revision": st.session_state.get("roi_revision", 0),
                "image": str(img_path),
                "fingerprint": _roi_fingerprint(rois),
            }
            st.session_state["roi_analysis_rows"] = rows_a

    if save_btn:
        roi_output = st.session_state.get("roi_output_folder", "")
        if not rois:
            st.warning("Draw at least one ROI before saving.")
        elif not roi_output:
            st.error("Specify an output folder.")
        else:
            out_dir = Path(roi_output)
            out_dir.mkdir(parents=True, exist_ok=True)
            with st.spinner("Saving..."):
                _rows, paths = run_roi_analysis(
                    str(img_path), rois, out_dir, scale=scale, stem=stem
                )
            analysis = st.session_state.get("roi_analysis")
            artifact_paths: dict[str, Path] = {}
            if analysis and analysis.get("fingerprint") == _roi_fingerprint(rois):
                sel = st.session_state.get("roi_analysis_selection")
                ids = sel["ids"] if sel else [r.get("id") for r in rois if r.get("id") is not None]
                names = sel["names"] if sel else [r.get("name", "ROI") for r in rois]
                colors = sel.get("colors") if sel else [r.get("color") for r in rois]
                thr = st.session_state.get("roi_thresholds", {})
                artifact_paths = write_analysis_outputs(
                    out_dir,
                    stem,
                    analysis,
                    ids,
                    names,
                    thr,
                    roi_colors=colors,
                    normalize=bool(st.session_state.get("roi_normalize", True)),
                )
            elif analysis:
                st.warning(
                    "Analysis artifacts were skipped because the ROIs changed after "
                    "Analyze — re-run Analyze to refresh the dashboard, then save again."
                )
            st.session_state["roi_save_result"] = {
                **{k: str(v) for k, v in paths.items()},
                **{k: str(v) for k, v in artifact_paths.items()},
                "dir": str(out_dir),
            }
            st.success(f"Saved ROI results to {out_dir}")

    # --- Dashboard (in-memory analysis; the core data is never touched) -----
    if st.session_state.get("roi_analysis"):
        _render_dashboard(
            img_path,
            rois,
            st.session_state["roi_analysis"],
            st.session_state.get("roi_analysis_rows", []),
        )

    # --- Saved files --------------------------------------------------------
    if st.session_state.get("roi_save_result"):
        res = st.session_state["roi_save_result"]
        with st.expander("Saved files"):
            for kind, path in res.items():
                if kind != "dir":
                    st.write(f"**{kind}:** `{path}`")


def _roi_fingerprint(rois: list[dict]) -> str:
    """Lightweight fingerprint of the ROI set (ids, names, vertices).

    Stored at Analyze time; any later add/delete/move/rename changes it, which
    makes the dashboard (and its saved artifacts) stale until re-analysis.
    """
    return json.dumps(
        [
            [
                r.get("id"),
                r.get("name"),
                [[round(x, 1), round(y, 1)] for x, y in r.get("points", [])],
            ]
            for r in rois
        ],
        sort_keys=True,
    )


def _render_dashboard(img_path: Path, rois: list[dict], analysis: dict, rows: list[dict]) -> None:
    """Interactive histogram dashboard over the analyzed pixel data.

    Pure analysis layer: nothing here modifies the ROIs, the image, or the
    core measurements — the user can clip ranges and export at will.
    """
    if analysis.get("image") != str(img_path):
        return
    if analysis.get("fingerprint") != _roi_fingerprint(rois):
        st.warning(
            "The ROIs changed after you last ran Analyze. The dashboard below "
            "shows the previous analysis — run Analyze again to refresh it "
            "before drawing conclusions or saving analysis artifacts."
        )
        st.info("Press **Analyze** to re-analyze the current ROIs.")
        return

    st.markdown("#### Dashboard")
    param = st.selectbox("Parameter", PARAMS, key="roi_param")

    c_norm = st.columns([1, 4])[0]
    normalize = c_norm.checkbox(
        "Normalize",
        value=True,
        key="roi_normalize",
        help="Scale each ROI's histogram to sum to 1 (fraction of pixels) so "
        "distributions are comparable even when ROIs differ in area. "
        "Uncheck to show raw pixel counts.",
    )

    st.markdown("Include ROIs:")
    ncols = min(max(len(rois), 1), 6)
    cols = st.columns(ncols)
    for i, roi in enumerate(rois):
        with cols[i % ncols]:
            st.checkbox(
                roi.get("name", "ROI"),
                value=True,
                key=f"roi_inc_{roi.get('id')}",
            )
    selected = [r for r in rois if st.session_state.get(f"roi_inc_{r.get('id')}", True)]
    if not selected:
        st.info("Check at least one ROI to see the histogram.")
        return
    sel_ids = [r.get("id") for r in selected]
    sel_names = [r.get("name", "ROI") for r in selected]
    sel_colors = [r.get("color") for r in selected]
    st.session_state["roi_analysis_selection"] = {
        "ids": sel_ids,
        "names": sel_names,
        "colors": sel_colors,
    }

    hists = analysis["hists"]
    exact = analysis["exact"]
    # Only ROIs present in the analysis get drawn (should be all of them).
    pairs = [
        (rid, name, color)
        for rid, name, color in zip(sel_ids, sel_names, sel_colors, strict=True)
        if rid in hists
    ]
    if not pairs:
        st.info("The selected ROIs have no analyzed pixel data.")
        return

    per_roi_hists = [hists[rid][param] for rid, _, _ in pairs]
    per_roi_stats = [exact[rid][param] for rid, _, _ in pairs]
    per_roi_names = [name for _, name, _ in pairs]
    per_roi_colors = [color for _, _, color in pairs]

    combined = pooled_exact_stats(exact, [rid for rid, _, _ in pairs], param)
    n_total, mean, std = combined
    if n_total == 0:
        st.info("The selected ROIs contain no pixels.")
        return

    counts = pooled_counts(hists, [rid for rid, _, _ in pairs], param)

    lo0, hi0 = PARAM_RANGES[param]
    lo, hi = st.slider(
        "Threshold range",
        lo0,
        hi0,
        st.session_state.get(f"roi_thr_{param}", (lo0, hi0)),
        key=f"roi_thr_{param}",
    )
    st.session_state.setdefault("roi_thresholds", {})[param] = (float(lo), float(hi))

    clipped_n, clipped_mean, clipped_std = stats_from_hist(counts, param, lo, hi)
    fig = build_histogram_figure(
        param,
        per_roi_hists,
        per_roi_stats,
        per_roi_names,
        per_roi_colors,
        lo,
        hi,
        combined,
        clipped_n,
        clipped_mean,
        clipped_std,
        normalize=normalize,
    )
    st.pyplot(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Pixels (full)", f"{n_total:,}")
    c2.metric("Mean ± Std (full)", f"{mean:.3f} ± {std:.3f}")
    c3.metric("Pixels kept", f"{clipped_n:,} ({100.0 * clipped_n / n_total:.1f}%)")

    # Per-ROI table reflecting the current thresholds.
    area_by_name = {r.get("name"): r.get("area_px") for r in rows}
    table_rows = []
    for roi in selected:
        rid = roi.get("id")
        if rid not in exact:
            continue
        n_full, roi_mean, roi_std = exact[rid][param]
        n_kept = count_in_range(hists[rid][param], param, lo, hi)
        table_rows.append(
            {
                "ROI": roi.get("name"),
                "area_px": area_by_name.get(roi.get("name")),
                "mean": round(roi_mean, 4),
                "std": round(roi_std, 4),
                "pixels (full)": n_full,
                "pixels kept": n_kept,
                "% kept": round(100.0 * n_kept / n_full, 1) if n_full else 0.0,
            }
        )
    if table_rows:
        import pandas as pd  # type: ignore[import-untyped]

        st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")
