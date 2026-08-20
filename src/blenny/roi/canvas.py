"""Interactive ROI drawing canvas — a Streamlit v2 bidirectional component.

``streamlit-drawable-canvas`` cannot do what ROI mode needs (multiple
simultaneous polygons in different colours, right-click-to-close, vertex
drag/delete), and ``st.components.v1.html`` is deprecated in current
Streamlit and cannot return values anyway. So this module defines a small
hand-written component with ``st.components.v2.component`` — plain inline
HTML/CSS/JS, no build step — that sends its full state back to Python via
``setStateValue``.

Interaction model
-----------------
* **Left-click** adds a vertex to the current draft polygon.
* **Right-click** closes the draft into a committed ROI (auto-named
  ``ROI N``, next palette colour); right-clicking *on a vertex* of a
  committed ROI deletes that vertex instead.
* **Drag** a vertex (committed or draft) to move it.
* Click a vertex and press **Delete/Backspace** to remove it; **Escape**
  clears the in-progress draft.

State flow (why it behaves the way it does)
-------------------------------------------
Streamlit re-invokes the component's JS module on *every* rerun (its mount
effect depends on ``data``), and for an *unkeyed* component a change in
``data`` also changes the component identity → the iframe is torn down and
recreated, which is the "flash" and — because Python can only apply the
delivered widget value *after* the canvas has rendered — made every
interaction lag one click behind.

This module therefore:
* mounts the component **keyed** (``key="roi_canvas"``) so ``data`` changes
  never remount the iframe;
* keeps all geometry in **element-scoped JS state** that survives
  re-invocations, attaching event listeners exactly once;
* re-initialises from Python's ``data`` **only when ``revision`` changes**
  (bumped by Python for image switches / renames / deletes), so user
  interactions are never wiped by a re-render;
* reloads the background image only when it actually changed.

With that, a left-click shows its point immediately and the polygon closes
on the first right-click — no flash, no double-click.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import streamlit as st
from PIL import Image

#: Distinct, colourblind-considerate palette; cycled for each new ROI.
PALETTE: list[str] = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#800000",
    "#aaffc3",
    "#fffac8",
]

MAX_DISPLAY_DIM = 1200  # longest side (px) of the embedded canvas image

_HTML = """
<div id="roi-wrap" style="position:relative;">
  <canvas id="roi-canvas" style="border:1px solid #ccc;border-radius:4px;
          background:#fafafa;cursor:crosshair;display:block;max-width:100%;"></canvas>
  <div style="font-size:12px;color:#666;margin-top:4px;">
    Left-click: add point &nbsp;·&nbsp; Right-click: close polygon (or delete a vertex) &nbsp;·&nbsp;
    Drag: move vertex &nbsp;·&nbsp; Select + Delete: remove point &nbsp;·&nbsp; Escape: clear draft
  </div>
</div>
"""

_CSS = """
#roi-canvas { touch-action: none; max-width: 100%; }
"""

_JS_TEMPLATE = r"""
export default function(component) {
  const { data, setStateValue, parentElement } = component;
  const wrap = parentElement.querySelector('#roi-wrap');
  const canvas = wrap.querySelector('#roi-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = data.width, H = data.height;
  // Only resize the bitmap when it actually changed — resetting canvas.width
  // on every re-invocation clears the canvas and adds paint churn.
  if (canvas.width !== W) canvas.width = W;
  if (canvas.height !== H) canvas.height = H;

  const PALETTE = __PALETTE__;

  // Element-scoped state survives re-invocations (Streamlit re-runs this
  // module whenever `data` changes on any rerun).
  if (!canvas.__roiState) {
    canvas.__roiState = {
      rois: [], draft: [], nextId: 1, revision: -1, img: null,
      selected: null, dragging: null, attached: false,
    };
  }
  const s = canvas.__roiState;

  // NOTE: the canvas is deliberately left to pure CSS sizing (`max-width:
  // 100%`). We used to pin `style.height` here to keep the aspect ratio
  // correct while CSS-scaled, but that created a positive feedback loop:
  // `getBoundingClientRect().width` is border-box (includes the 1px
  // borders) while `style.height` is content-box, so each ResizeObserver
  // cycle computed height = round(height + 2*H/W) and the canvas grew ~2px
  // per frame until it overflowed the layout. Browsers maintain the
  // intrinsic aspect ratio for a replaced element constrained only by
  // `max-width`, and `canvasPos()` below maps clicks via `W / rect.width` /
  // `H / rect.height`, so coordinates stay correct at any display size.
  function send() {
    setStateValue('state', { rois: s.rois, draft: s.draft, nextId: s.nextId });
  }

  function dist(ax, ay, bx, by) { return Math.hypot(ax - bx, ay - by); }

  function findVertex(x, y) {
    for (let i = 0; i < s.rois.length; i++) {
      const pts = s.rois[i].points;
      for (let j = 0; j < pts.length; j++) {
        if (dist(x, y, pts[j][0], pts[j][1]) <= 8) return {roi: i, vi: j};
      }
    }
    for (let j = 0; j < s.draft.length; j++) {
      if (dist(x, y, s.draft[j][0], s.draft[j][1]) <= 8) return {draft: true, vi: j};
    }
    return null;
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    if (s.img && s.img.complete && s.img.naturalWidth > 0) ctx.drawImage(s.img, 0, 0);

    s.rois.forEach(roi => {
      if (roi.points.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(roi.points[0][0], roi.points[0][1]);
      for (let i = 1; i < roi.points.length; i++) ctx.lineTo(roi.points[i][0], roi.points[i][1]);
      ctx.closePath();
      ctx.fillStyle = roi.color + '33';
      ctx.strokeStyle = roi.color;
      ctx.lineWidth = 2.5;
      ctx.fill();
      ctx.stroke();

      roi.points.forEach((p, vi) => {
        ctx.beginPath();
        ctx.arc(p[0], p[1], 4, 0, 2 * Math.PI);
        ctx.fillStyle = s.selected && s.selected.roi === s.rois.indexOf(roi) && s.selected.vi === vi ? '#ff0' : '#fff';
        ctx.fill();
        ctx.strokeStyle = roi.color;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });

      const cx = roi.points.reduce((a, p) => a + p[0], 0) / roi.points.length;
      const cy = roi.points.reduce((a, p) => a + p[1], 0) / roi.points.length;
      ctx.font = 'bold 14px sans-serif';
      const tw = ctx.measureText(roi.name).width;
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.fillRect(cx - tw / 2 - 4, cy - 20, tw + 8, 18);
      ctx.textAlign = 'center';
      ctx.fillStyle = roi.color;
      ctx.fillText(roi.name, cx, cy - 6);
    });

    if (s.draft.length) {
      ctx.beginPath();
      ctx.moveTo(s.draft[0][0], s.draft[0][1]);
      for (let i = 1; i < s.draft.length; i++) ctx.lineTo(s.draft[i][0], s.draft[i][1]);
      ctx.strokeStyle = '#222';
      ctx.lineWidth = 2;
      ctx.stroke();
      s.draft.forEach(p => {
        ctx.beginPath();
        ctx.arc(p[0], p[1], 4, 0, 2 * Math.PI);
        ctx.fillStyle = '#222';
        ctx.fill();
      });
    }
  }

  function canvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    return [(e.clientX - rect.left) * (W / rect.width),
            (e.clientY - rect.top) * (H / rect.height)];
  }

  function onMouseDown(e) {
    e.preventDefault();
    const [x, y] = canvasPos(e);

    if (e.button === 2) {
      const hit = findVertex(x, y);
      if (hit && hit.roi !== undefined) {
        s.rois[hit.roi].points.splice(hit.vi, 1);
        if (s.rois[hit.roi].points.length < 3) s.rois.splice(hit.roi, 1);
        s.selected = null;
        send();
      } else if (s.draft.length >= 3) {
        s.rois.push({
          id: s.nextId,
          name: 'ROI ' + s.nextId,
          color: PALETTE[(s.rois.length) % PALETTE.length],
          points: s.draft,
        });
        s.nextId += 1;
        s.draft = [];
        s.selected = null;
        send();
      } else {
        s.draft = [];
        s.selected = null;
        send();
      }
      return;
    }
    if (e.button !== 0) return;

    const hit = findVertex(x, y);
    if (hit) {
      s.selected = hit;
      s.dragging = hit;
      draw();
      return;
    }
    s.draft.push([x, y]);
    draw();
    send();
  }

  function onMouseMove(e) {
    if (!s.dragging) return;
    const [x, y] = canvasPos(e);
    if (s.dragging.roi !== undefined) s.rois[s.dragging.roi].points[s.dragging.vi] = [x, y];
    else if (s.dragging.draft) s.draft[s.dragging.vi] = [x, y];
    draw();
  }

  function onMouseUp() {
    if (s.dragging) { s.dragging = null; send(); }
  }

  function onKeyDown(e) {
    if ((e.key === 'Delete' || e.key === 'Backspace') && s.selected) {
      e.preventDefault();
      if (s.selected.roi !== undefined) {
        s.rois[s.selected.roi].points.splice(s.selected.vi, 1);
        if (s.rois[s.selected.roi].points.length < 3) s.rois.splice(s.selected.roi, 1);
      } else if (s.selected.draft) {
        s.draft.splice(s.selected.vi, 1);
      }
      s.selected = null;
      send();
    } else if (e.key === 'Escape') {
      s.draft = [];
      s.selected = null;
      send();
    }
  }

  if (!s.attached) {
    s.attached = true;
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('mouseup', onMouseUp);
    window.addEventListener('keydown', onKeyDown);
  }

  // Re-initialise from Python's data only when the python-side revision
  // changed (image switch, rename, delete). User interactions never trigger
  // this, so drawn points survive every rerun.
  if (s.revision !== data.revision) {
    s.revision = data.revision;
    s.rois = (data.rois || []).map(r => ({
      id: r.id, name: r.name, color: r.color,
      points: r.points.map(p => [p[0], p[1]]),
    }));
    // Python's copy of the draft lags one interaction behind. If the user is
    // mid-polygon when a python-side edit lands, keep the local (newer)
    // draft instead of wiping the points they just placed.
    const dataDraft = data.draft || [];
    if (dataDraft.length >= s.draft.length) {
      s.draft = dataDraft.map(p => [p[0], p[1]]);
    }
    s.nextId = data.nextId || 1;
    s.selected = null;
    s.dragging = null;
  }

  // Reload the background image only when it actually changed. Switching to
  // a different image means a new analysis context — drop any stale
  // in-progress draft (but NOT on the first mount, where the data is
  // authoritative).
  if (s.img === null || s.img.src !== data.image) {
    const imageChanged = s.img !== null && s.img.src !== data.image;
    const img = new Image();
    img.onload = draw;
    img.src = data.image;
    s.img = img;
    if (imageChanged) {
      s.draft = [];
    }
  }
  draw();
}
"""

#: The JS is an ES module; the palette must be inlined (JSON) at registration time.
_JS = _JS_TEMPLATE.replace("__PALETTE__", json.dumps(PALETTE))

_CANVAS = None
_CANVAS_MANAGER = None


def _get_canvas_component():
    """Return the (lazily registered, per-manager cached) canvas component.

    The component is registered against the current bidirectional-component
    manager. In a running Streamlit app the manager is a process singleton, so
    this registers exactly once; under AppTest each script run gets a fresh
    mock-runtime manager, so we key the cache by the manager instance to
    re-register when it changes.
    """
    global _CANVAS, _CANVAS_MANAGER
    from streamlit.components.v2.get_bidi_component_manager import (
        get_bidi_component_manager,
    )

    manager = get_bidi_component_manager()
    if _CANVAS is None or _CANVAS_MANAGER is not manager:
        _CANVAS = st.components.v2.component(
            "blenny_roi_canvas",
            html=_HTML,
            css=_CSS,
            js=_JS,
        )
        _CANVAS_MANAGER = manager
    return _CANVAS


def roi_canvas(
    image: Image.Image,
    rois: list[dict[str, Any]],
    draft: list[list[float]],
    *,
    next_id: int = 1,
    revision: int = 0,
    image_b64: str | None = None,
    height: int = 620,
) -> Any:
    """Mount the ROI canvas and return its ``ComponentResult``.

    ``image`` is the (already downscaled) display image; ``rois`` and
    ``draft`` are polygon vertex lists in display-image coordinates.
    ``revision`` is a python-side counter: the canvas re-initialises from
    ``data`` only when it changes (image switch / rename / delete), so user
    interactions are never wiped by re-renders.

    ``image_b64`` may be supplied to avoid re-encoding the (unchanged)
    display image on every rerun — pass the same value until the image
    actually changes.

    The result exposes ``result.state`` — a fresh ``{"rois", "draft",
    "nextId"}`` dict after each canvas interaction, ``None`` otherwise. The
    caller applies it (guarding against stale re-deliveries) and the canvas
    keeps its own internal state in sync via the revision mechanism.
    """
    if image_b64 is None:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buf.getvalue()).decode()

    data = {
        "image": f"data:image/jpeg;base64,{image_b64}",
        "width": image.size[0],
        "height": image.size[1],
        "rois": rois,
        "draft": draft,
        "nextId": next_id,
        "revision": revision,
    }
    return _get_canvas_component()(data=data, key="roi_canvas", height=height)
