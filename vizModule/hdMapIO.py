"""
HD-Map persistence: writer and reader for editable polygon, centerline,
and bike-lane segments produced by HDMapGridAccumulator.

Also provides HDMapRenderer – a standalone OpenGL renderer that displays
a loaded HDMapData snapshot without depending on the accumulator:
  • polygons      → green GL_LINE_LOOP outlines
  • centerline    → purple PosePathRenderer ribbon (adjustable width)
  • bike-lane segs → green PosePathRenderer ribbons (adjustable width)

File format: JSON  (human-readable, easy to diff in git)
Schema version: 1
"""

from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PathRendererModule import PosePathRenderer


# ---------------------------------------------------------------------------
# Minimal solid-colour line shader used by HDMapRenderer for polygon outlines
# ---------------------------------------------------------------------------

_POLY_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 view;
uniform mat4 projection;
uniform float u_z_lift;
void main() {
    vec3 pos = aPos + vec3(0.0, 0.0, u_z_lift);
    gl_Position = projection * view * vec4(pos, 1.0);
}
"""

_POLY_FRAG = """
#version 330 core
uniform vec4 u_color;
out vec4 FragColor;
void main() {
    FragColor = u_color;
}
"""


# ---------------------------------------------------------------------------
# Catmull-Rom open spline (self-contained, no ipmModule dependency)
# ---------------------------------------------------------------------------

def _catmull_rom_open(pts: np.ndarray, samples_per_seg: int = 5) -> np.ndarray:
    """
    Open Catmull-Rom spline through *pts*.
    First and last control points are clamped (no wrapping).
    Returns a dense (N, 3) float32 array.
    """
    n = len(pts)
    if n < 2:
        return pts.copy()
    # Phantom endpoints by clamping
    extended = np.vstack([pts[0:1], pts, pts[-1:]])   # (n+2, 3)
    result = []
    for i in range(1, n):                              # n-1 segments
        p0, p1, p2, p3 = extended[i-1], extended[i], extended[i+1], extended[i+2]
        for j in range(samples_per_seg):
            t = j / samples_per_seg
            result.append(0.5 * (
                2*p1 + (-p0 + p2)*t +
                (2*p0 - 5*p1 + 4*p2 - p3)*t*t +
                (-p0 + 3*p1 - 3*p2 + p3)*t*t*t
            ))
    result.append(pts[-1].copy())
    return np.array(result, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class HDMapData:
    """
    All editable HD-map elements that can be saved and restored.

    Attributes
    ----------
    polygons : list of (N, 3) float32 arrays
        Editable drivable-area polygon control points (one array per polygon).
    centerline : (N, 3) float32 array or None
        Vehicle centreline control points.
    bike_lane_segments : list of (N, 3) float32 arrays
        Each stored (committed) bike-lane segment as control points.
    bike_lane_active : (N, 3) float32 array or None
        The currently-being-edited bike-lane segment (not yet stored).
    bike_lane_width : float
        Half-width used when computing left/right boundaries (metres).
    """

    polygons: List[np.ndarray] = field(default_factory=list)
    centerline: Optional[np.ndarray] = None
    bike_lane_segments: List[np.ndarray] = field(default_factory=list)
    bike_lane_active: Optional[np.ndarray] = None
    bike_lane_width: float = 1.5
    crosswalks: List[np.ndarray] = field(default_factory=list)  # each (2, 3): [pt1, pt2]
    crosswalk_width: float = 3.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _arr_to_list(arr: Optional[np.ndarray]):
    """numpy array → nested Python list (JSON-serialisable), or None."""
    if arr is None:
        return None
    return arr.tolist()


def _list_to_arr(lst) -> Optional[np.ndarray]:
    """Nested Python list → float32 numpy array, or None."""
    if lst is None:
        return None
    return np.array(lst, dtype=np.float32)


# ---------------------------------------------------------------------------
# IO class
# ---------------------------------------------------------------------------

class HDMapIO:
    """Static writer / reader for HDMapData."""

    VERSION = 1

    # ------------------------------------------------------------------ save
    @staticmethod
    def save(filepath: str, data: HDMapData) -> None:
        """
        Serialise *data* to a JSON file at *filepath*.

        Parent directories are created automatically.
        Raises ``IOError`` on write failure.
        """
        payload = {
            "version": HDMapIO.VERSION,
            "polygons": [_arr_to_list(p) for p in data.polygons],
            "centerline": _arr_to_list(data.centerline),
            "bike_lane_segments": [_arr_to_list(s) for s in data.bike_lane_segments],
            "bike_lane_active": _arr_to_list(data.bike_lane_active),
            "bike_lane_width": float(data.bike_lane_width),
            "crosswalks": [_arr_to_list(c) for c in data.crosswalks],
            "crosswalk_width": float(data.crosswalk_width),
        }
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    # ------------------------------------------------------------------ load
    @staticmethod
    def load(filepath: str) -> HDMapData:
        """
        Deserialise a JSON file written by :meth:`save` and return an
        :class:`HDMapData` instance.

        Raises ``FileNotFoundError`` if *filepath* does not exist.
        Raises ``ValueError`` if the file version is unsupported.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"HD-map file not found: {filepath}")

        with open(path, "r") as fh:
            payload = json.load(fh)

        version = payload.get("version", 0)
        if version != HDMapIO.VERSION:
            raise ValueError(
                f"Unsupported HD-map file version {version} "
                f"(expected {HDMapIO.VERSION})"
            )

        return HDMapData(
            polygons=[
                _list_to_arr(p)
                for p in payload.get("polygons", [])
                if p is not None
            ],
            centerline=_list_to_arr(payload.get("centerline")),
            bike_lane_segments=[
                _list_to_arr(s)
                for s in payload.get("bike_lane_segments", [])
                if s is not None
            ],
            bike_lane_active=_list_to_arr(payload.get("bike_lane_active")),
            bike_lane_width=float(payload.get("bike_lane_width", 1.5)),
            crosswalks=[
                _list_to_arr(c)
                for c in payload.get("crosswalks", [])
                if c is not None
            ],
            crosswalk_width=float(payload.get("crosswalk_width", 3.0)),
        )

    # --------------------------------------------------------------- summary
    @staticmethod
    def summary(data: HDMapData) -> str:
        """Return a one-line human-readable summary of *data*."""
        n_poly_pts = sum(len(p) for p in data.polygons)
        n_cl = len(data.centerline) if data.centerline is not None else 0
        n_bl_segs = len(data.bike_lane_segments)
        n_bl_active = len(data.bike_lane_active) if data.bike_lane_active is not None else 0
        return (
            f"polygons={len(data.polygons)}({n_poly_pts}pts)  "
            f"centerline={n_cl}pts  "
            f"bike_lane={n_bl_segs}segs+{n_bl_active}active  "
            f"width={data.bike_lane_width:.2f}m  "
            f"crosswalks={len(data.crosswalks)}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crosswalk_rect(pt1: np.ndarray, pt2: np.ndarray,
                    width: float) -> Optional[np.ndarray]:
    """
    Compute the 4 corners of a crosswalk rectangle from its two end-centre
    points and the desired width.  Returns a (4, 3) float32 array ordered
    for ``GL_LINE_LOOP``, or ``None`` if the two points are degenerate.
    """
    dx = float(pt2[0] - pt1[0])
    dy = float(pt2[1] - pt1[1])
    L = np.sqrt(dx * dx + dy * dy)
    if L < 1e-6:
        return None
    rx, ry = -dy / L, dx / L          # unit vector perpendicular to the axis
    hw = width * 0.5
    return np.array([
        [pt1[0] + rx * hw, pt1[1] + ry * hw, 0.0],
        [pt1[0] - rx * hw, pt1[1] - ry * hw, 0.0],
        [pt2[0] - rx * hw, pt2[1] - ry * hw, 0.0],
        [pt2[0] + rx * hw, pt2[1] + ry * hw, 0.0],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Standalone GPU renderer
# ---------------------------------------------------------------------------

class HDMapRenderer:
    """
    Standalone OpenGL renderer for a loaded :class:`HDMapData` snapshot.

    Rendering:
    • **Polygon outlines** — green ``GL_LINE_LOOP`` (same style as run())
    • **Centerline**        — purple :class:`PosePathRenderer` ribbon
    • **Bike-lane segments** — green :class:`PosePathRenderer` ribbons
      (stored segments + active segment are all rendered)

    Typical usage::

        renderer = HDMapRenderer()            # once, after GL context exists
        renderer.update(data)                 # after load / edit
        # inside render loop:
        renderer.draw(view.T, proj.T)
    """

    POLY_COLOR = (0.3, 1.0, 0.3, 1.0)    # green
    CL_COLOR   = (0.6, 0.0, 0.9, 0.85)   # purple
    BL_COLOR   = (0.0, 0.85, 0.3, 0.85)  # green
    CW_COLOR   = (1.0, 1.0, 1.0, 1.0)    # white

    # ------------------------------------------------------------------
    def __init__(self, cl_width: float = 1.0, bl_width: float = 1.5):
        # Compile the solid-colour polygon-line shader
        self._prog = compileProgram(
            compileShader(_POLY_VERT, GL_VERTEX_SHADER),
            compileShader(_POLY_FRAG, GL_FRAGMENT_SHADER),
        )
        glUseProgram(self._prog)
        self._loc_view   = glGetUniformLocation(self._prog, "view")
        self._loc_proj   = glGetUniformLocation(self._prog, "projection")
        self._loc_color  = glGetUniformLocation(self._prog, "u_color")
        self._loc_z_lift = glGetUniformLocation(self._prog, "u_z_lift")
        glUseProgram(0)

        self.cl_width = cl_width
        self.bl_width = bl_width

        # PosePathRenderer instances (created once; ribbons are re-uploaded on update)
        self._cl_ribbon  = PosePathRenderer(width=cl_width)
        self._bl_ribbons: List[PosePathRenderer] = []  # rebuilt in update()

        # Polygon line geometry: list of (VAO, VBO, vertex_count)
        self._poly_geom: list = []

        # Crosswalk rectangle geometry: list of (VAO, VBO, vertex_count=4)
        self._cw_geom: list = []

    # ------------------------------------------------------------------
    def update(self, data: HDMapData,
               cl_width: Optional[float] = None,
               bl_width: Optional[float] = None) -> None:
        """
        Upload *data* geometry to the GPU.
        Safe to call inside the render loop — active GL program / VAO
        are saved and restored around every :class:`PosePathRenderer`
        construction.
        """
        if cl_width is not None:
            self.cl_width = cl_width
        if bl_width is not None:
            self.bl_width = bl_width

        # Save GL state (PosePathRenderer.__init__ calls glUseProgram)
        prev_prog = int(glGetIntegerv(GL_CURRENT_PROGRAM))
        prev_vao  = int(glGetIntegerv(GL_VERTEX_ARRAY_BINDING))

        # ── Centerline ribbon (purple) ────────────────────────────────
        self._cl_ribbon.width = self.cl_width
        if data.centerline is not None and len(data.centerline) >= 2:
            smooth = _catmull_rom_open(data.centerline)
            self._cl_ribbon.update_from_positions(list(smooth))
        else:
            self._cl_ribbon._vertex_count = 0

        # ── Bike-lane ribbons (green) — stored + active ───────────────
        self._bl_ribbons = []
        all_bl_pts = list(data.bike_lane_segments)
        if data.bike_lane_active is not None and len(data.bike_lane_active) >= 2:
            all_bl_pts.append(data.bike_lane_active)

        for seg in all_bl_pts:
            if seg is None or len(seg) < 2:
                continue
            smooth = _catmull_rom_open(seg)
            ribbon = PosePathRenderer(width=self.bl_width)
            ribbon.update_from_positions(list(smooth))
            self._bl_ribbons.append(ribbon)

        # ── Polygon line geometry ─────────────────────────────────────
        # Free any previously allocated buffers
        for (vao, vbo, _) in self._poly_geom:
            glDeleteVertexArrays(1, [vao])
            glDeleteBuffers(1, [vbo])
        self._poly_geom = []

        for poly in data.polygons:
            if poly is None or len(poly) < 2:
                continue
            verts = np.asarray(poly, dtype=np.float32)
            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            glBindVertexArray(0)
            self._poly_geom.append((int(vao), int(vbo), len(verts)))

        # ── Crosswalk rectangle geometry ─────────────────────────────────
        for (vao, vbo, _) in self._cw_geom:
            glDeleteVertexArrays(1, [vao])
            glDeleteBuffers(1, [vbo])
        self._cw_geom = []

        cw_width = data.crosswalk_width
        for cw in data.crosswalks:
            if cw is None or np.asarray(cw).shape != (2, 3):
                continue
            cw = np.asarray(cw, dtype=np.float32)
            corners = _crosswalk_rect(cw[0], cw[1], cw_width)
            if corners is None:
                continue
            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, corners.nbytes, corners, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            glBindVertexArray(0)
            self._cw_geom.append((int(vao), int(vbo), 4))

        # Restore GL state
        glUseProgram(prev_prog)
        glBindVertexArray(prev_vao)

    # ------------------------------------------------------------------
    def draw(self, view: np.ndarray, proj: np.ndarray,
             z_lift: float = 0.07) -> None:
        """
        Draw all HD-map elements.

        *view* and *proj* must already be column-major (i.e. transposed
        compared to row-major NumPy convention — pass ``view.T`` / ``proj.T``
        from the render loop, exactly as you would to PosePathRenderer).
        """
        # ── Centerline ribbon (purple) ────────────────────────────────
        self._cl_ribbon.draw(view, proj, color=self.CL_COLOR)

        # ── Bike-lane ribbons (green) ─────────────────────────────────
        for ribbon in self._bl_ribbons:
            ribbon.draw(view, proj, color=self.BL_COLOR)

        # ── Polygon outlines (green GL_LINE_LOOP) ─────────────────────
        if not self._poly_geom:
            return

        glUseProgram(self._prog)
        glUniformMatrix4fv(self._loc_view,  1, GL_FALSE, view)
        glUniformMatrix4fv(self._loc_proj,  1, GL_FALSE, proj)
        glUniform4f(self._loc_color, *self.POLY_COLOR)
        glUniform1f(self._loc_z_lift, z_lift)

        glEnable(GL_DEPTH_TEST)
        glLineWidth(2.0)

        for (vao, _, count) in self._poly_geom:
            glBindVertexArray(vao)
            glDrawArrays(GL_LINE_LOOP, 0, count)

        # ── Crosswalk rectangles (white GL_LINE_LOOP) ─────────────────────
        if self._cw_geom:
            glUniform4f(self._loc_color, *self.CW_COLOR)
            glLineWidth(2.5)
            for (vao, _, count) in self._cw_geom:
                glBindVertexArray(vao)
                glDrawArrays(GL_LINE_LOOP, 0, count)
            glLineWidth(1.0)

        glBindVertexArray(0)
        glLineWidth(1.0)
        glUseProgram(0)
