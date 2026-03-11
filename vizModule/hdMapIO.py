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
import math
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
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float u_z_lift;
out float vWorldZ;
void main() {
    vec4 worldPos = model * vec4(aPos + vec3(0.0, 0.0, u_z_lift), 1.0);
    vWorldZ = worldPos.z;
    gl_Position = projection * view * worldPos;
}
"""

_POLY_FRAG = """
#version 330 core
in float vWorldZ;
uniform vec4 u_color;
uniform vec4 u_color_bottom;
uniform float u_z_top;
out vec4 FragColor;
void main() {
    // When u_z_top == 0 (default) t=1 → flat u_color.  Otherwise blend
    // from u_color_bottom (z=0) to u_color (z=u_z_top) for height gradient.
    float t = (u_z_top > 0.0) ? clamp(vWorldZ / u_z_top, 0.0, 1.0) : 1.0;
    FragColor = mix(u_color_bottom, u_color, t);
}
"""

# ---------------------------------------------------------------------------
# Flat textured-quad shader (bike-lane road icons)
# ---------------------------------------------------------------------------

_BIKE_ICON_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTex;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec2 vTex;
void main() {
    vTex = aTex;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

_BIKE_ICON_FRAG = """
#version 330 core
in vec2 vTex;
out vec4 FragColor;
uniform sampler2D u_texture;
void main() {
    FragColor = texture(u_texture, vTex);
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
    buildings: List[np.ndarray] = field(default_factory=list)   # each (N, 3): closed polygon pts


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


def _ensure_3d(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Ensure array is (N, 3) float32; if points are (N, 2), append z=0. Preserves existing z."""
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] == 2:
        arr = np.column_stack([arr, np.zeros(len(arr), dtype=np.float32)])
    return arr


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
            "buildings": [_arr_to_list(b) for b in data.buildings],
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
                _ensure_3d(_list_to_arr(p))
                for p in payload.get("polygons", [])
                if p is not None
            ],
            centerline=_ensure_3d(_list_to_arr(payload.get("centerline"))),
            bike_lane_segments=[
                _ensure_3d(_list_to_arr(s))
                for s in payload.get("bike_lane_segments", [])
                if s is not None
            ],
            bike_lane_active=_ensure_3d(_list_to_arr(payload.get("bike_lane_active"))),
            bike_lane_width=float(payload.get("bike_lane_width", 1.5)),
            crosswalks=[
                _ensure_3d(_list_to_arr(c))
                for c in payload.get("crosswalks", [])
                if c is not None
            ],
            crosswalk_width=float(payload.get("crosswalk_width", 3.0)),
            buildings=[
                _ensure_3d(_list_to_arr(b))
                for b in payload.get("buildings", [])
                if b is not None
            ],
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
            f"crosswalks={len(data.crosswalks)}  "
            f"buildings={len(data.buildings)}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ear_clip_triangulate(pts_2d: np.ndarray) -> list:
    """
    Triangulate a simple (non-self-intersecting) 2-D polygon using ear clipping.

    Works correctly for concave/non-convex polygons.  Fan triangulation fails
    on concave shapes because it connects vertex 0 to every other vertex,
    cutting straight across re-entrant corners.

    pts_2d : (N, 2) float array – polygon vertices in order (CW or CCW).
    Returns : flat list of (2,) float64 arrays, length == (N-2)*3.
              Every 3 entries form one triangle.
    """
    pts = np.asarray(pts_2d, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [pts[0], pts[1], pts[2]]

    # Ensure counter-clockwise winding (signed area > 0)
    x, y = pts[:, 0], pts[:, 1]
    signed_area2 = (np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])
                    + x[-1] * y[0] - x[0] * y[-1])
    if signed_area2 < 0:
        pts = pts[::-1].copy()

    idx = list(range(len(pts)))
    tris: list = []
    max_iters = len(idx) ** 2 + len(idx)
    i = 0

    while len(idx) > 3 and max_iters > 0:
        max_iters -= 1
        nc = len(idx)
        pi, ci, ni = idx[(i - 1) % nc], idx[i % nc], idx[(i + 1) % nc]
        a, b, c = pts[pi], pts[ci], pts[ni]

        # Cross product of edges AB and AC — positive means convex (left-turn) vertex
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if cross > 1e-10:
            # Verify no other polygon vertex lies strictly inside triangle ABC
            ear = True
            for j in range(nc):
                v = idx[j]
                if v in (pi, ci, ni):
                    continue
                p = pts[v]
                d1 = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                d2 = (c[0]-b[0])*(p[1]-b[1]) - (c[1]-b[1])*(p[0]-b[0])
                d3 = (a[0]-c[0])*(p[1]-c[1]) - (a[1]-c[1])*(p[0]-c[0])
                if d1 > -1e-10 and d2 > -1e-10 and d3 > -1e-10:
                    ear = False
                    break
            if ear:
                tris += [a.copy(), b.copy(), c.copy()]
                idx.pop(i % nc)
                i = max(0, (i - 1) % max(len(idx), 1))
                continue

        i = (i + 1) % len(idx)

    # Last remaining triangle
    if len(idx) == 3:
        tris += [pts[idx[0]], pts[idx[1]], pts[idx[2]]]

    return tris


def _offset_polygon_outward(pts_2d: np.ndarray, offset: float,
                             max_miter: float = 3.0) -> np.ndarray:
    """
    Offset each vertex of a simple polygon outward by *offset* metres using
    miter joints (the standard technique for sharp-corner offset).

    The polygon is normalised to CCW winding internally so that "outward" is
    always away from the interior.  Miter length is clamped to
    *max_miter* × *offset* to prevent extreme spikes at very sharp corners.

    pts_2d : (N, 2) float array – polygon vertices.
    Returns : (N, 2) float64 array – offset polygon vertices.
    """
    pts = np.asarray(pts_2d, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return pts.copy()

    # Ensure CCW winding (outward normal = right-perpendicular of each edge)
    x, y = pts[:, 0], pts[:, 1]
    area2 = (np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])
             + x[-1] * y[0] - x[0] * y[-1])
    if area2 < 0:
        pts = pts[::-1].copy()

    result = np.zeros_like(pts)
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]

        e1 = p1 - p0
        e2 = p2 - p1
        l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
        if l1 < 1e-9 or l2 < 1e-9:
            result[i] = p1
            continue

        # Right-perpendicular = outward unit normal for a CCW polygon
        n1 = np.array([ e1[1], -e1[0]]) / l1
        n2 = np.array([ e2[1], -e2[0]]) / l2

        bisector = n1 + n2
        b_len = np.linalg.norm(bisector)
        if b_len < 1e-9:
            # 180° straight segment — just shift along n1
            result[i] = p1 + n1 * offset
        else:
            bisector /= b_len
            dot_val = float(np.dot(bisector, n1))
            # clamp denominator to avoid division by ~0 at very sharp corners
            dot_val = max(abs(dot_val), 1.0 / max_miter) * (1 if dot_val >= 0 else -1)
            miter_len = offset / dot_val
            result[i] = p1 + bisector * miter_len

    return result


# Road surface z = poly vertex z + ROAD_LIFT. Crosswalks use road z + CROSSWALK_LIFT so they draw on top.
ROAD_LIFT = 0.001
CROSSWALK_LIFT = 0.004   # crosswalk geometry sits this much above road to avoid z-fighting and clipping


def _road_z_at_xy(x: float, y: float, polygons: list, road_lift: float = 0.001,
                  extra_lift: float = 0.0) -> float:
    """Return road surface z at (x,y) from nearest polygon vertex. extra_lift e.g. CROSSWALK_LIFT."""
    best_z = 0.0
    best_d2 = float("inf")
    for poly in polygons or []:
        if poly is None or len(poly) < 3:
            continue
        poly = np.asarray(poly, dtype=np.float64)
        if poly.shape[1] < 3:
            continue
        for i in range(len(poly)):
            d2 = (x - float(poly[i, 0])) ** 2 + (y - float(poly[i, 1])) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_z = float(poly[i, 2]) + road_lift + extra_lift
    return best_z


def _crosswalk_rect(pt1: np.ndarray, pt2: np.ndarray,
                    width: float, polygons: Optional[list] = None) -> Optional[np.ndarray]:
    """
    Compute the 4 corners of a crosswalk rectangle from its two end-centre
    points and the desired width.  Returns a (4, 3) float32 array ordered
    for ``GL_LINE_LOOP``, or ``None`` if the two points are degenerate.
    If *polygons* is provided, z is taken from road surface at each corner (so crosswalk sits on street).
    """
    dx = float(pt2[0] - pt1[0])
    dy = float(pt2[1] - pt1[1])
    L = np.sqrt(dx * dx + dy * dy)
    if L < 1e-6:
        return None
    rx, ry = -dy / L, dx / L          # unit vector perpendicular to the axis
    hw = width * 0.5
    if polygons:
        z1 = _road_z_at_xy(float(pt1[0]), float(pt1[1]), polygons, extra_lift=CROSSWALK_LIFT)
        z2 = _road_z_at_xy(float(pt2[0]), float(pt2[1]), polygons, extra_lift=CROSSWALK_LIFT)
    else:
        z1 = float(pt1[2]) if len(pt1) >= 3 else 0.0
        z2 = float(pt2[2]) if len(pt2) >= 3 else 0.0
    return np.array([
        [pt1[0] + rx * hw, pt1[1] + ry * hw, z1],
        [pt1[0] - rx * hw, pt1[1] - ry * hw, z1],
        [pt2[0] - rx * hw, pt2[1] - ry * hw, z2],
        [pt2[0] + rx * hw, pt2[1] + ry * hw, z2],
    ], dtype=np.float32)


def _crosswalk_stripe_tris(pt1: np.ndarray, pt2: np.ndarray,
                            width: float, polygons: Optional[list] = None) -> Optional[np.ndarray]:
    """
    Build zebra-stripe triangle vertices for a crosswalk defined by two
    centre-line endpoints and a width.

    Returns a (N, 3) float32 array where N is a multiple of 3 (GL_TRIANGLES),
    or ``None`` if the points are degenerate.
    If *polygons* is provided, z is taken from road surface at each vertex (so crosswalk sits on street).
    """
    dx = float(pt2[0] - pt1[0])
    dy = float(pt2[1] - pt1[1])
    L = np.sqrt(dx * dx + dy * dy)
    if L < 1e-6:
        return None
    ux, uy = dx / L, dy / L           # unit along crosswalk
    rx, ry = -dy / L, dx / L          # unit perpendicular
    hw = width * 0.5

    # 5 stripes minimum; add 1 per metre beyond 5 m (mirrors C++ / ipmModule logic)
    num_stripes = 5
    if L > 5.0:
        num_stripes += int(L - 5.0)

    stripe_length = L / (2 * num_stripes)
    verts: list = []
    for idx in range(num_stripes):
        start_t = idx * 2 * stripe_length
        end_t   = start_t + stripe_length
        x0 = pt1[0] + ux * start_t
        y0 = pt1[1] + uy * start_t
        x1 = pt1[0] + ux * end_t
        y1 = pt1[1] + uy * end_t
        if polygons:
            z_start = _road_z_at_xy(x0, y0, polygons, extra_lift=CROSSWALK_LIFT)
            z_end   = _road_z_at_xy(x1, y1, polygons, extra_lift=CROSSWALK_LIFT)
        else:
            z1 = float(pt1[2]) if len(pt1) >= 3 else 0.0
            z2 = float(pt2[2]) if len(pt2) >= 3 else 0.0
            t0, t1 = start_t / L, end_t / L
            z_start = (1.0 - t0) * z1 + t0 * z2
            z_end   = (1.0 - t1) * z1 + t1 * z2

        c0 = [x0 + rx * hw, y0 + ry * hw, z_start]
        c1 = [x0 - rx * hw, y0 - ry * hw, z_start]
        c2 = [x1 - rx * hw, y1 - ry * hw, z_end]
        c3 = [x1 + rx * hw, y1 + ry * hw, z_end]

        # Two triangles per stripe quad
        verts.extend([c0, c1, c2, c0, c2, c3])

    return np.array(verts, dtype=np.float32) if verts else None


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

    POLY_COLOR      = (0.3, 1.0, 0.3, 1.0)      # green outline
    POLY_FILL_COLOR = (0.45, 0.45, 0.45, 0.85)  # gray solid fill (road)
    SIDEWALK_COLOR  = (0.72, 0.72, 0.67, 1.0)   # warm light-gray (sidewalk top / wall top)
    SIDEWALK_BOT_COLOR = (0.18, 0.18, 0.16, 1.0) # dark charcoal (wall bottom)
    CL_COLOR        = (0.6, 0.0, 0.9, 0.8)      # purple, flat solid opacity 0.8
    BL_COLOR        = (0.0, 0.85, 0.3, 0.8)     # green,  flat solid opacity 0.8
    CW_COLOR        = (1.0, 1.0, 1.0, 1.0)      # white
    BLD_COLOR       = (0.2, 0.85, 1.0, 0.55)    # cyan-blue, semi-transparent fill
    BLD_EDGE_COLOR  = (0.2, 0.85, 1.0, 1.0)     # cyan-blue opaque outline

    # ------------------------------------------------------------------
    def __init__(self, cl_width: float = 1.0, bl_width: float = 1.5,
                 building_height: float = 3.0,
                 sidewalk_width: float = 0.3,
                 sidewalk_height: float = 0.2):
        # Compile the solid-colour polygon-line shader
        self._prog = compileProgram(
            compileShader(_POLY_VERT, GL_VERTEX_SHADER),
            compileShader(_POLY_FRAG, GL_FRAGMENT_SHADER),
        )
        glUseProgram(self._prog)
        self._loc_model    = glGetUniformLocation(self._prog, "model")
        self._loc_view     = glGetUniformLocation(self._prog, "view")
        self._loc_proj     = glGetUniformLocation(self._prog, "projection")
        self._loc_color    = glGetUniformLocation(self._prog, "u_color")
        self._loc_color_bot = glGetUniformLocation(self._prog, "u_color_bottom")
        self._loc_z_top    = glGetUniformLocation(self._prog, "u_z_top")
        self._loc_z_lift   = glGetUniformLocation(self._prog, "u_z_lift")
        glUseProgram(0)

        self.cl_width        = cl_width
        self.bl_width        = bl_width
        self.building_height = building_height
        self.sidewalk_width  = sidewalk_width
        self.sidewalk_height = sidewalk_height

        self.lane_z      = 0.005   # z baked into centerline & bike-lane ribbon vertices
        self.crosswalk_z = 0.004   # z_lift applied to crosswalk geometry in the shader

        # Set to False to hide the green polygon outlines (e.g. after a map upload)
        self.show_polygon_outlines = False

        # PosePathRenderer instances — centerline: flat solid fill
        self._cl_ribbon  = PosePathRenderer(width=cl_width, flat_color=True)
        self._bl_ribbons: List[PosePathRenderer] = []  # rebuilt in update()

        # Sidewalk top-face geometry (flat ring at z=sh, no gradient)
        self._sw_top_geom: list = []
        # Sidewalk wall geometry (inner + outer walls, dark→gray gradient)
        self._sw_wall_geom: list = []

        # Polygon solid fill geometry (fan triangles, flat): list of (VAO, VBO, vertex_count)
        self._poly_solid_geom: list = []

        # Polygon line geometry: list of (VAO, VBO, vertex_count)
        self._poly_geom: list = []

        # Crosswalk rectangle geometry: list of (VAO, VBO, vertex_count=4)
        self._cw_geom: list = []

        # Crosswalk zebra-stripe triangle geometry: list of (VAO, VBO, vertex_count)
        self._cw_stripe_geom: list = []

        # Building solid geometry (floor + walls + roof triangles): list of (VAO, VBO, count)
        self._bld_solid_geom: list = []

        # Building outline geometry (roof loop): list of (VAO, VBO, vertex_count)
        self._bld_geom: list = []

        # ── Flat bike-lane icon quads ──────────────────────────────────────
        self._bike_icon_prog = compileProgram(
            compileShader(_BIKE_ICON_VERT, GL_VERTEX_SHADER),
            compileShader(_BIKE_ICON_FRAG, GL_FRAGMENT_SHADER),
        )
        glUseProgram(self._bike_icon_prog)
        self._loc_bi_model = glGetUniformLocation(self._bike_icon_prog, "model")
        self._loc_bi_view  = glGetUniformLocation(self._bike_icon_prog, "view")
        self._loc_bi_proj  = glGetUniformLocation(self._bike_icon_prog, "projection")
        self._loc_bi_tex   = glGetUniformLocation(self._bike_icon_prog, "u_texture")
        glUseProgram(0)

        self._bike_icon_tex   = 0   # GL texture ID (0 = none loaded yet)
        self._bike_icon_count = 0   # number of vertices
        self._bike_icon_vao   = glGenVertexArrays(1)
        self._bike_icon_vbo   = glGenBuffers(1)
        _bi_stride = (3 + 2) * 4
        glBindVertexArray(self._bike_icon_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._bike_icon_vbo)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, _bi_stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, _bi_stride, ctypes.c_void_p(3 * 4))
        glBindVertexArray(0)

        # Configurable icon parameters
        self.bike_icon_size     = 1.0   # side length of the flat square icon (metres)
        self.bike_icon_spacing  = 2.0   # distance between icon centres along the lane (metres)
        self.bike_icon_z        = self.lane_z + 0.05   # floats 0.05 m above the ribbon
        self.bike_icon_rotation = 180.0   # extra rotation applied to the icon (degrees, CCW from above)

    # ------------------------------------------------------------------
    def load_bike_lane_icon(self, pil_icons: dict) -> None:
        """
        Upload the ``"bicycle_lane"`` entry from *pil_icons* as a GL texture.

        Call once after the GL context is ready, passing the same icons dict
        you give to ``IconRenderer.load_icons()``.  Icons will not be drawn
        until this method is called.

        Example::

            renderer.load_bike_lane_icon(icons)   # icons["bicycle_lane"] is a PIL Image
        """
        img = pil_icons.get("bicycle_lane")
        if img is None:
            return

        if hasattr(img, "convert"):         # PIL Image
            img = img.convert("RGBA")
            w, h = img.size
            data = np.asarray(img, dtype=np.uint8)
        else:                               # already a numpy (H, W, 4) array
            data = np.asarray(img, dtype=np.uint8)
            h, w = data.shape[:2]

        if self._bike_icon_tex:
            glDeleteTextures(1, [self._bike_icon_tex])

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, data)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        self._bike_icon_tex = int(tex)

    # ------------------------------------------------------------------
    def _sample_lane_with_tangents(self, positions: list) -> list:
        """
        Walk *positions* (list of np.array([x,y,z])) and collect one
        ``(pos, forward_unit)`` tuple every ``bike_icon_spacing`` metres.

        A margin of ``bike_icon_size / 2`` is enforced at both ends so no
        icon quad ever extends beyond the lane's start or end point.
        """
        pts = [np.asarray(p, dtype=np.float32) for p in positions]
        n = len(pts)
        if n < 2:
            return []

        # Total arc length (needed to enforce the end margin)
        total_len = 0.0
        for i in range(n - 1):
            total_len += float(np.linalg.norm(pts[i + 1] - pts[i]))

        margin   = self.bike_icon_size * 0.5
        start_d  = margin                   # first icon centre >= margin from start
        end_d    = total_len - margin        # last  icon centre <= margin from end

        if end_d < start_d:                 # lane too short to fit even one icon
            return []

        out:  list = []
        accum  = 0.0
        next_d = start_d

        for i in range(n - 1):
            p0, p1 = pts[i], pts[i + 1]
            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-8:
                continue
            forward = seg / seg_len

            while accum + seg_len >= next_d and next_d <= end_d:
                t = (next_d - accum) / seg_len
                out.append((p0 + seg * t, forward.copy()))
                next_d += self.bike_icon_spacing

            accum += seg_len

        return out

    # ------------------------------------------------------------------
    def _build_bike_icon_quads(self, all_lanes: list) -> np.ndarray:
        """
        For every lane in *all_lanes* (list of position-lists), sample every
        ``bike_icon_spacing`` m and build a flat horizontal ``GL_TRIANGLES``
        quad at each sample.

        The quad lies in the XY plane at ``z = bike_icon_z``, with its
        "top" edge pointing in the lane's forward direction.
        ``bike_icon_rotation`` (degrees, CCW from above) is applied on top of
        the lane-aligned orientation so the icon can be fine-tuned visually.

        Returns a flat float32 array of (pos3 + uv2) values.
        """
        verts: list = []
        half = self.bike_icon_size * 0.5

        # Pre-compute rotation components (only once per rebuild)
        angle = math.radians(self.bike_icon_rotation)
        rc, rs = math.cos(angle), math.sin(angle)

        for positions in all_lanes:
            for pos, forward in self._sample_lane_with_tangents(positions):
                # Lane-aligned basis in world XY
                right = np.array([-forward[1], forward[0], 0.0], dtype=np.float32)
                rn = float(np.linalg.norm(right))
                if rn < 1e-8:
                    continue
                right /= rn

                fwd = np.array([forward[0], forward[1], 0.0], dtype=np.float32)
                fn = float(np.linalg.norm(fwd))
                if fn < 1e-8:
                    continue
                fwd /= fn

                # Apply extra rotation around Z: rotate both basis vectors by angle
                r_rot = np.array([
                    rc * right[0] - rs * right[1],
                    rs * right[0] + rc * right[1],
                    0.0,
                ], dtype=np.float32)
                f_rot = np.array([
                    rc * fwd[0] - rs * fwd[1],
                    rs * fwd[0] + rc * fwd[1],
                    0.0,
                ], dtype=np.float32)

                # Use lane z when available so icons sit just above the ribbon
                icon_z = (float(pos[2]) + 0.05) if len(pos) >= 3 else self.bike_icon_z
                p = np.array([pos[0], pos[1], icon_z], dtype=np.float32)

                bl = p - r_rot * half - f_rot * half
                br = p + r_rot * half - f_rot * half
                tr = p + r_rot * half + f_rot * half
                tl = p - r_rot * half + f_rot * half

                # Two CCW triangles; UV: bottom of image → -f_rot, top → +f_rot
                verts += [
                    *bl, 0.0, 1.0,
                    *br, 1.0, 1.0,
                    *tr, 1.0, 0.0,
                    *bl, 0.0, 1.0,
                    *tr, 1.0, 0.0,
                    *tl, 0.0, 0.0,
                ]

        if not verts:
            return np.empty(0, dtype=np.float32)
        return np.array(verts, dtype=np.float32)

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
            pts = [
                np.array([p[0], p[1], p[2] if len(p) >= 3 else self.lane_z], dtype=np.float32)
                for p in smooth
            ]
            self._cl_ribbon.update_from_positions(pts)
        else:
            self._cl_ribbon._vertex_count = 0

        # ── Bike-lane ribbons (green) — stored + active ───────────────
        self._bl_ribbons = []
        all_bl_pts = list(data.bike_lane_segments)
        if data.bike_lane_active is not None and len(data.bike_lane_active) >= 2:
            all_bl_pts.append(data.bike_lane_active)

        _all_smooth_bl: list = []   # kept for icon placement below
        for seg in all_bl_pts:
            if seg is None or len(seg) < 2:
                continue
            smooth = _catmull_rom_open(seg)
            pts = [
                np.array([p[0], p[1], p[2] if len(p) >= 3 else self.lane_z], dtype=np.float32)
                for p in smooth
            ]
            ribbon = PosePathRenderer(width=self.bl_width, outlined=True)
            ribbon.update_from_positions(pts)
            self._bl_ribbons.append(ribbon)
            _all_smooth_bl.append(pts)

        # ── Bike-lane icon quads (flat, every bike_icon_spacing metres) ──
        icon_data = self._build_bike_icon_quads(_all_smooth_bl)
        self._bike_icon_count = len(icon_data) // 5
        glBindBuffer(GL_ARRAY_BUFFER, self._bike_icon_vbo)
        if self._bike_icon_count > 0:
            glBufferData(GL_ARRAY_BUFFER, icon_data.nbytes, icon_data, GL_DYNAMIC_DRAW)
        else:
            glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)

        # ── Sidewalk 3-D band geometry ────────────────────────────────
        for geom_list in (self._sw_top_geom, self._sw_wall_geom):
            for (vao, vbo, _) in geom_list:
                glDeleteVertexArrays(1, [vao])
                glDeleteBuffers(1, [vbo])
        self._sw_top_geom  = []
        self._sw_wall_geom = []

        def _upload_verts(verts_list, target):
            if not verts_list:
                return
            arr = np.array(verts_list, dtype=np.float32)
            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            glBindVertexArray(0)
            target.append((int(vao), int(vbo), len(arr)))

        sw = self.sidewalk_width
        sh = self.sidewalk_height
        ROAD_LIFT = 0.001  # road surface above polygon vertex; sidewalk base matches road
        for poly in data.polygons:
            if poly is None or len(poly) < 3:
                continue
            poly_arr = np.asarray(poly, dtype=np.float64)
            inner_2d = poly_arr[:, :2]
            inner_z = poly_arr[:, 2] if poly_arr.shape[1] >= 3 else np.zeros(len(poly_arr), dtype=np.float64)

            # Normalise inner polygon to CCW so that inner_2d[i] and outer_2d[i]
            # are correctly paired (offset_polygon_outward reverses CW input
            # internally, which would shift the index correspondence otherwise).
            x, y = inner_2d[:, 0], inner_2d[:, 1]
            area2 = (np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])
                     + x[-1] * y[0] - x[0] * y[-1])
            if area2 < 0:
                inner_2d = inner_2d[::-1]
                inner_z = inner_z[::-1]

            outer_2d = _offset_polygon_outward(inner_2d, sw, max_miter=1.5)
            n_sw = len(inner_2d)

            top_tris:  list = []
            wall_tris: list = []
            for i in range(n_sw):
                j = (i + 1) % n_sw
                A2 = inner_2d[i];  B2 = inner_2d[j]
                C2 = outer_2d[j];  D2 = outer_2d[i]
                # Sidewalk base at road level (poly z + ROAD_LIFT) so it matches street
                zi, zj = float(inner_z[i]) + ROAD_LIFT, float(inner_z[j]) + ROAD_LIFT

                A0 = [A2[0], A2[1], zi];  A1 = [A2[0], A2[1], zi + sh]
                B0 = [B2[0], B2[1], zj];  B1 = [B2[0], B2[1], zj + sh]
                C0 = [C2[0], C2[1], zj];  C1 = [C2[0], C2[1], zj + sh]
                D0 = [D2[0], D2[1], zi];  D1 = [D2[0], D2[1], zi + sh]

                # ── Top face (flat at z=sh) ────────────────────────────────
                if np.sum((C2 - A2) ** 2) <= np.sum((D2 - B2) ** 2):
                    top_tris += [A1, B1, C1,  A1, C1, D1]
                else:
                    top_tris += [A1, B1, D1,  B1, C1, D1]

                # ── Walls (dark→gray gradient over z=0..sh) ───────────────
                wall_tris += [A0, B0, B1,  A0, B1, A1]   # inner wall
                wall_tris += [D1, C1, C0,  D1, C0, D0]   # outer wall

            _upload_verts(top_tris,  self._sw_top_geom)
            _upload_verts(wall_tris, self._sw_wall_geom)

        # ── Polygon solid fill + outline geometry ─────────────────────
        for (vao, vbo, _) in self._poly_solid_geom:
            glDeleteVertexArrays(1, [vao])
            glDeleteBuffers(1, [vbo])
        self._poly_solid_geom = []

        for (vao, vbo, _) in self._poly_geom:
            glDeleteVertexArrays(1, [vao])
            glDeleteBuffers(1, [vbo])
        self._poly_geom = []

        for poly in data.polygons:
            if poly is None or len(poly) < 3:
                continue
            pts = np.asarray(poly, dtype=np.float32)

            # Ear-clipping triangulation — handles concave road polygons correctly.
            tri_pts_2d = _ear_clip_triangulate(pts[:, :2])
            if not tri_pts_2d:
                continue
            # Road fill: per-vertex z from polygon so road follows street elevation
            ROAD_LIFT = 0.001
            poly_xy = np.asarray(pts[:, :2], dtype=np.float64)
            if pts.shape[1] >= 3:
                poly_z = np.asarray(pts[:, 2], dtype=np.float32)
                solid_verts = []
                for p in tri_pts_2d:
                    d2 = np.sum((poly_xy - np.array([p[0], p[1]], dtype=np.float64)) ** 2, axis=1)
                    z = float(poly_z[np.argmin(d2)]) + ROAD_LIFT
                    solid_verts.append([float(p[0]), float(p[1]), z])
                solid_verts = np.array(solid_verts, dtype=np.float32)
            else:
                solid_verts = np.array(
                    [[float(p[0]), float(p[1]), ROAD_LIFT] for p in tri_pts_2d], dtype=np.float32
                )

            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, solid_verts.nbytes, solid_verts, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            glBindVertexArray(0)
            self._poly_solid_geom.append((int(vao), int(vbo), len(solid_verts)))

            # Outline loop: preserve polygon z when available, plus small lift
            outline_verts = np.asarray(poly, dtype=np.float32).copy()
            if outline_verts.shape[1] >= 3:
                outline_verts[:, 2] = outline_verts[:, 2] + 0.002
            else:
                outline_verts = np.column_stack([
                    outline_verts,
                    np.full(len(outline_verts), 0.002, dtype=np.float32)
                ])
            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, outline_verts.nbytes, outline_verts, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            glBindVertexArray(0)
            self._poly_geom.append((int(vao), int(vbo), len(outline_verts)))

        # ── Building solid + outline geometry ────────────────────────────
        for (vao, vbo, _) in self._bld_solid_geom:
            glDeleteVertexArrays(1, [vao])
            glDeleteBuffers(1, [vbo])
        self._bld_solid_geom = []

        for (vao, vbo, _) in self._bld_geom:
            glDeleteVertexArrays(1, [vao])
            glDeleteBuffers(1, [vbo])
        self._bld_geom = []

        bh = self.building_height
        for bld in data.buildings:
            if bld is None or len(bld) < 3:
                continue
            smooth  = _catmull_rom_open(bld)
            floor_p = np.asarray(smooth, dtype=np.float64)   # (N, 3)
            roof_p  = floor_p.copy()
            roof_p[:, 2] += bh
            n = len(floor_p)

            tris: list = []

            # Floor fan (winding: clockwise when viewed from below)
            for i in range(1, n - 1):
                tris.extend([floor_p[0], floor_p[i + 1], floor_p[i]])

            # Roof fan (counter-clockwise when viewed from above)
            for i in range(1, n - 1):
                tris.extend([roof_p[0], roof_p[i], roof_p[i + 1]])

            # Walls: one quad per edge → two triangles
            for i in range(n):
                j = (i + 1) % n
                tris.extend([floor_p[i], floor_p[j], roof_p[j]])
                tris.extend([floor_p[i], roof_p[j],  roof_p[i]])

            solid_verts = np.array(tris, dtype=np.float32)
            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, solid_verts.nbytes, solid_verts, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            glBindVertexArray(0)
            self._bld_solid_geom.append((int(vao), int(vbo), len(solid_verts)))

            # Roof outline loop for the edge highlight
            roof_closed = np.vstack([roof_p, roof_p[:1]]).astype(np.float32)
            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, roof_closed.nbytes, roof_closed, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            glBindVertexArray(0)
            self._bld_geom.append((int(vao), int(vbo), len(roof_closed)))

        # ── Crosswalk rectangle + stripe geometry ────────────────────────
        for (vao, vbo, _) in self._cw_geom:
            glDeleteVertexArrays(1, [vao])
            glDeleteBuffers(1, [vbo])
        self._cw_geom = []

        for (vao, vbo, _) in self._cw_stripe_geom:
            glDeleteVertexArrays(1, [vao])
            glDeleteBuffers(1, [vbo])
        self._cw_stripe_geom = []

        cw_width = data.crosswalk_width
        for cw in data.crosswalks:
            if cw is None or np.asarray(cw).shape != (2, 3):
                continue
            cw = np.asarray(cw, dtype=np.float32)

            # Outline — snap z to road surface so crosswalk sits on street
            corners = _crosswalk_rect(cw[0], cw[1], cw_width, polygons=data.polygons)
            if corners is not None:
                vao = glGenVertexArrays(1)
                vbo = glGenBuffers(1)
                glBindVertexArray(vao)
                glBindBuffer(GL_ARRAY_BUFFER, vbo)
                glBufferData(GL_ARRAY_BUFFER, corners.nbytes, corners, GL_STATIC_DRAW)
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
                glBindVertexArray(0)
                self._cw_geom.append((int(vao), int(vbo), 4))

            # Zebra stripes — snap z to road surface so crosswalk sits on street
            stripe_verts = _crosswalk_stripe_tris(cw[0], cw[1], cw_width, polygons=data.polygons)
            if stripe_verts is not None:
                vao = glGenVertexArrays(1)
                vbo = glGenBuffers(1)
                glBindVertexArray(vao)
                glBindBuffer(GL_ARRAY_BUFFER, vbo)
                glBufferData(GL_ARRAY_BUFFER, stripe_verts.nbytes, stripe_verts, GL_STATIC_DRAW)
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
                glBindVertexArray(0)
                self._cw_stripe_geom.append((int(vao), int(vbo), len(stripe_verts)))

        # Always restore GL state, even if geometry construction raises
        glUseProgram(prev_prog)
        glBindVertexArray(prev_vao)

    # ------------------------------------------------------------------
    def draw(self, view: np.ndarray, proj: np.ndarray,
             model: np.ndarray = None,
             z_lift: float = 0.07) -> None:
        """
        Draw all HD-map elements.

        *view* and *proj* must already be column-major (i.e. transposed
        compared to row-major NumPy convention — pass ``view.T`` / ``proj.T``
        from the render loop, exactly as you would to PosePathRenderer).
        *model* is an optional column-major 4x4 model matrix (defaults to identity).
        """
        if model is None:
            model = np.identity(4, dtype=np.float32)

        # ── Centerline ribbon (purple) ────────────────────────────────
        self._cl_ribbon.draw(view, proj, color=self.CL_COLOR, model=model)

        # ── Bike-lane ribbons (green) ─────────────────────────────────
        for ribbon in self._bl_ribbons:
            ribbon.draw(view, proj, color=self.BL_COLOR, model=model)

        glUseProgram(self._prog)
        glUniformMatrix4fv(self._loc_model, 1, GL_FALSE, model)
        glUniformMatrix4fv(self._loc_view,  1, GL_FALSE, view)
        glUniformMatrix4fv(self._loc_proj,  1, GL_FALSE, proj)
        # All geometry has baked z values; use z_lift=0 so those values are exact.
        # Crosswalks use a small per-draw override to float above the road fill.
        glUniform1f(self._loc_z_lift, 0.0)
        # Gradient disabled by default (u_z_top=0 → t=1 → flat u_color).
        glUniform1f(self._loc_z_top, 0.0)
        glUniform4f(self._loc_color_bot, 0.0, 0.0, 0.0, 1.0)  # unused when flat
        glEnable(GL_DEPTH_TEST)

        # ── Road fill (dark gray, z=0.001 baked) ─────────────────────────
        if self._poly_solid_geom:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glUniform4f(self._loc_color, *self.POLY_FILL_COLOR)
            for (vao, _, count) in self._poly_solid_geom:
                glBindVertexArray(vao)
                glDrawArrays(GL_TRIANGLES, 0, count)
            glDisable(GL_BLEND)

        # ── Road outline (green, z=0.002 baked) ──────────────────────────
        if self._poly_geom and self.show_polygon_outlines:
            glUniform4f(self._loc_color, *self.POLY_COLOR)
            glLineWidth(2.0)
            for (vao, _, count) in self._poly_geom:
                glBindVertexArray(vao)
                glDrawArrays(GL_LINE_LOOP, 0, count)
            glLineWidth(1.0)

        # ── Sidewalk top face (flat warm-gray at z=sh) ────────────────────
        if self._sw_top_geom:
            glUniform4f(self._loc_color, *self.SIDEWALK_COLOR)
            for (vao, _, count) in self._sw_top_geom:
                glBindVertexArray(vao)
                glDrawArrays(GL_TRIANGLES, 0, count)

        # ── Sidewalk walls (dark→gray gradient over z=0..sh) ─────────────
        if self._sw_wall_geom:
            glUniform1f(self._loc_z_top, self.sidewalk_height)
            glUniform4f(self._loc_color,     *self.SIDEWALK_COLOR)    # top (z=sh)
            glUniform4f(self._loc_color_bot, *self.SIDEWALK_BOT_COLOR) # bottom (z=0)
            for (vao, _, count) in self._sw_wall_geom:
                glBindVertexArray(vao)
                glDrawArrays(GL_TRIANGLES, 0, count)
            glUniform1f(self._loc_z_top, 0.0)  # restore gradient-off for rest

        # ── Building solid fill (cyan-blue semi-transparent GL_TRIANGLES) ──
        if self._bld_solid_geom:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glUniform4f(self._loc_color, *self.BLD_COLOR)
            for (vao, _, count) in self._bld_solid_geom:
                glBindVertexArray(vao)
                glDrawArrays(GL_TRIANGLES, 0, count)
            glDisable(GL_BLEND)

        # ── Building roof outline (cyan-blue opaque GL_LINE_STRIP) ────────
        if self._bld_geom:
            glUniform4f(self._loc_color, *self.BLD_EDGE_COLOR)
            glLineWidth(2.0)
            for (vao, _, count) in self._bld_geom:
                glBindVertexArray(vao)
                glDrawArrays(GL_LINE_STRIP, 0, count)
            glLineWidth(1.0)

        # ── Crosswalk zebra stripes and outlines (z already baked; no extra lift) ──
        if self._cw_stripe_geom or self._cw_geom:
            glUniform1f(self._loc_z_lift, 0.0)

        if self._cw_stripe_geom:
            glUniform4f(self._loc_color, 1.0, 1.0, 1.0, 0.9)
            for (vao, _, count) in self._cw_stripe_geom:
                glBindVertexArray(vao)
                glDrawArrays(GL_TRIANGLES, 0, count)

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

        # ── Bike-lane icons (flat horizontal quads at z=bike_icon_z) ─────────
        if self._bike_icon_count > 0 and self._bike_icon_tex:
            glUseProgram(self._bike_icon_prog)
            glUniformMatrix4fv(self._loc_bi_model, 1, GL_FALSE, model)
            glUniformMatrix4fv(self._loc_bi_view,  1, GL_FALSE, view)
            glUniformMatrix4fv(self._loc_bi_proj,  1, GL_FALSE, proj)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._bike_icon_tex)
            glUniform1i(self._loc_bi_tex, 0)
            glBindVertexArray(self._bike_icon_vao)
            glDrawArrays(GL_TRIANGLES, 0, self._bike_icon_count)
            glBindVertexArray(0)
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_BLEND)
            glUseProgram(0)
