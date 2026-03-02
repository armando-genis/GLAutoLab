import math
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PathRendererModule import PosePathRenderer, PathSphereMarkerRenderer


_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
}
"""

_FRAGMENT_SHADER = """
#version 330 core
uniform vec4 u_color;
out vec4 FragColor;
void main() {
    FragColor = u_color;
}
"""


def _norm(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > eps else v * 0.0


# ------------------------------------------------------------------
# CPU-side geometry builders (no GL calls)
# ------------------------------------------------------------------

def _sphere_verts(cx, cy, cz, radius, stacks, slices):
    r = radius
    verts = []
    for si in range(stacks):
        for sli in range(slices):
            def _pt(st, sl):
                phi   = math.pi * st / stacks
                theta = 2.0 * math.pi * sl / slices
                return (cx + r * math.sin(phi) * math.cos(theta),
                        cy + r * math.sin(phi) * math.sin(theta),
                        cz + r * math.cos(phi))
            p00 = _pt(si,     sli)
            p10 = _pt(si + 1, sli)
            p01 = _pt(si,     sli + 1)
            p11 = _pt(si + 1, sli + 1)
            verts += [*p00, *p10, *p11, *p00, *p11, *p01]
    return np.array(verts, dtype=np.float32)


def _arrow_verts(cx, cy, cz, heading, roll, pitch, arrow_len):
    z = cz + 0.01
    cy_ = math.cos(heading); sy_ = math.sin(heading)
    cp  = math.cos(pitch);   sp  = math.sin(pitch)
    cr  = math.cos(roll);    sr  = math.sin(roll)

    forward = _norm(np.array([cy_ * cp, sy_ * cp, -sp], dtype=np.float32))
    side    = _norm(np.array([cy_ * sp * sr - sy_ * cr,
                              sy_ * sp * sr + cy_ * cr,
                              cp * sr], dtype=np.float32))

    tip       = np.array([cx, cy, z]) + forward * arrow_len
    n         = side * (arrow_len * 0.04)
    base      = np.array([cx, cy, z], dtype=np.float32)
    p0 = base - n; p1 = base + n
    p2 = tip  + n; p3 = tip  - n

    head_size = arrow_len * 0.25
    left  = tip - forward * head_size + side * head_size * 0.55
    right = tip - forward * head_size - side * head_size * 0.55

    return np.array([
        *p0, *p1, *p2, *p0, *p2, *p3,
        *tip, *left, *right,
    ], dtype=np.float32)


class PoseRenderer:
    """
    Draws ego-vehicle pose as a filled sphere + heading arrow.
    Current pose  → green sphere / yellow arrow  (dynamic VBOs, updated on select)
    All other poses → purple (single static VBO, uploaded once via build_all_poses)
    """

    _SPHERE_COLOR = (0.2, 1.0, 0.4, 1.0)
    _ARROW_COLOR  = (1.0, 0.9, 0.1, 1.0)
    _OTHER_COLOR  = (0.6, 0.0, 0.9, 0.75)

    def __init__(self, radius: float = 0.25, stacks: int = 12, slices: int = 16,
                 arrow_len: float = 0.8):
        self.radius    = float(radius)
        self.stacks    = int(stacks)
        self.slices    = int(slices)
        self.arrow_len = float(arrow_len)

        self.shader = compileProgram(
            compileShader(_VERTEX_SHADER,  GL_VERTEX_SHADER),
            compileShader(_FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )
        glUseProgram(self.shader)
        self._loc_view  = glGetUniformLocation(self.shader, "view")
        self._loc_proj  = glGetUniformLocation(self.shader, "projection")
        self._loc_color = glGetUniformLocation(self.shader, "u_color")

        # --- current-pose VAOs (dynamic) ---
        sphere_bytes = stacks * slices * 6 * 3 * 4
        self._sphere_vao = glGenVertexArrays(1)
        self._sphere_vbo = glGenBuffers(1)
        glBindVertexArray(self._sphere_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._sphere_vbo)
        glBufferData(GL_ARRAY_BUFFER, sphere_bytes, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glBindVertexArray(0)

        self._arrow_vao = glGenVertexArrays(1)
        self._arrow_vbo = glGenBuffers(1)
        glBindVertexArray(self._arrow_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._arrow_vbo)
        glBufferData(GL_ARRAY_BUFFER, 9 * 3 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glBindVertexArray(0)

        self._sphere_count = 0

        # --- all-poses VAO (static, built once) ---
        self._all_vao   = glGenVertexArrays(1)
        self._all_vbo   = glGenBuffers(1)
        self._all_count = 0
        

    # ------------------------------------------------------------------
    # Current pose
    # ------------------------------------------------------------------

    def update(self, location: np.ndarray, heading: float,
               roll: float = 0.0, pitch: float = 0.0):
        loc = np.asarray(location, dtype=np.float32)
        sv = _sphere_verts(loc[0], loc[1], loc[2],
                           self.radius, self.stacks, self.slices)
        glBindBuffer(GL_ARRAY_BUFFER, self._sphere_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, sv.nbytes, sv)
        self._sphere_count = len(sv) // 3

        av = _arrow_verts(loc[0], loc[1], loc[2],
                          heading, roll, pitch, self.arrow_len)
        glBindBuffer(GL_ARRAY_BUFFER, self._arrow_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, av.nbytes, av)

    # ------------------------------------------------------------------
    # Batch build (called once with every pose from the dataset)
    # ------------------------------------------------------------------

    def build_all_poses(self, poses: list):
        """Pre-compute all sphere+arrow vertices into one static VBO."""
        if not poses:
            self._all_count = 0
            return

        parts = []
        for location, heading, roll, pitch in poses:
            loc = np.asarray(location, dtype=np.float32)
            parts.append(_sphere_verts(loc[0], loc[1], loc[2],
                                       self.radius, self.stacks, self.slices))
            parts.append(_arrow_verts(loc[0], loc[1], loc[2],
                                      heading, roll, pitch, self.arrow_len))

        data = np.concatenate(parts)
        self._all_count = len(data) // 3

        glBindVertexArray(self._all_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._all_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glBindVertexArray(0)

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def _begin(self, view, projection):
        self._prev_program = glGetIntegerv(GL_CURRENT_PROGRAM)
        self._prev_vao     = glGetIntegerv(GL_VERTEX_ARRAY_BINDING)
        glUseProgram(self.shader)
        glUniformMatrix4fv(self._loc_view, 1, GL_FALSE, view)
        glUniformMatrix4fv(self._loc_proj, 1, GL_FALSE, projection)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-1.0, -1.0)

    def _end(self):
        glDisable(GL_POLYGON_OFFSET_FILL)
        glDisable(GL_BLEND)
        glBindVertexArray(self._prev_vao)
        glUseProgram(self._prev_program)

    def draw_all(self, view: np.ndarray, projection: np.ndarray):
        """Draw the pre-built batch of all poses in purple (one draw call)."""
        if self._all_count == 0:
            return
        self._begin(view, projection)
        r, g, b, a = self._OTHER_COLOR
        glUniform4f(self._loc_color, r, g, b, a)
        glBindVertexArray(self._all_vao)
        glDrawArrays(GL_TRIANGLES, 0, self._all_count)
        self._end()

    def draw(self, view: np.ndarray, projection: np.ndarray):
        """Draw the current pose (green sphere + yellow arrow)."""
        self._begin(view, projection)

        r, g, b, a = self._SPHERE_COLOR
        glUniform4f(self._loc_color, r, g, b, a)
        glBindVertexArray(self._sphere_vao)
        glDrawArrays(GL_TRIANGLES, 0, self._sphere_count)

        r, g, b, a = self._ARROW_COLOR
        glUniform4f(self._loc_color, r, g, b, a)
        glBindVertexArray(self._arrow_vao)
        glDrawArrays(GL_TRIANGLES, 0, 9)

        self._end()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PoseManager:
    """
    Manages the current ego-vehicle pose and renders it in the 3D view.
    Call load_all_poses(dataset) once at startup to pre-build the purple
    markers for every pose in the dataset (single static VBO).
    """

    def __init__(self):
        self._renderer = PoseRenderer()
        self._has_pose = False
        self._path_renderer = PosePathRenderer(width=1.5)
        self._marker_renderer = PathSphereMarkerRenderer()
        self._path_positions = []

    def load_all_poses(self, dataset):
        """Collect every pose from the dataset and upload to one static VBO."""
        poses = []
        for idx in dataset.indices():
            result = dataset.load_pose(idx)
            if result is not None:
                location, heading, roll, pitch = result
                poses.append((np.asarray(location, dtype=np.float32),
                              heading, roll, pitch))
        self._renderer.build_all_poses(poses)

        z_offset = float(getattr(dataset, "world_offset_height", 0.0))
        positions = []
        for idx in dataset.indices():
            result = dataset.load_pose(idx)
            if result:
                location, _, _, _ = result
                p = np.asarray(location, dtype=np.float32)
                p[2] += z_offset
                positions.append(p)

        self._path_positions = positions
        self._path_renderer.update_from_positions(positions)
        self._marker_renderer.build_from_path_positions(positions)

    @property
    def path_positions(self):
        return self._path_positions

    @property
    def path_width(self):
        return self._path_renderer.width

    def update(self, location: np.ndarray, heading: float,
               roll: float = 0.0, pitch: float = 0.0):
        self._renderer.update(location, heading, roll, pitch)
        self._has_pose = True

    def draw(self, view: np.ndarray, projection: np.ndarray, show_pose: bool = True):
        if show_pose:
            self._renderer.draw_all(view, projection)
            self._renderer.draw(view, projection)
        if self._has_pose:
            self._path_renderer.draw(view, projection)
            self._marker_renderer.draw(view, projection)