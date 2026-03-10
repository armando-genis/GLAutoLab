import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from pathlib import Path
from PIL import Image
import ctypes
import cv2

# Basic transforms
def translate(x, y, z):
    M = np.identity(4, dtype=np.float32)
    M[0, 3] = x
    M[1, 3] = y
    M[2, 3] = z
    return M

def scale(sx, sy, sz):
    M = np.identity(4, dtype=np.float32)
    M[0, 0] = sx
    M[1, 1] = sy
    M[2, 2] = sz
    return M

# shaders for icons
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTex;

uniform mat4 view;
uniform mat4 projection;

out vec2 TexCoord;

void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
    TexCoord = aTex;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D u_texture;
uniform vec3  u_color;
uniform float u_alpha;
uniform int   u_useTexture;

void main() {
    if (u_useTexture == 1) {
        vec4 texColor = texture(u_texture, TexCoord);
        FragColor = vec4(texColor.rgb, texColor.a * u_alpha);
    } else {
        FragColor = vec4(u_color, u_alpha);
    }
}
"""


def _normalize(v: np.ndarray, eps=1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


class IconRenderer:
    """
    Draws:
      1) colored disc (triangle fan)
      2) textured quad (2 triangles)

    Anchoring:
      - uses label.model_matrix() to compute the true top-center in world coords
      - adds fixed z offset in meters (e.g. +0.5)
    """

    def __init__(self, icon_size=0.6, disc_segments=32, z_above_top=0.5, z_fight_eps=0.02):
        self.icon_size = float(icon_size)
        self.disc_segments = int(disc_segments)
        self.z_above_top = float(z_above_top)
        self.z_fight_eps = float(z_fight_eps)

        self.textures = {}

        self.shader = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )

        self.disc_depth_offset = 0.01
        self.quad_depth_offset = 0.03

        # Cache uniform locations (speed)
        glUseProgram(self.shader)
        self.loc_view = glGetUniformLocation(self.shader, "view")
        self.loc_proj = glGetUniformLocation(self.shader, "projection")
        self.loc_tex  = glGetUniformLocation(self.shader, "u_texture")
        self.loc_col  = glGetUniformLocation(self.shader, "u_color")
        self.loc_a    = glGetUniformLocation(self.shader, "u_alpha")
        self.loc_useT = glGetUniformLocation(self.shader, "u_useTexture")

        # VAO/VBO for quad (6 verts * (3+2))
        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)
        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, 6 * 5 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(12))
        glBindVertexArray(0)

        # VAO/VBO for disc (fan: center + (segments+1) edge verts)
        self.disc_vao = glGenVertexArrays(1)
        self.disc_vbo = glGenBuffers(1)
        max_verts = 1 + (self.disc_segments + 1)
        glBindVertexArray(self.disc_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.disc_vbo)
        glBufferData(GL_ARRAY_BUFFER, max_verts * 5 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(12))
        glBindVertexArray(0)

        # VAO/VBO for arrow
        self.arrow_vao = glGenVertexArrays(1)
        self.arrow_vbo = glGenBuffers(1)
        glBindVertexArray(self.arrow_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.arrow_vbo)
        glBufferData(GL_ARRAY_BUFFER, 9 * 5 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(12))
        glBindVertexArray(0)

        # Pre-computed unit circle for vectorized disc generation
        angles = np.linspace(0, 2.0 * math.pi, self.disc_segments + 1, dtype=np.float32)
        self._circle_cos = np.cos(angles)
        self._circle_sin = np.sin(angles)
        self._z_eps_vec = np.array([0.0, 0.0, self.z_fight_eps], dtype=np.float32)

    # ----------------------------
    # Texture upload
    # ----------------------------

    def load_icons(self, pil_icons: dict):
        for name, img in pil_icons.items():
            # Force RGBA for predictable alpha
            img = img.convert("RGBA")
            data = np.asarray(img, dtype=np.uint8)

            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, data)
            glBindTexture(GL_TEXTURE_2D, 0)

            self.textures[name] = tex

    # ----------------------------
    # Anchor computation
    # ----------------------------

    def _world_top_center(self, label) -> np.ndarray:
        M = label.model_matrix()

        # compute two local points:
        bottom = np.array([0.5, 0.5, 0.0, 1.0], dtype=np.float32)
        top    = np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float32)

        world_top = (M @ top)[:3]
        world_bottom = (M @ bottom)[:3]

        # if top is below bottom, flip assumption
        if world_top[2] < world_bottom[2]:
            local = np.array([0.0, 0.0, 0.5, 1.0], dtype=np.float32)
            return (M @ local)[:3]

        return world_top


    def _icon_anchor(self, label) -> np.ndarray:
        cx, cy, cz = label.center
        _, _, lz = label.size

        return np.array([
            cx,
            cy,
            cz + lz * 0.5 + 0.5
        ], dtype=np.float32)


    def _basis_from_yaw(self, yaw: float):
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        forward = _normalize(forward)
        right = np.array([-forward[1], forward[0], 0.0], dtype=np.float32)
        right = _normalize(right)
        if float(np.linalg.norm(right)) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return right, up

    # ----------------------------
    # Lidar-frame -> world helpers
    # ----------------------------

    @staticmethod
    def _apply_model_point(p: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Transform a 3D point by a 4x4 matrix."""
        ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float32)
        return (M @ ph)[:3]

    @staticmethod
    def _apply_model_dir(d: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Rotate a direction vector by a 4x4 matrix (no translation)."""
        dh = np.array([d[0], d[1], d[2], 0.0], dtype=np.float32)
        return (M @ dh)[:3]

    # ----------------------------
    # Public draw
    # ----------------------------
    def draw(self, label, view, projection, color, lidar_model=None):
        if label.label_type not in self.textures:
            return

        if lidar_model is None:
            lidar_model = np.identity(4, dtype=np.float32)

        prev_program = glGetIntegerv(GL_CURRENT_PROGRAM)
        prev_vao = glGetIntegerv(GL_VERTEX_ARRAY_BINDING)

        glUseProgram(self.shader)

        glUniformMatrix4fv(self.loc_view, 1, GL_FALSE, view)
        glUniformMatrix4fv(self.loc_proj, 1, GL_FALSE, projection)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # IMPORTANT: do NOT disable depth writing
        glDepthMask(GL_TRUE)

        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)

        # Disc
        self._draw_disc(label, lidar_model)

        glDisable(GL_POLYGON_OFFSET_FILL)

        # Quad
        self._draw_quad(label, lidar_model)

        # Arrow
        self.draw_yaw_arrow(label, color, lidar_model)

        glDisable(GL_BLEND)

        glBindVertexArray(prev_vao)
        glUseProgram(prev_program)

    # ----------------------------
    # Disc
    # ----------------------------
    def _draw_disc(self, label, lidar_model):
        anchor_local = self._icon_anchor(label) + np.array([0, 0, self.z_fight_eps], dtype=np.float32)
        anchor = self._apply_model_point(anchor_local, lidar_model)

        # icon plane basis — rotate lidar-frame vectors into world space
        right_l, up_l = self._basis_from_yaw(label.yaw)
        right = _normalize(self._apply_model_dir(right_l, lidar_model))
        up    = _normalize(self._apply_model_dir(up_l,    lidar_model))

        radius = self.icon_size * 0.65

        # fan verts: (pos3 + tex2)
        verts = []

        # center
        verts.extend([anchor[0], anchor[1], anchor[2], 0.5, 0.5])

        for i in range(self.disc_segments + 1):
            a = 2.0 * math.pi * i / self.disc_segments
            p = anchor + right * (radius * math.cos(a)) + up * (radius * math.sin(a))
            # tex coords not used in solid mode
            verts.extend([p[0], p[1], p[2], 0.5, 0.5])

        verts = np.array(verts, dtype=np.float32)

        # uniforms (solid color)
        r, g, b = label.color()
        glUniform1i(self.loc_useT, 0)
        glUniform3f(self.loc_col, float(r), float(g), float(b))
        glUniform1f(self.loc_a, 1.0)

        glBindVertexArray(self.disc_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.disc_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)

        glDrawArrays(GL_TRIANGLE_FAN, 0, len(verts) // 5)

        glBindVertexArray(0)

    # ----------------------------
    # Textured quad
    # ----------------------------

    def _draw_quad(self, label, lidar_model):
        tex_id = self.textures[label.label_type]

        center = self._apply_model_point(self._icon_anchor(label), lidar_model)
        right_l, up_l = self._basis_from_yaw(label.yaw)
        right = _normalize(self._apply_model_dir(right_l, lidar_model))
        up    = _normalize(self._apply_model_dir(up_l,    lidar_model))

        half_w = self.icon_size * 0.5
        height = self.icon_size * 1.2
        half_h = height * 0.5

        bl = center - right * half_w - up * half_h
        br = center + right * half_w - up * half_h
        tr = center + right * half_w + up * half_h
        tl = center - right * half_w + up * half_h

        verts = np.array([
            bl[0], bl[1], bl[2], 0.0, 1.0,
            br[0], br[1], br[2], 1.0, 1.0,
            tr[0], tr[1], tr[2], 1.0, 0.0,

            bl[0], bl[1], bl[2], 0.0, 1.0,
            tr[0], tr[1], tr[2], 1.0, 0.0,
            tl[0], tl[1], tl[2], 0.0, 0.0,
        ], dtype=np.float32)

        glUniform1i(self.loc_useT, 1)
        glUniform1f(self.loc_a, 1.0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glUniform1i(self.loc_tex, 0)

        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)

        glDrawArrays(GL_TRIANGLES, 0, 6)

        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)


    def world_bottom_center(self, label) -> np.ndarray:
        # bottom center in WORLD should be center XY and z min
        cx, cy, cz = label.center
        _, _, lz = label.size
        return np.array([cx, cy, cz - 0.5 * lz], dtype=np.float32)

    def draw_yaw_arrow(self, label, color, lidar_model):
        base_local = self.world_bottom_center(label).copy()
        base_local[2] += self.z_fight_eps
        base = self._apply_model_point(base_local, lidar_model)

        # forward from yaw in lidar frame, rotated to world space
        forward_l = np.array([math.cos(label.yaw), math.sin(label.yaw), 0.0], dtype=np.float32)
        forward = _normalize(self._apply_model_dir(forward_l, lidar_model))

        arrow_len = float(label.size[0]) * 0.8
        tip = base + forward * arrow_len

        # arrow head
        side_l = np.array([-forward_l[1], forward_l[0], 0.0], dtype=np.float32)
        side = _normalize(self._apply_model_dir(side_l, lidar_model))
        head = arrow_len * 0.25

        left  = tip - forward * head + side * head * 0.5
        right = tip - forward * head - side * head * 0.5

        # We will render using the same shader by drawing solid triangles/lines
        # with u_useTexture=0
        glUniform1i(self.loc_useT, 0)
        glUniform3f(self.loc_col, float(color[0]), float(color[1]), float(color[2]))
        glUniform1f(self.loc_a, 1.0)

        # --- Draw shaft as 2-vertex "thin triangle" (instead of GL_LINES) ---
        # Core profile friendly and consistent: make a tiny quad strip
        thickness = arrow_len * 0.03
        n = np.array([-forward[1], forward[0], 0.0], dtype=np.float32)  # perpendicular
        n = _normalize(n) * thickness

        p0 = base - n
        p1 = base + n
        p2 = tip + n
        p3 = tip - n

        # 2 triangles for shaft + 1 triangle for head
        verts = np.array([
            # shaft tri 1
            p0[0], p0[1], p0[2], 0.0, 0.0,
            p1[0], p1[1], p1[2], 0.0, 0.0,
            p2[0], p2[1], p2[2], 0.0, 0.0,
            # shaft tri 2
            p0[0], p0[1], p0[2], 0.0, 0.0,
            p2[0], p2[1], p2[2], 0.0, 0.0,
            p3[0], p3[1], p3[2], 0.0, 0.0,
            # head
            tip[0], tip[1], tip[2], 0.0, 0.0,
            left[0], left[1], left[2], 0.0, 0.0,
            right[0], right[1], right[2], 0.0, 0.0,
        ], dtype=np.float32)

        # reuse quad VAO/VBO (same format pos3+tex2)
        glBindVertexArray(self.arrow_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.arrow_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
        glDrawArrays(GL_TRIANGLES, 0, len(verts) // 5)
        glBindVertexArray(0)

    # ----------------------------
    # CPU-only geometry builders (for batching)
    # ----------------------------

    def _build_disc_triangles(self, label, lidar_model):
        """Build disc as GL_TRIANGLES (not fan) so multiple discs can be batched."""
        anchor = self._apply_model_point(
            self._icon_anchor(label) + self._z_eps_vec, lidar_model)

        right_l, up_l = self._basis_from_yaw(label.yaw)
        right = _normalize(self._apply_model_dir(right_l, lidar_model))
        up    = _normalize(self._apply_model_dir(up_l,    lidar_model))

        radius = self.icon_size * 0.65

        # Vectorised edge ring: (segments+1, 3)
        edge = (anchor
                + np.outer(self._circle_cos * radius, right)
                + np.outer(self._circle_sin * radius, up))

        n = self.disc_segments
        tris = np.empty((n, 3, 5), dtype=np.float32)
        tris[:, 0, :3] = anchor
        tris[:, 0, 3:] = 0.5
        tris[:, 1, :3] = edge[:n]
        tris[:, 1, 3:] = 0.5
        tris[:, 2, :3] = edge[1:n + 1]
        tris[:, 2, 3:] = 0.5
        return tris.ravel()

    def _build_quad_verts(self, label, lidar_model):
        """Build textured quad (6 verts, 2 triangles)."""
        center = self._apply_model_point(self._icon_anchor(label), lidar_model)
        right_l, up_l = self._basis_from_yaw(label.yaw)
        right = _normalize(self._apply_model_dir(right_l, lidar_model))
        up    = _normalize(self._apply_model_dir(up_l,    lidar_model))

        hw = self.icon_size * 0.5
        hh = self.icon_size * 1.2 * 0.5

        bl = center - right * hw - up * hh
        br = center + right * hw - up * hh
        tr = center + right * hw + up * hh
        tl = center - right * hw + up * hh

        return np.array([
            bl[0], bl[1], bl[2], 0.0, 1.0,
            br[0], br[1], br[2], 1.0, 1.0,
            tr[0], tr[1], tr[2], 1.0, 0.0,
            bl[0], bl[1], bl[2], 0.0, 1.0,
            tr[0], tr[1], tr[2], 1.0, 0.0,
            tl[0], tl[1], tl[2], 0.0, 0.0,
        ], dtype=np.float32)

    def _build_arrow_verts(self, label, lidar_model):
        """Build yaw arrow (9 verts, 3 triangles)."""
        base_local = self.world_bottom_center(label).copy()
        base_local[2] += self.z_fight_eps
        base = self._apply_model_point(base_local, lidar_model)

        forward_l = np.array([math.cos(label.yaw), math.sin(label.yaw), 0.0],
                             dtype=np.float32)
        forward = _normalize(self._apply_model_dir(forward_l, lidar_model))

        arrow_len = float(label.size[0]) * 0.8
        tip = base + forward * arrow_len

        side_l = np.array([-forward_l[1], forward_l[0], 0.0], dtype=np.float32)
        side = _normalize(self._apply_model_dir(side_l, lidar_model))
        head_sz = arrow_len * 0.25

        left_pt  = tip - forward * head_sz + side * head_sz * 0.5
        right_pt = tip - forward * head_sz - side * head_sz * 0.5

        thickness = arrow_len * 0.03
        n = _normalize(np.array([-forward[1], forward[0], 0.0],
                                dtype=np.float32)) * thickness

        p0, p1, p2, p3 = base - n, base + n, tip + n, tip - n

        return np.array([
            p0[0], p0[1], p0[2], 0.0, 0.0,
            p1[0], p1[1], p1[2], 0.0, 0.0,
            p2[0], p2[1], p2[2], 0.0, 0.0,
            p0[0], p0[1], p0[2], 0.0, 0.0,
            p2[0], p2[1], p2[2], 0.0, 0.0,
            p3[0], p3[1], p3[2], 0.0, 0.0,
            tip[0], tip[1], tip[2], 0.0, 0.0,
            left_pt[0], left_pt[1], left_pt[2], 0.0, 0.0,
            right_pt[0], right_pt[1], right_pt[2], 0.0, 0.0,
        ], dtype=np.float32)

    # ----------------------------
    # Batched draw (all labels, minimal GL calls)
    # ----------------------------

    def draw_all(self, labels, label_colors, view, projection, lidar_model=None):
        if not labels:
            return
        if lidar_model is None:
            lidar_model = np.identity(4, dtype=np.float32)

        prev_program = glGetIntegerv(GL_CURRENT_PROGRAM)
        prev_vao = glGetIntegerv(GL_VERTEX_ARRAY_BINDING)

        glUseProgram(self.shader)
        glUniformMatrix4fv(self.loc_view, 1, GL_FALSE, view)
        glUniformMatrix4fv(self.loc_proj, 1, GL_FALSE, projection)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_TRUE)

        # ---- Build all geometry on CPU, grouped by color / texture ----
        disc_by_color  = {}
        quad_by_tex    = {}
        arrow_by_color = {}

        for label, color in zip(labels, label_colors):
            if label.label_type not in self.textures:
                continue

            dc = label.color()
            disc_by_color.setdefault(dc, []).append(
                self._build_disc_triangles(label, lidar_model))

            tid = self.textures[label.label_type]
            quad_by_tex.setdefault(tid, []).append(
                self._build_quad_verts(label, lidar_model))

            ac = (float(color[0]), float(color[1]), float(color[2]))
            arrow_by_color.setdefault(ac, []).append(
                self._build_arrow_verts(label, lidar_model))

        # ---- Discs (one draw call per colour) ----
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        glUniform1i(self.loc_useT, 0)
        glUniform1f(self.loc_a, 1.0)
        glBindVertexArray(self.disc_vao)

        for (r, g, b), parts in disc_by_color.items():
            batch = np.concatenate(parts)
            glUniform3f(self.loc_col, float(r), float(g), float(b))
            glBindBuffer(GL_ARRAY_BUFFER, self.disc_vbo)
            glBufferData(GL_ARRAY_BUFFER, batch.nbytes, batch, GL_STREAM_DRAW)
            glDrawArrays(GL_TRIANGLES, 0, len(batch) // 5)

        glDisable(GL_POLYGON_OFFSET_FILL)

        # ---- Quads (one draw call per texture) ----
        glUniform1i(self.loc_useT, 1)
        glUniform1f(self.loc_a, 1.0)
        glBindVertexArray(self.quad_vao)

        for tid, parts in quad_by_tex.items():
            batch = np.concatenate(parts)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, tid)
            glUniform1i(self.loc_tex, 0)
            glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
            glBufferData(GL_ARRAY_BUFFER, batch.nbytes, batch, GL_STREAM_DRAW)
            glDrawArrays(GL_TRIANGLES, 0, len(batch) // 5)

        glBindTexture(GL_TEXTURE_2D, 0)

        # ---- Arrows (one draw call per colour) ----
        glUniform1i(self.loc_useT, 0)
        glUniform1f(self.loc_a, 1.0)
        glBindVertexArray(self.arrow_vao)

        for (r, g, b), parts in arrow_by_color.items():
            batch = np.concatenate(parts)
            glUniform3f(self.loc_col, float(r), float(g), float(b))
            glBindBuffer(GL_ARRAY_BUFFER, self.arrow_vbo)
            glBufferData(GL_ARRAY_BUFFER, batch.nbytes, batch, GL_STREAM_DRAW)
            glDrawArrays(GL_TRIANGLES, 0, len(batch) // 5)

        # ---- Cleanup ----
        glDisable(GL_BLEND)
        glBindVertexArray(prev_vao)
        glUseProgram(prev_program)


# Label3D
class Label3D:
    """
    Represents a 3D bounding box in LiDAR frame.
    size = (length, width, height)
    center = (x, y, z)
    yaw in radians
    """

    CLASS_COLORS = {
        "person":  (1.0, 0.2, 0.2),
        "car":     (0.2, 0.6, 1.0),
        "bus":     (1.0, 0.6, 0.2),
        "bicycle": (0.6, 1.0, 0.2),
    }

    def __init__(self, center, size, yaw=0.0, label_type="car"):
        self.center = np.array(center, dtype=np.float32)
        self.size = np.array(size, dtype=np.float32)
        self.yaw = float(yaw)
        self.label_type = label_type

    # Rendering
    def model_matrix(self):
        cx, cy, cz = self.center
        lx, ly, lz = self.size

        T = translate(cx, cy, cz)

        Rz = np.array([
            [math.cos(self.yaw), -math.sin(self.yaw), 0, 0],
            [math.sin(self.yaw),  math.cos(self.yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        S = scale(lx, ly, lz)

        return T @ Rz @ S

    def color(self):
        return self.CLASS_COLORS.get(self.label_type, (1.0, 1.0, 1.0))

    # KITTI Format
    def to_kitti_line(self):
        """
        KITTI format:
        type truncated occluded alpha
        bbox(4 dummy)
        h w l
        x y z
        yaw
        """

        l, w, h = self.size
        x, y, z = self.center
        yaw = self.yaw

        return (
            f"{self.label_type} 0 0 0 0 0 0 0 "
            f"{h:.3f} {w:.3f} {l:.3f} "
            f"{x:.3f} {y:.3f} {z:.3f} {yaw:.3f}"
        )

    def intersect_ray(self, ray_origin, ray_dir):
        M = self.model_matrix()
        inv_M = np.linalg.inv(M)

        # Transform ray to local cube space
        origin_local = inv_M @ np.append(ray_origin, 1.0)
        dir_local = inv_M @ np.append(ray_dir, 0.0)

        origin_local = origin_local[:3]
        dir_local = dir_local[:3]

        # Slab method for AABB [-0.5,0.5]
        tmin = -np.inf
        tmax = np.inf

        for i in range(3):
            if abs(dir_local[i]) < 1e-6:
                if origin_local[i] < -0.5 or origin_local[i] > 0.5:
                    return None
            else:
                t1 = (-0.5 - origin_local[i]) / dir_local[i]
                t2 = (0.5 - origin_local[i]) / dir_local[i]
                t_near = min(t1, t2)
                t_far = max(t1, t2)
                tmin = max(tmin, t_near)
                tmax = min(tmax, t_far)
                if tmin > tmax:
                    return None

        if tmax < 0:
            return None

        return tmin if tmin > 0 else tmax

    @staticmethod
    def from_kitti_line(line):
        parts = line.strip().split()
        if len(parts) < 15:
            return None

        label_type = parts[0]

        h, w, l = map(float, parts[8:11])
        x, y, z = map(float, parts[11:14])
        yaw = float(parts[14])

        size = [l, w, h]
        return Label3D([x, y, z], size, yaw, label_type)


# LabelManager
class LabelManager:

    def __init__(self, save_dir="labels"):
        self.scene_labels = {}        # scene_id -> list[Label3D]
        self.current_scene = None
        self.selected_index = None
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.icons = {}


        self.icon_renderer = IconRenderer()


    # Scene Handling
    def set_scene(self, scene_id):
        """
        Switch active scene.
        Safe for int or string scene ids.
        """

        self.current_scene = scene_id
        self.selected_index = None

        if scene_id not in self.scene_labels:
            self.scene_labels[scene_id] = []

        self.load_kitti(scene_id)

    def labels(self):
        if self.current_scene is None:
            return []
        return self.scene_labels.get(self.current_scene, [])


    # Add / Remove
    def add_label(self, center, size, yaw=0.0, label_type="car"):
        if self.current_scene is None:
            return

        label = Label3D(center, size, yaw, label_type)
        self.scene_labels[self.current_scene].append(label)
        self.selected_index = len(self.scene_labels[self.current_scene]) - 1

    def remove_selected(self):
        if self.current_scene is None:
            return

        if self.selected_index is not None:
            labels = self.scene_labels[self.current_scene]
            if 0 <= self.selected_index < len(labels):
                labels.pop(self.selected_index)

        self.selected_index = None

    # Selection
    def select(self, index):
        if 0 <= index < len(self.labels()):
            self.selected_index = index

    def selected(self):
        if self.selected_index is None:
            return None

        labels = self.labels()
        if 0 <= self.selected_index < len(labels):
            return labels[self.selected_index]

        return None

    # Move Selected (world/lidar frame)
    def move_selected(self, dx, dy, dz):
        label = self.selected()
        if label is not None:
            label.center += np.array([dx, dy, dz], dtype=np.float32)

    # Move Selected in the label's local frame (forward/right relative to yaw)
    def move_selected_local(self, forward, right, up):
        label = self.selected()
        if label is not None:
            fwd = np.array([math.cos(label.yaw), math.sin(label.yaw), 0.0], dtype=np.float32)
            lft = np.array([-math.sin(label.yaw), math.cos(label.yaw), 0.0], dtype=np.float32)
            label.center += forward * fwd + right * lft + np.array([0.0, 0.0, up], dtype=np.float32)

    # Rotate Selected
    def rotate_selected(self, dyaw):
        label = self.selected()
        if label is not None:
            label.yaw += dyaw

    # Drawing
    def draw(self, cube, set_model_color, view_matrix, projection_matrix, lidar_model=None):
        if lidar_model is None:
            lidar_model = np.identity(4, dtype=np.float32)

        labels = self.labels()
        icon_colors = []

        for i, label in enumerate(labels):
            model = lidar_model @ label.model_matrix()
            r, g, b = label.color()

            if i == self.selected_index:
                r, g, b = 1.0, 1.0, 0.0

            icon_colors.append((r, g, b))

            set_model_color(model, r, g, b, 1.0)

            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            cube.draw()
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        self.icon_renderer.draw_all(
            labels, icon_colors, view_matrix, projection_matrix, lidar_model)


    # KITTI IO
    def kitti_path(self, scene_id):
        """
        Supports int or string scene id.
        """

        if isinstance(scene_id, int):
            name = f"{scene_id:06d}.txt"
        else:
            name = f"{str(scene_id)}.txt"

        return self.save_dir / name

    def save_kitti(self):
        if self.current_scene is None:
            return

        path = self.kitti_path(self.current_scene)

        with open(path, "w") as f:
            for label in self.labels():
                f.write(label.to_kitti_line() + "\n")

    def load_kitti(self, scene_id):
        path = self.kitti_path(scene_id)

        self.scene_labels[scene_id] = []

        if not path.exists():
            return

        with open(path, "r") as f:
            for line in f:
                label = Label3D.from_kitti_line(line)
                if label is not None:
                    self.scene_labels[scene_id].append(label)

    def remove_last(self):
        if self.current_scene is None:
            return
        labels = self.scene_labels.get(self.current_scene, [])
        if labels:
            labels.pop()
        self.selected_index = None

    def set_icons(self, icons):
        self.icons = icons
        print(icons)
        self.icon_renderer.load_icons(icons)


# Unit-cube corners in local label space (used by Label3D.model_matrix)
_BOX_CORNERS_LOCAL = np.array([
    [-0.5, -0.5, -0.5, 1.0],
    [ 0.5, -0.5, -0.5, 1.0],
    [ 0.5,  0.5, -0.5, 1.0],
    [-0.5,  0.5, -0.5, 1.0],
    [-0.5, -0.5,  0.5, 1.0],
    [ 0.5, -0.5,  0.5, 1.0],
    [ 0.5,  0.5,  0.5, 1.0],
    [-0.5,  0.5,  0.5, 1.0],
], dtype=np.float64)

_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
]

# +X face (forward / yaw direction): corners 1, 2, 5, 6
_FRONT_EDGES = frozenset([(1, 2), (5, 6), (1, 5), (2, 6)])


class LabelCameraManager:
    def __init__(self):
        self.lidar_camera_extrinsics_array = []
        self.camera_array_intrinsics = []
        self.camera_label_transforms = []

    def load_camera_lidar_parameters(self, dataset):
        self.lidar_camera_extrinsics_array = dataset.load_camera_lidar_extrinsics_array()
        self.camera_array_intrinsics = dataset.load_camera_array_intrinsics()

    def compute_camera_label_transform(self, lidar_frame=None):
        if lidar_frame is None:
            lidar_frame = np.identity(4, dtype=np.float64)

        for ext in self.lidar_camera_extrinsics_array:
            if ext is None:
                continue

            # Inverse of the extrinsic: camera pose expressed in LiDAR frame (robot convention)
            R_cam_in_lidar = ext.R_robot.T                          # (3, 3)
            t_cam_in_lidar = (-ext.R_robot.T @ ext.t_robot).ravel() # (3,)

            cam_pose_lidar = np.identity(4, dtype=np.float64)
            cam_pose_lidar[:3, :3] = R_cam_in_lidar
            cam_pose_lidar[:3,  3] = t_cam_in_lidar

            cam_world = (lidar_frame @ cam_pose_lidar).astype(np.float32)
            self.camera_label_transforms.append(cam_world)

    def draw_labels_on_camera(self, images, labels):
        """
        Project 3D bounding-box labels onto each camera image.

        Parameters
        ----------
        images : list[np.ndarray | None]
            One image per camera slot (indexed by camera id).
        labels : list[Label3D]
            Labels defined in LiDAR frame.

        Returns
        -------
        list[np.ndarray | None]
            Annotated copies of the input images.
        """
        result = []

        for cam_idx, img in enumerate(images):
            if img is None:
                result.append(None)
                continue

            ext = (self.lidar_camera_extrinsics_array[cam_idx]
                   if cam_idx < len(self.lidar_camera_extrinsics_array) else None)
            cam = (self.camera_array_intrinsics[cam_idx]
                   if cam_idx < len(self.camera_array_intrinsics) else None)

            if ext is None or cam is None:
                result.append(img)
                continue

            R = ext.R_opencv.astype(np.float64)
            t = ext.t_opencv.astype(np.float64).ravel()
            K = cam.get_K().astype(np.float64)

            img_out = img.copy()
            h, w = img_out.shape[:2]

            for label in labels:
                self._draw_box_on_image(img_out, label, R, t, K, w, h)

            result.append(img_out)

        return result

    def bake_path_on_images(self, images, positions, lidar_model,
                            path_width=1.5):
        """
        Render the 3D path ribbon onto copies of the camera images.

        Called once per frame change. The returned list is cached and used
        as the base for subsequent (cheap) label re-draws.

        Parameters
        ----------
        images : list[np.ndarray | None]
            Raw camera images (will not be modified).
        positions : list[np.ndarray]
            Path positions in **world frame** (x, y, z).
        lidar_model : np.ndarray (4, 4)
            Current ego-pose matrix (LiDAR body -> world).
        path_width : float
            Ribbon width in meters (should match PosePathRenderer.width).

        Returns
        -------
        list[np.ndarray | None]
            Copies of the input images with the path baked in.
        """
        result = [img.copy() if img is not None else None for img in images]

        if len(positions) < 2:
            return result

        half_w = path_width * 0.5
        world_pts = [np.asarray(p[:3], dtype=np.float64) for p in positions]

        left_world = []
        right_world = []
        prev_right = None

        for i in range(len(world_pts)):
            p = world_pts[i]

            if i == 0:
                fwd = world_pts[1] - p
            elif i == len(world_pts) - 1:
                fwd = p - world_pts[i - 1]
            else:
                fwd = world_pts[i + 1] - world_pts[i - 1]

            n = np.linalg.norm(fwd)
            if n < 1e-8:
                left_world.append(None)
                right_world.append(None)
                continue
            fwd /= n

            right = np.array([-fwd[1], fwd[0], 0.0], dtype=np.float64)
            rn = np.linalg.norm(right)
            if rn < 1e-8:
                right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                right /= rn

            if prev_right is not None and np.dot(prev_right, right) < 0.0:
                right = -right
            prev_right = right

            left_world.append(p - right * half_w)
            right_world.append(p + right * half_w)

        valid = [(i, left_world[i], right_world[i])
                 for i in range(len(left_world))
                 if left_world[i] is not None]
        if len(valid) < 2:
            return result

        _, lefts, rights = zip(*valid)
        left_arr = np.array(lefts)
        right_arr = np.array(rights)

        lidar_inv = np.linalg.inv(lidar_model.astype(np.float64))
        R_body, t_body = lidar_inv[:3, :3], lidar_inv[:3, 3]

        left_lidar = (R_body @ left_arr.T).T + t_body
        right_lidar = (R_body @ right_arr.T).T + t_body

        path_color = (51, 204, 255)
        eps = 0.05

        for cam_idx, img in enumerate(result):
            if img is None:
                continue

            ext = (self.lidar_camera_extrinsics_array[cam_idx]
                   if cam_idx < len(self.lidar_camera_extrinsics_array) else None)
            cam = (self.camera_array_intrinsics[cam_idx]
                   if cam_idx < len(self.camera_array_intrinsics) else None)
            if ext is None or cam is None:
                continue

            R = ext.R_opencv.astype(np.float64)
            t = ext.t_opencv.astype(np.float64).ravel()
            K = cam.get_K().astype(np.float64)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            left_cam = (R @ left_lidar.T).T + t.reshape(1, 3)
            right_cam = (R @ right_lidar.T).T + t.reshape(1, 3)

            visible = (
                (left_cam[:-1, 2] > eps) & (left_cam[1:, 2] > eps) &
                (right_cam[:-1, 2] > eps) & (right_cam[1:, 2] > eps)
            )

            overlay = img.copy()
            drawn = False

            for i in np.where(visible)[0]:
                def _proj(p):
                    return (int(round(fx * p[0] / p[2] + cx)),
                            int(round(fy * p[1] / p[2] + cy)))

                quad = np.array([
                    _proj(left_cam[i]),
                    _proj(right_cam[i]),
                    _proj(right_cam[i + 1]),
                    _proj(left_cam[i + 1]),
                ], dtype=np.int32)
                cv2.fillConvexPoly(overlay, quad, path_color, cv2.LINE_AA)
                drawn = True

            if drawn:
                cv2.addWeighted(overlay, 0.5, img, 0.5, 0, dst=img)

        return result

    @staticmethod
    def _clip_edge_near_plane(p1, p2, eps=0.05):
        """Clip a 3D segment to the half-space Z > eps.
        Returns the clipped (p1, p2) pair or None if fully behind."""
        z1, z2 = p1[2], p2[2]
        if z1 <= eps and z2 <= eps:
            return None
        if z1 > eps and z2 > eps:
            return p1, p2
        alpha = (eps - z1) / (z2 - z1)
        mid = p1 + alpha * (p2 - p1)
        return (mid, p2) if z1 <= eps else (p1, mid)

    @staticmethod
    def _draw_box_on_image(img, label, R, t, K, w, h):
        M = label.model_matrix().astype(np.float64)

        # Local unit-cube corners -> LiDAR frame  (8, 3)
        corners_lidar = (_BOX_CORNERS_LOCAL @ M.T)[:, :3]

        # LiDAR -> Camera frame  (OpenCV: X-right, Y-down, Z-forward)
        corners_cam = (R @ corners_lidar.T).T + t.reshape(1, 3)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        def _project(p3d):
            u = int(round(fx * p3d[0] / p3d[2] + cx))
            v = int(round(fy * p3d[1] / p3d[2] + cy))
            return u, v

        r, g, b = label.color()
        color_rgb = (int(r * 255), int(g * 255), int(b * 255))
        front_rgb = (
            min(int(r * 255) + 80, 255),
            min(int(g * 255) + 80, 255),
            min(int(b * 255) + 80, 255),
        )

        for i, j in _BOX_EDGES:
            clipped = LabelCameraManager._clip_edge_near_plane(
                corners_cam[i], corners_cam[j]
            )
            if clipped is None:
                continue

            p1 = _project(clipped[0])
            p2 = _project(clipped[1])

            is_front = (i, j) in _FRONT_EDGES or (j, i) in _FRONT_EDGES
            line_color = front_rgb if is_front else color_rgb
            thickness = 3 if is_front else 2

            cv2.line(img, p1, p2, line_color, thickness, cv2.LINE_AA)