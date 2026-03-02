import math
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


_LINE_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTex;

uniform mat4 view;
uniform mat4 projection;

out vec2 TexCoord;

void main() {
    TexCoord = aTex;
    gl_Position = projection * view * vec4(aPos, 1.0);
}
"""

_LINE_FRAGMENT_SHADER = """
#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

uniform vec4 u_color;

void main() {

    float distance_from_center = abs(TexCoord.x - 0.5) * 2.0;
    float alpha = u_color.a * distance_from_center;

    FragColor = vec4(u_color.rgb, alpha);
}
"""


class PosePathRenderer:
    """
    Draws a ribbon-style line connecting pose positions.
    Gradient: edges solid -> center transparent.
    """

    def __init__(self, width: float = 0.3):
        self.width = float(width)

        self.shader = compileProgram(
            compileShader(_LINE_VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(_LINE_FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )

        glUseProgram(self.shader)
        self._loc_view  = glGetUniformLocation(self.shader, "view")
        self._loc_proj  = glGetUniformLocation(self.shader, "projection")
        self._loc_color = glGetUniformLocation(self.shader, "u_color")

        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)

        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)

        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)

        stride = (3 + 2) * 4

        # Position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                              stride, ctypes.c_void_p(0))

        # TexCoord
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                              stride, ctypes.c_void_p(3 * 4))

        glBindVertexArray(0)

        self._vertex_count = 0

    # ------------------------------------------------------------
    # Geometry builder
    # ------------------------------------------------------------

    def update_from_positions(self, positions: list):
        """
        positions: list of np.array([x,y,z])
        Generates a stable ribbon without triangle strip flipping.
        """

        if len(positions) < 2:
            self._vertex_count = 0
            return

        positions = [np.asarray(p, dtype=np.float32) for p in positions]

        verts = []
        half_w = self.width * 0.5

        prev_right = None

        for i in range(len(positions)):

            p = positions[i]

            # --- Compute smooth tangent (averaged direction) ---
            if i == 0:
                forward = positions[1] - p
            elif i == len(positions) - 1:
                forward = p - positions[i - 1]
            else:
                forward = positions[i + 1] - positions[i - 1]

            norm = np.linalg.norm(forward)
            if norm < 1e-8:
                continue

            forward /= norm

            # --- Compute perpendicular (Z-up assumption) ---
            right = np.array([-forward[1], forward[0], 0.0], dtype=np.float32)
            rnorm = np.linalg.norm(right)
            if rnorm < 1e-8:
                right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                right /= rnorm

            # --- Enforce consistent orientation (CRITICAL FIX) ---
            if prev_right is not None:
                if np.dot(prev_right, right) < 0.0:
                    right = -right

            prev_right = right

            # --- Generate left & right vertices ---
            left_pos  = p - right * half_w
            right_pos = p + right * half_w

            v_coord = i / (len(positions) - 1)

            # left vertex (u = 0)
            verts.extend([*left_pos, 0.0, v_coord])

            # right vertex (u = 1)
            verts.extend([*right_pos, 1.0, v_coord])

        if not verts:
            self._vertex_count = 0
            return

        data = np.array(verts, dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        self._vertex_count = len(data) // 5

    # ------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------

    def draw(self, view: np.ndarray, projection: np.ndarray,
             color=(0.2, 0.8, 1.0, 0.8)):

        if self._vertex_count == 0:
            return

        glUseProgram(self.shader)

        glUniformMatrix4fv(self._loc_view, 1, GL_FALSE, view)
        glUniformMatrix4fv(self._loc_proj, 1, GL_FALSE, projection)

        glUniform4f(self._loc_color, *color)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, self._vertex_count)

        glBindVertexArray(0)
        glDisable(GL_BLEND)


_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 instanceOffset;

uniform mat4 view;
uniform mat4 projection;

flat out int vInstanceID;

void main()
{
    vInstanceID = gl_InstanceID;
    vec3 worldPos = aPos + instanceOffset;
    gl_Position = projection * view * vec4(worldPos, 1.0);
}
"""

_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

uniform vec4 u_color;
uniform int selected_index;

flat in int vInstanceID;

void main()
{
    if (selected_index >= 0 && vInstanceID == selected_index)
        FragColor = vec4(1.0, 0.2, 0.2, 1.0);
    else
        FragColor = u_color;
}
"""

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

class PathSphereMarkerRenderer:
    """
    Renders blue spheres at fixed spacing along a path (e.g., every 2m).
    Builds one static VBO for all marker spheres (one draw call).
    """

    def __init__(self, radius: float = 0.12, stacks: int = 10, slices: int = 14, color: tuple = (0.2, 0.5, 1.0, 0.9), drag_enabled: bool = False):
        self.radius = float(radius)
        self.stacks = int(stacks)
        self.slices = int(slices)
        self._COLOR = color

        # --- Drag support ---
        self._drag_enabled = drag_enabled
        self._dragging = False
        self._drag_plane_z = 0.0
        self._drag_offset = np.zeros(3, dtype=np.float32)

        self.cam_pos = np.zeros(3, dtype=np.float32)
        self.ray_dir = np.zeros(3, dtype=np.float32)

        self.shader = compileProgram(
            compileShader(_VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(_FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )

        glUseProgram(self.shader)
        self._loc_view  = glGetUniformLocation(self.shader, "view")
        self._loc_proj  = glGetUniformLocation(self.shader, "projection")
        self._loc_color = glGetUniformLocation(self.shader, "u_color")
        self._loc_selected = glGetUniformLocation(self.shader, "selected_index")

        # VAO
        self._vao = glGenVertexArrays(1)
        glBindVertexArray(self._vao)

        # Sphere mesh (single)
        sphere_data = _sphere_verts(0.0, 0.0, 0.0, self.radius, self.stacks, self.slices)
        sphere_data = sphere_data.astype(np.float32)
        self._verts_per_sphere = len(sphere_data) // 3

        self._sphere_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._sphere_vbo)
        glBufferData(GL_ARRAY_BUFFER, sphere_data.nbytes, sphere_data, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))

        # Instance buffer (centers)
        self._instance_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)  # <- IMPORTANT

        glBindVertexArray(0)

        self.sphere_height = 0.40
        self._centers = []
        self._instance_count = 0
        self._selected_index = -1

    def build_from_path_positions(self, positions: list, step_m: float = 2.0):

        marker_pts = self.sample_polyline_every(positions, step_m=step_m)

        if not marker_pts:
            self._instance_count = 0
            return

        centers = np.array([
            [p[0], p[1], p[2] + self.sphere_height]
            for p in marker_pts
        ], dtype=np.float32)

        self._centers = centers
        self._instance_count = len(centers)
        self._selected_index = -1

        glBindBuffer(GL_ARRAY_BUFFER, self._instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, centers.nbytes, centers, GL_DYNAMIC_DRAW)

    def build_from_positions_direct(self, positions: list):
        """
        Place one sphere at each position without re-sampling. Use when positions
        are already the desired control points (e.g. per-polygon 2m samples)
        so the mesh stays 1:1 with neighbors and does not join across polygons.
        """
        if not positions:
            self._instance_count = 0
            return

        centers = np.array([
            [np.float32(p[0]), np.float32(p[1]), np.float32(p[2]) + self.sphere_height]
            for p in positions
        ], dtype=np.float32)

        self._centers = centers
        self._instance_count = len(centers)
        self._selected_index = -1

        glBindBuffer(GL_ARRAY_BUFFER, self._instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, centers.nbytes, centers, GL_DYNAMIC_DRAW)

    def draw(self, view: np.ndarray, projection: np.ndarray):
        if self._instance_count == 0:
            return

        glUseProgram(self.shader)

        glUniformMatrix4fv(self._loc_view, 1, GL_FALSE, view)
        glUniformMatrix4fv(self._loc_proj, 1, GL_FALSE, projection)

        glUniform4f(self._loc_color, *self._COLOR)
        glUniform1i(self._loc_selected, int(self._selected_index))

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindVertexArray(self._vao)

        glDrawArraysInstanced(
            GL_TRIANGLES,
            0,
            self._verts_per_sphere,
            self._instance_count
        )

        glBindVertexArray(0)
        glDisable(GL_BLEND)

    def sample_polyline_every(self,points: list, step_m: float = 2.0) -> list:
        """
        points: list of np.array([x,y,z]) in meters
        returns: list of interpolated positions every step_m along the polyline
        """
        if len(points) < 2:
            return []

        pts = [np.asarray(p, dtype=np.float32) for p in points]

        out = [pts[0].copy()]
        accum = 0.0
        next_d = step_m

        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-8:
                continue

            # Walk along this segment, dropping markers whenever we cross next_d
            while accum + seg_len >= next_d:
                t = (next_d - accum) / seg_len  # 0..1
                out.append(p0 + seg * t)
                next_d += step_m

            accum += seg_len

        return out

    def intersect_ray(self, ray_origin, ray_dir):
        """
        Returns (index, t) of the closest sphere hit by ray, or None.
        """
        closest = None
        min_t = float("inf")

        for i, center in enumerate(self._centers):
            oc = ray_origin - center
            r = self.radius

            a = np.dot(ray_dir, ray_dir)
            b = 2.0 * np.dot(oc, ray_dir)
            c = np.dot(oc, oc) - r * r

            disc = b*b - 4*a*c
            if disc < 0:
                continue

            sqrt_disc = math.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2*a)
            t2 = (-b + sqrt_disc) / (2*a)

            t = None
            if t1 > 0:
                t = t1
            elif t2 > 0:
                t = t2

            if t is not None and t < min_t:
                min_t = t
                closest = i

        if closest is None:
            return None
        return closest, min_t

    def select(self, index):
        self._selected_index = index

    def begin_drag(self, ray_origin, ray_dir):
        if self._selected_index < 0 or self._selected_index >= self._instance_count:
            return False

        if not self._drag_enabled:
            return False

        center = self._centers[self._selected_index]

        # Drag plane is XY plane at sphere height
        self._drag_plane_z = center[2]

        denom = ray_dir[2]
        if abs(denom) < 1e-6:
            return False

        t = (self._drag_plane_z - ray_origin[2]) / denom
        if t < 0:
            return False

        hit_point = ray_origin + t * ray_dir

        # Keep offset so dragging feels natural
        self._drag_offset = center - hit_point
        self._dragging = True

        return True

    def drag(self, ray_origin, ray_dir):
        if not self._dragging:
            return

        denom = ray_dir[2]
        if abs(denom) < 1e-6:
            return

        t = (self._drag_plane_z - ray_origin[2]) / denom
        if t < 0:
            return

        hit_point = ray_origin + t * ray_dir

        new_center = hit_point + self._drag_offset

        # Constrain Z
        new_center[2] = self._drag_plane_z

        # Update center
        self._centers[self._selected_index] = new_center

        # Update GPU instance buffer
        glBindBuffer(GL_ARRAY_BUFFER, self._instance_vbo)
        glBufferSubData(
            GL_ARRAY_BUFFER,
            self._selected_index * 3 * 4,
            3 * 4,
            new_center.astype(np.float32)
        )

    def end_drag(self):
        self._dragging = False