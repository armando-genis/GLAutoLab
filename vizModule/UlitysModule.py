import ctypes
from typing import List
import numpy as np
import math
from OpenGL.GL import (
    glGenVertexArrays, glGenBuffers, glBindVertexArray, glBindBuffer,
    glBufferData, glBufferSubData, glVertexAttribPointer, glEnableVertexAttribArray,
    glDrawElements, glDrawArrays, glEnable, glDisable, glPolygonOffset,
    glLineWidth, glGenTextures, glBindTexture, glTexParameteri,
    glTexImage2D, glTexSubImage2D, glDeleteTextures,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, GL_DYNAMIC_DRAW,
    GL_FLOAT, GL_FALSE, GL_TRIANGLES, GL_LINES, GL_POINTS, GL_UNSIGNED_INT,
    GL_POLYGON_OFFSET_FILL, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER, GL_LINEAR, GL_RGB, GL_UNSIGNED_BYTE,
)

from typing import List, Tuple, Optional
import cv2

def look_at(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.identity(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    T = np.identity(4, dtype=np.float32)
    T[:3, 3] = -eye
    return M @ T

# Matrix utilities
def perspective(fov_rad, aspect, near, far):
    f = 1.0 / math.tan(fov_rad / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1
    return M


class Cube:
    """Indexed unit cube (centered at origin, side length 1)."""

    def __init__(self):
        vertices = np.array([
            [-0.5, -0.5, -0.5],  # 0
            [-0.5, -0.5,  0.5],  # 1
            [-0.5,  0.5, -0.5],  # 2
            [-0.5,  0.5,  0.5],  # 3
            [ 0.5, -0.5, -0.5],  # 4
            [ 0.5, -0.5,  0.5],  # 5
            [ 0.5,  0.5, -0.5],  # 6
            [ 0.5,  0.5,  0.5],  # 7
        ], dtype=np.float32)

        indices = np.array([
            0, 1, 2,  1, 3, 2,
            1, 5, 3,  3, 5, 7,
            0, 4, 1,  1, 4, 5,
            3, 6, 2,  3, 7, 6,
            0, 2, 4,  2, 6, 4,
            4, 6, 5,  5, 6, 7
        ], dtype=np.uint32)

        self._index_count = indices.size

        self._vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)

        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def draw(self):
        """Draw the cube. Set model matrix and material_color before calling."""
        glBindVertexArray(self._vao)
        glDrawElements(GL_TRIANGLES, self._index_count, GL_UNSIGNED_INT, ctypes.c_void_p(0))


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

def rotate_x(angle):
    c = math.cos(angle)
    s = math.sin(angle)

    M = np.identity(4, dtype=np.float32)
    M[1,1] = c
    M[1,2] = -s
    M[2,1] = s
    M[2,2] = c

    return M

def rotate_y(angle):
    c = math.cos(angle)
    s = math.sin(angle)

    M = np.identity(4, dtype=np.float32)
    M[0,0] = c
    M[0,2] = s
    M[2,0] = -s
    M[2,2] = c

    return M

def rotate_z(angle):
    c = math.cos(angle)
    s = math.sin(angle)

    M = np.identity(4, dtype=np.float32)
    M[0,0] = c
    M[0,1] = -s
    M[1,0] = s
    M[1,1] = c

    return M


def draw_axes(cube, set_model_color, identity, axis_length=0.5, axis_thickness=0.03):
    """Draw X (red), Y (green), Z (blue) axes at the origin using box primitives."""
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)

    x_model = identity @ translate(axis_length / 2, 0, 0) @ scale(axis_length, axis_thickness, axis_thickness)
    set_model_color(x_model, 1, 0, 0)
    cube.draw()

    y_model = identity @ translate(0, axis_length / 2, 0) @ scale(axis_thickness, axis_length, axis_thickness)
    set_model_color(y_model, 0, 1, 0)
    cube.draw()

    z_model = identity @ translate(0, 0, axis_length / 2) @ scale(axis_thickness, axis_thickness, axis_length)
    set_model_color(z_model, 0, 0, 1)
    cube.draw()

    glDisable(GL_POLYGON_OFFSET_FILL)


# Grid
class Grid:
    """XY grid in the z=0 plane."""

    def __init__(self, half_extent=5.0, step=1.0):
        vertices = []
        x = -half_extent
        while x <= half_extent + 1e-9:
            # vertical lines (parallel Y)
            vertices.extend([x, -half_extent, 0.0])
            vertices.extend([x,  half_extent, 0.0])
            # horizontal lines (parallel X)
            vertices.extend([-half_extent, x, 0.0])
            vertices.extend([ half_extent, x, 0.0])
            x += step

        self._vertices = np.array(vertices, dtype=np.float32)
        self._vertex_count = len(self._vertices) // 3

        self._vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, self._vertices.nbytes, self._vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def draw(self):
        """Draw the grid as lines. Set model matrix and material_color before calling."""
        glBindVertexArray(self._vao)
        glLineWidth(2.0)
        glDrawArrays(GL_LINES, 0, self._vertex_count)


# Point Cloud
class PointCloud:
    """Dynamic point cloud renderer (XYZ + per-point white-to-blue color)."""

    def __init__(self, max_points=1_000_000):
        self._max_points = max_points
        self._count = 0

        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        self._cbo = glGenBuffers(1)

        glBindVertexArray(self._vao)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(
            GL_ARRAY_BUFFER,
            max_points * 3 * 4,  # 3 floats per point
            None,
            GL_DYNAMIC_DRAW
        )
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, self._cbo)
        glBufferData(
            GL_ARRAY_BUFFER,
            max_points * 3 * 4,  # 3 floats per color
            None,
            GL_DYNAMIC_DRAW
        )
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

    def update(self, xyz: np.ndarray):
        """Upload Nx3 float32 array; colors are computed as white-to-blue by height (Z)."""
        if xyz.dtype != np.float32:
            xyz = xyz.astype(np.float32)

        self._count = min(len(xyz), self._max_points)
        xyz = xyz[:self._count]

        # White-to-blue scale by Z (low Z = blue, high Z = white); visible on dark bg
        z = xyz[:, 2]
        z_min, z_max = float(np.min(z)), float(np.max(z))
        if z_max > z_min:
            t = (z - z_min) / (z_max - z_min)
        else:
            t = np.ones(self._count, dtype=np.float32) * 0.5
        # (0.2, 0.2, 0.85) -> (1, 1, 1) so low = blue, high = white
        r = np.float32(0.2) + t * np.float32(0.8)
        g = np.float32(0.2) + t * np.float32(0.8)
        b = np.float32(0.85) + t * np.float32(0.15)
        colors = np.column_stack((r, g, b)).astype(np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, xyz.nbytes, xyz)

        glBindBuffer(GL_ARRAY_BUFFER, self._cbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, colors.nbytes, colors)

    def update_colored(self, xyz: np.ndarray, colors: np.ndarray):
        """Upload Nx3 float32 positions and Nx3 uint8 RGB colors."""
        if xyz.dtype != np.float32:
            xyz = xyz.astype(np.float32)
        colors_f = colors.astype(np.float32) / 255.0

        self._count = min(len(xyz), self._max_points)
        xyz = xyz[:self._count]
        colors_f = colors_f[:self._count]

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, xyz.nbytes, xyz)

        glBindBuffer(GL_ARRAY_BUFFER, self._cbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, colors_f.nbytes, colors_f)

    def draw(self):
        glBindVertexArray(self._vao)
        glDrawArrays(GL_POINTS, 0, self._count)

class ArcCameraControl:
    def __init__(self):
        self.center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.distance = 10.0

        self.theta = 0.0
        self.phi = -60.0 * math.pi / 180.0

        self.left_button_down = False
        self.middle_button_down = False

        self.drag_last_pos = np.array([0, 0], dtype=np.int32)

    # Mouse button
    def mouse(self, pos, button, down):
        if button == 0:
            self.left_button_down = down
        elif button == 2:
            self.middle_button_down = down

        self.drag_last_pos = np.array(pos, dtype=np.int32)

    # Mouse drag
    def drag(self, pos, button):
        pos = np.array(pos, dtype=np.int32)
        rel = pos - self.drag_last_pos

        if button == 0:
            # ORBIT
            self.theta -= rel[0] * 0.01
            self.phi   -= rel[1] * 0.01

            # normalize theta [-pi, pi]
            if self.theta > math.pi:
                self.theta -= 2 * math.pi
            if self.theta < -math.pi:
                self.theta += 2 * math.pi

            # clamp phi
            self.phi = np.clip(self.phi,
                               -math.pi/2 + 0.01,
                               math.pi/2 - 0.01)

        elif button == 2:
            # PAN
            rot = self._rotation_matrix()
            right = rot @ np.array([0, 1, 0], dtype=np.float32)
            up    = rot @ np.array([0, 0, 1], dtype=np.float32)

            pan = (-rel[0] * right + rel[1] * up) * self.distance * 0.001
            self.center += pan

        self.drag_last_pos = pos

    # Scroll
    def scroll(self, rel_y):
        if rel_y > 0:
            self.distance *= 0.9
        elif rel_y < 0:
            self.distance *= 1.1

        self.distance = max(0.1, self.distance)

    # Rotation
    def _rotation_matrix(self):
        Rz = np.array([
            [math.cos(self.theta), -math.sin(self.theta), 0],
            [math.sin(self.theta),  math.cos(self.theta), 0],
            [0, 0, 1]
        ])

        Ry = np.array([
            [ math.cos(self.phi), 0, math.sin(self.phi)],
            [0, 1, 0],
            [-math.sin(self.phi), 0, math.cos(self.phi)]
        ])

        return Rz @ Ry

    # View matrix
    def view_matrix(self):
        rot = self._rotation_matrix()

        offset = rot @ np.array([self.distance, 0, 0], dtype=np.float32)
        eye = self.center + offset

        return look_at(eye, self.center, np.array([0, 0, 1], dtype=np.float32))


# Image Texture
class ImageTexture:
    """Creates an OpenGL texture from a numpy RGB image."""

    def __init__(self, image_rgb: np.ndarray):
        assert image_rgb.dtype == np.uint8
        assert image_rgb.ndim == 3

        self.height, self.width = image_rgb.shape[:2]

        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            self.width,
            self.height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            image_rgb
        )

        glBindTexture(GL_TEXTURE_2D, 0)

    def update(self, image_rgb: np.ndarray):
        """Replace texture data (same dimensions) with a new RGB image."""
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexSubImage2D(
            GL_TEXTURE_2D, 0, 0, 0,
            self.width, self.height,
            GL_RGB, GL_UNSIGNED_BYTE,
            np.ascontiguousarray(image_rgb, dtype=np.uint8),
        )
        glBindTexture(GL_TEXTURE_2D, 0)

    def delete(self):
        glDeleteTextures([self.texture_id])

