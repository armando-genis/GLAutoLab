import ctypes
from typing import List
import numpy as np
from OpenGL.GL import (
    glGenVertexArrays, glGenBuffers, glBindVertexArray, glBindBuffer,
    glBufferData, glVertexAttribPointer, glEnableVertexAttribArray,
    glDrawElements, glEnable, glDisable, glPolygonOffset,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW,
    GL_FLOAT, GL_FALSE, GL_TRIANGLES, GL_UNSIGNED_INT,
    GL_POLYGON_OFFSET_FILL,
)

from typing import List, Tuple, Optional
import cv2


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


def _translate(x, y, z):
    M = np.identity(4, dtype=np.float32)
    M[0, 3] = x
    M[1, 3] = y
    M[2, 3] = z
    return M


def _scale(sx, sy, sz):
    M = np.identity(4, dtype=np.float32)
    M[0, 0] = sx
    M[1, 1] = sy
    M[2, 2] = sz
    return M


def draw_axes(cube, set_model_color, identity, axis_length=0.5, axis_thickness=0.03):
    """Draw X (red), Y (green), Z (blue) axes at the origin using box primitives."""
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)

    x_model = identity @ _translate(axis_length / 2, 0, 0) @ _scale(axis_length, axis_thickness, axis_thickness)
    set_model_color(x_model, 1, 0, 0)
    cube.draw()

    y_model = identity @ _translate(0, axis_length / 2, 0) @ _scale(axis_thickness, axis_length, axis_thickness)
    set_model_color(y_model, 0, 1, 0)
    cube.draw()

    z_model = identity @ _translate(0, 0, axis_length / 2) @ _scale(axis_thickness, axis_thickness, axis_length)
    set_model_color(z_model, 0, 0, 1)
    cube.draw()

    glDisable(GL_POLYGON_OFFSET_FILL)


