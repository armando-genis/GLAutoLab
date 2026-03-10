import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import math

# uv pip install PyOpenGL PyOpenGL_accelerate glfw

def create_cube_indexed():
    # 8 vertices, identical to your C++
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

    # Indices, identical ordering to your C++
    indices = np.array([
        0, 1, 2,  1, 3, 2,
        1, 5, 3,  3, 5, 7,
        0, 4, 1,  1, 4, 5,
        3, 6, 2,  3, 7, 6,
        0, 2, 4,  2, 6, 4,
        4, 6, 5,  5, 6, 7
    ], dtype=np.uint32)

    return vertices, indices

# ==============================
# Matrix utilities
# ==============================

def perspective(fov, aspect, near, far):
    f = 1.0 / math.tan(fov / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1
    return M

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

def translate(x, y, z):
    M = np.identity(4, dtype=np.float32)
    M[0,3] = x
    M[1,3] = y
    M[2,3] = z
    return M

def scale(sx, sy, sz):
    M = np.identity(4, dtype=np.float32)
    M[0,0] = sx
    M[1,1] = sy
    M[2,2] = sz
    return M

# ==============================
# Shaders
# ==============================

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
uniform vec4 material_color;
out vec4 FragColor;
void main()
{
    FragColor = material_color;
}
"""

def create_shader():
    v = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(v, VERTEX_SHADER)
    glCompileShader(v)

    f = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(f, FRAGMENT_SHADER)
    glCompileShader(f)

    program = glCreateProgram()
    glAttachShader(program, v)
    glAttachShader(program, f)
    glLinkProgram(program)

    glDeleteShader(v)
    glDeleteShader(f)
    return program

# ==============================
# Grid creation
# ==============================

def create_grid(half_extent=5.0, step=1.0):
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

    return np.array(vertices, dtype=np.float32)


# ==============================
# Main
# ==============================

def main():
    if not glfw.init():
        raise Exception("GLFW failed")

    window = glfw.create_window(900, 700, "3D Grid", None, None)
    glfw.make_context_current(window)

    glEnable(GL_DEPTH_TEST)

    shader = create_shader()

    grid_vertices = create_grid(half_extent=5.0, step=1.0)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, grid_vertices.nbytes, grid_vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glUseProgram(shader)

    # cube
    cube_vertices, cube_indices = create_cube_indexed()

    cubeVAO = glGenVertexArrays(1)
    cubeVBO = glGenBuffers(1)
    cubeEBO = glGenBuffers(1)

    glBindVertexArray(cubeVAO)

    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO)
    glBufferData(GL_ARRAY_BUFFER, cube_vertices.nbytes, cube_vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cube_indices.nbytes, cube_indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindVertexArray(0)

    axis_length = 1.0
    axis_thickness = 0.05

    # Projection
    projection = perspective(math.radians(45), 900/700, 0.1, 100)
    proj_loc = glGetUniformLocation(shader, "projection")
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection.T)

    # View
    eye = np.array([10, 10, 10], dtype=np.float32)
    target = np.array([0, 0, 0], dtype=np.float32)
    up = np.array([0, 0, 1], dtype=np.float32)
    view = look_at(eye, target, up)
    view_loc = glGetUniformLocation(shader, "view")
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view.T)

    model_loc = glGetUniformLocation(shader, "model")

    color_loc = glGetUniformLocation(shader, "material_color")
    glUniform4f(color_loc, 0.8, 0.8, 0.8, 1.0)

    def draw_cube(model_mat, r, g, b, a=1.0):
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_mat.T)
        glUniform4f(color_loc, r, g, b, a)
        glDrawElements(GL_TRIANGLES, cube_indices.size, GL_UNSIGNED_INT, ctypes.c_void_p(0))

    while not glfw.window_should_close(window):
        glfw.poll_events()

        width, height = glfw.get_framebuffer_size(window)
        glViewport(0, 0, width, height)

        glClearColor(0.05, 0.05, 0.05, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        identity = np.identity(4, dtype=np.float32)

        # ---------- GRID (draw first)
        grid_model = translate(0, 0, -0.001)  # tiny offset to avoid z-fighting
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, grid_model.T)
        glUniform4f(color_loc, 0.25, 0.25, 0.25, 1.0)

        glBindVertexArray(VAO)
        glLineWidth(2.0)  # might be clamped to 1 on some drivers
        glDrawArrays(GL_LINES, 0, len(grid_vertices)//3)

        # ---------- AXES (draw after)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)

        glBindVertexArray(cubeVAO)

        # X (red)
        x_model = identity @ translate(axis_length/2, 0, 0) @ scale(axis_length, axis_thickness, axis_thickness)
        draw_cube(x_model, 1, 0, 0)

        # Y (green)
        y_model = identity @ translate(0, axis_length/2, 0) @ scale(axis_thickness, axis_length, axis_thickness)
        draw_cube(y_model, 0, 1, 0)

        # Z (blue)
        z_model = identity @ translate(0, 0, axis_length/2) @ scale(axis_thickness, axis_thickness, axis_length)
        draw_cube(z_model, 0, 0, 1)

        glDisable(GL_POLYGON_OFFSET_FILL)

        glfw.swap_buffers(window)   

    glfw.terminate()

if __name__ == "__main__":
    main()
