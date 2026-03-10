"""
OpenGL 3D visualization
"""

import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import math
from pathlib import Path
import imgui
from imgui.integrations.glfw import GlfwRenderer
from PIL import Image

# uv pip install PyOpenGL PyOpenGL_accelerate glfw
# uv pip install imgui[glfw]
# uv pip install pillow

from dataLoaderModule import SyncDataset

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
    gl_PointSize = 2.0;
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

def _create_shader_program():
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


# Cube
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
    """Dynamic point cloud renderer (XYZ only)."""

    def __init__(self, max_points=1_000_000):
        self._max_points = max_points
        self._count = 0

        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)

        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)

        # Allocate empty buffer (dynamic)
        glBufferData(
            GL_ARRAY_BUFFER,
            max_points * 3 * 4,  # 3 floats per point
            None,
            GL_DYNAMIC_DRAW
        )

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def update(self, xyz: np.ndarray):
        """Upload Nx3 float32 array."""
        if xyz.dtype != np.float32:
            xyz = xyz.astype(np.float32)

        self._count = min(len(xyz), self._max_points)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            xyz[:self._count].nbytes,
            xyz[:self._count]
        )

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

        # Flip vertically (OpenGL origin is bottom-left)
        image_rgb = np.flipud(image_rgb)

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

    def delete(self):
        glDeleteTextures([self.texture_id])


# ImGui
class SceneUI:
    """ImGui-based control panel for scene navigation and rendering options."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.scene_indices = dataset.indices()
        self.current_scene = 0

        self.show_grid = True
        self.show_axes = True

        self._needs_reload = True
        self._textures = []  # store loaded images

    def load_images_for_scene(self, idx):
        self._textures.clear()

        images = self.dataset.load_images(idx)

        for cam_name, img_rgb in images.items():
            self._textures.append(
                (cam_name, ImageTexture(img_rgb))
            )

    def draw(self):
        imgui.begin("Scene Control", True)

        # Scene navigation
        if imgui.button("Prev"):
            self.current_scene = max(0, self.current_scene - 1)
            self._needs_reload = True

        imgui.same_line()

        if imgui.button("Next"):
            self.current_scene = min(len(self.scene_indices) - 1,
                                     self.current_scene + 1)
            self._needs_reload = True

        imgui.text(f"Scene: {self.current_scene}")

        imgui.separator()

        changed, self.show_grid = imgui.checkbox("Show Grid", self.show_grid)
        changed, self.show_axes = imgui.checkbox("Show Axes", self.show_axes)

        imgui.end()

        # Images panel
        imgui.begin("Images Panel", True)

        for cam_name, tex in self._textures:
            imgui.text(cam_name)

            aspect = tex.height / tex.width
            width = 300
            height = width * aspect

            imgui.image(tex.texture_id, width, height)
            imgui.separator()

        imgui.end()

    def consume_reload_flag(self):
        """Returns True if a new scene must be loaded."""
        if self._needs_reload:
            self._needs_reload = False
            return True
        return False

    def get_current_index(self):
        return self.scene_indices[self.current_scene]


# Viz (main renderer)
class Viz:
    """Main visualizer: window, shader, and render loop (grid + cube axes)."""

    def __init__(self, width=900, height=700, title="3D Grid", dataset=None):
        if not glfw.init():
            raise Exception("GLFW failed")

        self._window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(self._window)
        glEnable(GL_DEPTH_TEST)

        self._program = _create_shader_program()
        self._model_loc = glGetUniformLocation(self._program, "model")
        self._color_loc = glGetUniformLocation(self._program, "material_color")

        self._grid = Grid(half_extent=5.0, step=1.0)
        self._cube = Cube()

        self._camera = ArcCameraControl()

        self._pointcloud = PointCloud(max_points=500_000)

        imgui.create_context()
        self._impl = GlfwRenderer(self._window)

        self._dataset = dataset
        self._ui = SceneUI(dataset)

    def _set_model_color(self, model_matrix, r, g, b, a=1.0):
        glUniformMatrix4fv(self._model_loc, 1, GL_FALSE, model_matrix.T)
        glUniform4f(self._color_loc, r, g, b, a)

    def set_pointcloud(self, xyz: np.ndarray):
        self._pointcloud.update(xyz)

    def run(self):
        """Main render loop."""
        glUseProgram(self._program)

        # Pre-cache uniform locations (avoid querying every frame)
        proj_loc = glGetUniformLocation(self._program, "projection")
        view_loc = glGetUniformLocation(self._program, "view")

        while not glfw.window_should_close(self._window):

            # GLFW + ImGui frame start
            glfw.poll_events()
            self._impl.process_inputs()
            imgui.new_frame()

            # Mouse callback
            io = imgui.get_io()

            if not io.want_capture_mouse:
                x, y = glfw.get_cursor_pos(self._window)

                left = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
                middle = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
                shift = glfw.get_key(self._window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS

                btn = -1
                if shift and left:
                    btn = 2
                elif left:
                    btn = 0
                elif middle:
                    btn = 2

                if btn >= 0:
                    self._camera.drag((int(x), int(y)), btn)
                else:
                    self._camera.drag_last_pos = np.array([int(x), int(y)])

            # Scroll
            if not io.want_capture_mouse:
                scroll_y = io.mouse_wheel
                if scroll_y != 0:
                    self._camera.scroll(scroll_y)


            # Draw UI
            self._ui.draw()

            # Reload scene if needed
            if self._dataset is not None and self._ui.consume_reload_flag():
                idx = self._ui.get_current_index()
                xyz = self._dataset.load_lidar(idx)
                self._pointcloud.update(xyz)
                self._ui.load_images_for_scene(idx)

            # Setup viewport + camera
            width, height = glfw.get_framebuffer_size(self._window)
            glViewport(0, 0, width, height)

            aspect = width / height if height else 1.0
            proj = perspective(math.radians(45), aspect, 0.1, 100.0)
            view = self._camera.view_matrix()

            glUniformMatrix4fv(proj_loc, 1, GL_FALSE, proj.T)
            glUniformMatrix4fv(view_loc, 1, GL_FALSE, view.T)

            # Clear
            glClearColor(0.05, 0.05, 0.05, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            identity = np.identity(4, dtype=np.float32)

            # Grid
            if self._ui.show_grid:
                grid_model = translate(0, 0, -0.001)
                self._set_model_color(grid_model, 0.25, 0.25, 0.25, 1.0)
                self._grid.draw()

            # Point Cloud
            self._set_model_color(identity, 1.0, 1.0, 1.0, 1.0)
            self._pointcloud.draw()

            # Axes
            if self._ui.show_axes:
                axis_length = 1.0
                axis_thickness = 0.05

                glEnable(GL_POLYGON_OFFSET_FILL)
                glPolygonOffset(1.0, 1.0)

                x_model = identity @ translate(axis_length / 2, 0, 0) @ scale(axis_length, axis_thickness, axis_thickness)
                self._set_model_color(x_model, 1, 0, 0)
                self._cube.draw()

                y_model = identity @ translate(0, axis_length / 2, 0) @ scale(axis_thickness, axis_length, axis_thickness)
                self._set_model_color(y_model, 0, 1, 0)
                self._cube.draw()

                z_model = identity @ translate(0, 0, axis_length / 2) @ scale(axis_thickness, axis_thickness, axis_length)
                self._set_model_color(z_model, 0, 0, 1)
                self._cube.draw()

                glDisable(GL_POLYGON_OFFSET_FILL)

            # Render ImGui ON TOP
            imgui.render()
            self._impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self._window)

        # Shutdown cleanly
        self._impl.shutdown()
        glfw.terminate()



if __name__ == "__main__":

    dataset = SyncDataset(Path("sync_feb_12"))
    total = dataset.num_scenes()
    print(f"Total scenes: {total}")

    viz = Viz(900, 700, "3D Grid", dataset)
    viz.run()
