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

from dataLoaderModule import SyncDataset
from labelManager import LabelManager, LabelCameraManager
from poseManager import PoseManager
from UlitysModule import Cube, draw_axes
from CameraLidarModule import CameraLidarModule
from carModelModule import CarModel
from ipmModule import IpmModule, TexturedPlane, HDMapBoundaryAccumulator, HDMapGridAccumulator
from personDetectionModule import PersonDetectionModule

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

# Point cloud: per-vertex color (white-to-blue scale)
VERTEX_SHADER_POINTS = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 vColor;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    gl_PointSize = 2.0;
    vColor = color;
}
"""

FRAGMENT_SHADER_POINTS = """
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main()
{
    FragColor = vec4(vColor, 1.0);
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


def _create_pointcloud_shader_program():
    v = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(v, VERTEX_SHADER_POINTS)
    glCompileShader(v)

    f = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(f, FRAGMENT_SHADER_POINTS)
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


# ImGui
class SceneUI:
    """ImGui-based control panel for scene navigation and rendering options."""

    def __init__(self, dataset, label_manager):
        self.dataset = dataset
        self.scene_indices = dataset.indices()
        self.current_scene = 0

        self.show_grid = True
        self.show_axes = True
        self.show_colored_points = False
        self.show_pose = False
        self.show_ipm = False

        self._needs_reload = True
        self._textures = []   # store loaded GL textures
        self._undistorted_images = [] # raw numpy arrays in camera-index order
        self._raw_images = [] # raw numpy arrays in camera-index order
        self._images_mask = [] # mask numpy arrays in camera-index order

        self.label_manager = label_manager

        self.label_types = ["person", "car", "bus", "bicycle"]
        self.new_label_type_idx = 1  # "car"
        self._add_box_request = False
        self._remove_box_request = False
        self._delete_selected_request = False
        self._live_topic_mode = False  # when True: no selection, minimal UI; toggle with button

    def load_images_for_scene(self, idx):
        self._textures.clear()
        self._undistorted_images.clear()
        self._raw_images.clear()
        self._images_mask.clear()
        # raw images
        raw_images = self.dataset.load_raw_images(idx)

        for cam_name, img_rgb in raw_images.items():
            self._raw_images.append(img_rgb)

        # undistorted images
        images = self.dataset.load_images(idx)  # ordered by camera index

        for cam_name, img_rgb in images.items():
            self._textures.append(
                (cam_name, ImageTexture(img_rgb))
            )
            self._undistorted_images.append(img_rgb)

        # masks: align with camera index (same order as images); None where no mask
        masks = self.dataset.load_masks(idx)
        for cam_name, _ in images.items():
            mask = masks.get(cam_name) if masks else None
            self._images_mask.append(mask)

        # print the length of the images_mask
        print(f"Length of images_mask: {len(self._images_mask)}")

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

        if imgui.button("Live Topic Trigger"):
            self._live_topic_mode = not self._live_topic_mode

        imgui.separator()

        changed, self.show_grid = imgui.checkbox("Show Grid", self.show_grid)
        changed, self.show_axes = imgui.checkbox("Show Axes", self.show_axes)
        changed, self.show_colored_points = imgui.checkbox("Show Colored Points", self.show_colored_points)
        if changed:
            self._needs_reload = True
        _, self.show_pose = imgui.checkbox("Show Pose", self.show_pose)
        changed, self.show_ipm = imgui.checkbox("Show IPM", self.show_ipm)
        if changed:
            self._needs_reload = True

        # Label Controls
        imgui.text("3D Labels")

        _, self.new_label_type_idx = imgui.combo(
            "Type",
            self.new_label_type_idx,
            self.label_types
        )

        if imgui.button("Add Box"):
            self._add_box_request = True

        imgui.same_line()

        if imgui.button("Remove Last"):
            self._remove_box_request = True

        imgui.separator()

        # Label List (Scrollable)
        imgui.text("Existing Labels")

        imgui.begin_child("label_scroll_region", 0, 150, border=True)

        labels = self.label_manager.labels()

        if len(labels) == 0:
            imgui.text_disabled("No labels yet.")
        else:
            for i, label in enumerate(labels):
                selected = (i == self.label_manager.selected_index)

                clicked, _ = imgui.selectable(
                    f"{i}: {label.label_type} "
                    f"({label.center[0]:.2f}, "
                    f"{label.center[1]:.2f}, "
                    f"{label.center[2]:.2f})",
                    selected
                )

                if clicked:
                    self.label_manager.select(i)

        imgui.end_child()

        if imgui.button("Delete Selected"):
            self._delete_selected_request = True

        imgui.same_line()

        if imgui.button("Save KITTI"):
            self.label_manager.save_kitti()

        imgui.end()

        # Images panel
        imgui.begin("Images Panel", True)

        for cam_name, tex in self._textures:
            imgui.text(cam_name)

            aspect = tex.height / tex.width
            width = 300
            height = width * aspect

            imgui.image(tex.texture_id, width, height, (0, 0), (1, 1))
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

    def consume_add_request(self):
        if self._add_box_request:
            self._add_box_request = False
            return True
        return False

    def consume_remove_request(self):
        if self._remove_box_request:
            self._remove_box_request = False
            return True
        return False

    def consume_delete_selected_request(self):
        if self._delete_selected_request:
            self._delete_selected_request = False
            return True
        return False

    def selected_label_type(self):
        return self.label_types[self.new_label_type_idx]


# Viz (main renderer)
class Viz:
    """Main visualizer: window, shader, and render loop (grid + cube axes)."""

    def __init__(self, width=900, height=700, title="3D Grid", dataset=None, icons=None):
        if not glfw.init():
            raise Exception("GLFW failed")

        self._window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(self._window)
        glfw.swap_interval(1)
        glEnable(GL_DEPTH_TEST)

        self._program = _create_shader_program()
        self._program_points = _create_pointcloud_shader_program()
        self._model_loc = glGetUniformLocation(self._program, "model")
        self._color_loc = glGetUniformLocation(self._program, "material_color")
        self._model_loc_points = glGetUniformLocation(self._program_points, "model")
        self._view_loc_points = glGetUniformLocation(self._program_points, "view")
        self._proj_loc_points = glGetUniformLocation(self._program_points, "projection")

        self._grid = Grid(half_extent=5.0, step=1.0)
        self._cube = Cube()

        self._camera = ArcCameraControl()

        self._pointcloud = PointCloud(max_points=500_000)
        self._colored_pointcloud = PointCloud(max_points=500_000)
        self._current_xyz = None
        self._current_images = []
        self._path_images = []

        imgui.create_context()
        self._impl = GlfwRenderer(self._window)

        self._dataset = dataset
        self._labels = LabelManager()
        self._labels.set_icons(icons)
        self._ui = SceneUI(dataset, self._labels)
        self._pose = PoseManager()
        self._pose.load_all_poses(dataset)

        self._camera_lidar_module = CameraLidarModule()
        self._camera_lidar_module.load_camera_lidar_parameters(dataset)

        self._label_camera_mgr = LabelCameraManager()
        self._label_camera_mgr.load_camera_lidar_parameters(dataset)

        self._ipm_module = IpmModule(dataset)

        self._ipm_plane = TexturedPlane(alpha=0.4, flip_y=True)  # OpenCV image → flip for GL
        self._hd_grid_plane = TexturedPlane(alpha=1.0, flip_y=False) # World grid already oriented → Do need to flip

        # warp the images to the bird's-eye-view (raw images)
        H, W = self._ipm_module.outputRes
        pxPerM = self._ipm_module.pxPerM

        self.meters_x = W / pxPerM[1]
        self.meters_y = H / pxPerM[0]

        self.resolution = self.meters_x / W   # == 1.0 / pxPerM[1]

        self._hd_boundary_accumulator = HDMapBoundaryAccumulator()

        size = 200

        self._hd_grid_acc = HDMapGridAccumulator(
            size_x_m=size,
            size_y_m=size,
            resolution=self.resolution,
            origin_x=-size/2.0,
            origin_y=-size/2.0,
            decay=0.995
        )

        self._polys = []
        self._path_center = None  # Nx3 world
        self._path_left = None
        self._path_right = None

        self._lidar_model = np.identity(4, dtype=np.float32)
        self._basefootprint_model = np.identity(4, dtype=np.float32)
        self.model_ipm_plane = np.identity(4, dtype=np.float32)
        self._car_model = CarModel(dataset.car_settings)

        self._mouse_pressed = False
        self._mouse_press_pos = None
        self._click_threshold = 5  # pixels
        self._labels_dirty = True

        self._last_size = (0, 0)
        self._proj = None

        self._person_detection_module = PersonDetectionModule()
        self._person_detection_module.load_camera_lidar_parameters(dataset)

    def _set_model_color(self, model_matrix, r, g, b, a=1.0):
        glUseProgram(self._program)
        glUniformMatrix4fv(self._model_loc, 1, GL_FALSE, model_matrix.T)
        glUniform4f(self._color_loc, r, g, b, a)

    @staticmethod
    def _pose_to_matrix(location, heading, roll, pitch) -> np.ndarray:
        """Build a 4x4 model matrix from ego-pose (R = Rz(heading)@Ry(pitch)@Rx(roll), then translate)."""
        cy = math.cos(heading); sy = math.sin(heading)
        cp = math.cos(pitch);   sp = math.sin(pitch)
        cr = math.cos(roll);    sr = math.sin(roll)
        M = np.identity(4, dtype=np.float32)
        M[0, 0] = cy * cp
        M[1, 0] = sy * cp
        M[2, 0] = -sp
        M[0, 1] = cy * sp * sr - sy * cr
        M[1, 1] = sy * sp * sr + cy * cr
        M[2, 1] = cp * sr
        M[0, 2] = cy * sp * cr + sy * sr
        M[1, 2] = sy * sp * cr - cy * sr
        M[2, 2] = cp * cr
        M[0, 3] = float(location[0])
        M[1, 3] = float(location[1])
        M[2, 3] = float(location[2])
        return M

    def set_pointcloud(self, xyz: np.ndarray):
        self._pointcloud.update(xyz)

    def screen_to_world_ray(self, mouse_x, mouse_y, width, height, inv_proj, inv_view, cam_pos):
        x = (2.0 * mouse_x) / width - 1.0
        y = 1.0 - (2.0 * mouse_y) / height

        ray_clip = np.array([x, y, -1.0, 1.0], dtype=np.float32)

        ray_eye = inv_proj @ ray_clip
        ray_eye = np.array([ray_eye[0], ray_eye[1], -1.0, 0.0], dtype=np.float32)

        ray_world = inv_view @ ray_eye
        ray_world = ray_world[:3]
        ray_world /= np.linalg.norm(ray_world)

        return cam_pos, ray_world

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

            # --------------------------------------------------
            # Frame start (already in your loop)
            # --------------------------------------------------
            width, height = glfw.get_framebuffer_size(self._window)

            if (width, height) != self._last_size:
                self._last_size = (width, height)
                aspect = width / height if height else 1.0
                self._proj = perspective(math.radians(45), aspect, 0.1, 100.0)

            proj = self._proj
            view = self._camera.view_matrix()

            io = imgui.get_io()

            # --------------------------------------------------
            # Mouse handling
            # --------------------------------------------------
            if not io.want_capture_mouse:
                x, y = glfw.get_cursor_pos(self._window)
                x, y = int(x), int(y)

                left_pressed   = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
                middle_pressed = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
                shift          = glfw.get_key(self._window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS

                # -----------------------------
                # LEFT PRESS (begin click / maybe begin polygon sphere drag)
                # -----------------------------
                if left_pressed and not self._mouse_pressed:
                    self._mouse_pressed = True
                    self._mouse_press_pos = np.array([x, y])

                    # Compute ray ONLY ON PRESS (so orbit stays cheap)
                    inv_proj = np.linalg.inv(proj)
                    inv_view = np.linalg.inv(view)
                    cam_pos_world = inv_view[:3, 3]

                    cam_pos, ray_dir = self.screen_to_world_ray(
                        x, y, width, height,
                        inv_proj, inv_view, cam_pos_world
                    )

                    if not self._ui._live_topic_mode:

                        polygon_result = self._hd_grid_acc._polygon_spheres.intersect_ray(cam_pos, ray_dir)
                        if polygon_result is not None:
                            polygon_idx, _ = polygon_result
                            self._hd_grid_acc._polygon_spheres.select(polygon_idx)
                            self._hd_grid_acc._polygon_spheres.begin_drag(cam_pos, ray_dir)

                # -----------------------------
                # LEFT RELEASE (click select OR finish drag)
                # -----------------------------
                elif not left_pressed and self._mouse_pressed:
                    release_pos = np.array([x, y])
                    delta = np.linalg.norm(release_pos - self._mouse_press_pos)

                    # CLICK (not drag)
                    if delta < self._click_threshold:
                        inv_proj = np.linalg.inv(proj)
                        inv_view = np.linalg.inv(view)
                        cam_pos_world = inv_view[:3, 3]

                        cam_pos, ray_dir = self.screen_to_world_ray(
                            x, y, width, height,
                            inv_proj, inv_view, cam_pos_world
                        )

                        if not self._ui._live_topic_mode:

                            # bring ray into lidar frame so label.intersect_ray works correctly
                            inv_lidar = np.linalg.inv(self._lidar_model)
                            cam_pos_l = (inv_lidar @ np.append(cam_pos, 1.0))[:3]
                            ray_dir_l = (inv_lidar @ np.append(ray_dir, 0.0))[:3]
                            n = np.linalg.norm(ray_dir_l)
                            if n > 1e-8:
                                ray_dir_l /= n

                            closest_label = None
                            min_label_t = float("inf")

                            for i, label in enumerate(self._labels.labels()):
                                t = label.intersect_ray(cam_pos_l, ray_dir_l)
                                if t is not None and t < min_label_t:
                                    min_label_t = t
                                    closest_label = i

                            # pose sphere pick (world ray)
                            sphere_result = self._pose._marker_renderer.intersect_ray(cam_pos, ray_dir)
                            if sphere_result is not None:
                                sphere_idx, sphere_t = sphere_result
                            else:
                                sphere_idx, sphere_t = None, float("inf")

                            # polygon sphere pick (world ray)
                            polygon_result = self._hd_grid_acc._polygon_spheres.intersect_ray(cam_pos, ray_dir)
                            if polygon_result is not None:
                                polygon_idx, polygon_t = polygon_result
                            else:
                                polygon_idx, polygon_t = None, float("inf")

                            # selection priority (same as your logic)
                            if sphere_idx is not None and sphere_t < min_label_t and sphere_t < polygon_t:
                                self._pose._marker_renderer.select(sphere_idx)
                            elif polygon_idx is not None and polygon_t < min_label_t and polygon_t < sphere_t:
                                self._hd_grid_acc._polygon_spheres.select(polygon_idx)
                            elif closest_label is not None:
                                self._labels.select(closest_label)
                                self._labels_dirty = True

                    # Finish polygon drag (if any) + bake grid once on release (only when releasing)
                    if not self._ui._live_topic_mode:
                        drag_idx = self._hd_grid_acc._polygon_spheres._selected_index
                        was_dragging = self._hd_grid_acc._polygon_spheres._dragging

                        self._hd_grid_acc._polygon_spheres.end_drag()

                        if was_dragging and drag_idx >= 0:
                            # ensure final position is committed
                            self._hd_grid_acc.sync_sphere_to_polygon(drag_idx)
                            self._hd_grid_acc.rasterize_edited_polygons_to_grid()
                            rgba = self._hd_grid_acc.to_rgba(alpha_scale=220)
                            self._hd_grid_plane.set_texture_rgba(rgba)

                    self._mouse_pressed = False

                # -----------------------------
                # DRAG behavior (either sphere drag OR camera orbit/pan)
                # -----------------------------
                if left_pressed:
                    if self._hd_grid_acc._polygon_spheres._dragging and not self._ui._live_topic_mode:
                        # Only compute ray while dragging a polygon sphere
                        inv_proj = np.linalg.inv(proj)
                        inv_view = np.linalg.inv(view)
                        cam_pos_world = inv_view[:3, 3]

                        cam_pos, ray_dir = self.screen_to_world_ray(
                            x, y, width, height,
                            inv_proj, inv_view, cam_pos_world
                        )

                        self._hd_grid_acc._polygon_spheres.drag(cam_pos, ray_dir)
                        idx = self._hd_grid_acc._polygon_spheres._selected_index
                        if idx >= 0:
                            self._hd_grid_acc.sync_sphere_to_polygon(idx)

                    else:
                        # Pure camera movement (NO ray math)
                        if shift:
                            self._camera.drag((x, y), 2)  # pan
                        else:
                            self._camera.drag((x, y), 0)  # orbit

                elif middle_pressed:
                    self._camera.drag((x, y), 2)

                else:
                    self._camera.drag_last_pos = np.array([x, y])

            # --------------------------------------------------
            # Scroll (unchanged)
            # --------------------------------------------------
            if not io.want_capture_mouse:
                scroll_y = io.mouse_wheel
                if scroll_y != 0:
                    self._camera.scroll(scroll_y)

            # --------------------------------------------------
            # Keyboard (your original block, unchanged)
            # --------------------------------------------------
            if not io.want_capture_keyboard:
                speed = 0.05
                moved = False

                if not self._ui._live_topic_mode:

                    # ESC: release polygon edit (deselect sphere, end drag, allow HD map to update again)
                    if glfw.get_key(self._window, glfw.KEY_ESCAPE) == glfw.PRESS:
                        self._hd_grid_acc._polygon_spheres.end_drag()
                        self._hd_grid_acc._polygon_spheres.select(-1)
                        self._hd_grid_acc.clear_polygon_edits()

                    if glfw.get_key(self._window, glfw.KEY_UP) == glfw.PRESS:
                        self._labels.move_selected_local(speed, 0, 0); moved = True

                    if glfw.get_key(self._window, glfw.KEY_DOWN) == glfw.PRESS:
                        self._labels.move_selected_local(-speed, 0, 0); moved = True

                    if glfw.get_key(self._window, glfw.KEY_LEFT) == glfw.PRESS:
                        self._labels.move_selected_local(0, speed, 0); moved = True

                    if glfw.get_key(self._window, glfw.KEY_RIGHT) == glfw.PRESS:
                        self._labels.move_selected_local(0, -speed, 0); moved = True

                    if glfw.get_key(self._window, glfw.KEY_PAGE_UP) == glfw.PRESS:
                        self._labels.move_selected(0, 0, speed); moved = True

                    if glfw.get_key(self._window, glfw.KEY_PAGE_DOWN) == glfw.PRESS:
                        self._labels.move_selected(0, 0, -speed); moved = True

                    if glfw.get_key(self._window, glfw.KEY_W) == glfw.PRESS:
                        self._labels.move_selected(0, 0, speed); moved = True

                    if glfw.get_key(self._window, glfw.KEY_S) == glfw.PRESS:
                        self._labels.move_selected(0, 0, -speed); moved = True

                    if glfw.get_key(self._window, glfw.KEY_A) == glfw.PRESS:
                        self._labels.rotate_selected(0.02); moved = True

                    if glfw.get_key(self._window, glfw.KEY_D) == glfw.PRESS:
                        self._labels.rotate_selected(-0.02); moved = True

                    if moved:
                        self._labels_dirty = True

            if not self._ui._live_topic_mode:

                if self._dataset is not None and self._ui.consume_reload_flag():
                    # reload scene
                    idx = self._ui.get_current_index()
                    # set labels for scene
                    self._labels.set_scene(idx)
                    # load lidar for scene
                    xyz = self._dataset.load_lidar(idx)
                    self._pointcloud.update(xyz)
                    # load images for scene
                    self._ui.load_images_for_scene(idx)
                    self._current_images = self._ui._undistorted_images
                    self._current_raw_images = self._ui._raw_images
                    self._current_images_mask = self._ui._images_mask
                    # paint pointcloud with the rgb images (undistorted)
                    if self._ui.show_colored_points:
                        self._camera_lidar_module.start_colored_pointcloud(
                            lidar_xyz=xyz,
                            images=self._current_images,
                        )
                    # pose of the scene (location, heading, roll, pitch) of the ego vehicle
                    result = self._dataset.load_pose(idx)
                    if result is not None:
                        location, heading, roll, pitch = result
                        self._pose.update(location, heading, roll, pitch)
                        # set the lidar ego pose matrix based on the currect pose of the scene
                        self._lidar_model = Viz._pose_to_matrix(location, heading, roll, pitch)
                        self._basefootprint_model = self._car_model.get_basefootprint_frame(self._lidar_model)
                    else:
                        self._lidar_model = np.identity(4, dtype=np.float32)
                        self._basefootprint_model = np.identity(4, dtype=np.float32)
                    self._labels_dirty = True

                    # draw the path on the images
                    self._path_images = self._label_camera_mgr.bake_path_on_images(
                        self._current_images, self._pose.path_positions,
                        self._lidar_model, self._pose.path_width
                    )

                    self.model_ipm_plane = self._basefootprint_model @ scale(self.meters_x, self.meters_y, 1.0)

                    bev, valid_mask, bev_drivable = self._ipm_module.warp_images(self._current_raw_images, self._current_images_mask)

                    self._ipm_plane.set_texture(bev, valid_mask)

                    if bev_drivable is not None:
                        # poligon set
                        self._hd_boundary_accumulator.update(bev_drivable, self.model_ipm_plane)
                        # grid accumulation
                        self._hd_grid_acc.update(
                            bev_drivable,
                            bev,                 # <-- pass colored BEV image
                            self.model_ipm_plane
                        )

                        if self._hd_grid_acc._polygons_edited:
                            self._polys = self._hd_grid_acc.get_editable_polygons() or []
                        else:
                            self._polys = self._hd_grid_acc.extract_polygons()
                            self._hd_grid_acc.update_polygon_spheres(self._polys)

                        rgba = self._hd_grid_acc.to_rgba(alpha_scale=220)
                        self._hd_grid_plane.set_texture_rgba(rgba)

                        center = np.array(self._pose.path_positions, dtype=np.float32)
                        if center.ndim == 1:
                            center = center.reshape(-1, 2)
                        if center.shape[1] == 2:
                            center = np.concatenate([center, np.zeros((len(center), 1), dtype=np.float32)], axis=1)

                        if self._polys:
                            poly = self._polys[0]  # assuming single drivable region

                            left, right = self._hd_grid_acc.split_polygon_left_right_from_centerline(
                                polygon_world=poly,
                                centerline_world=np.asarray(self._pose.path_positions, dtype=np.float32)
                            )

                        self._path_center = center
                        self._path_left = left
                        self._path_right = right

                    # person detection
                    self._person_detection_module.get_camera_images(self._current_images)

                # only if true, color the pointcloud with the rgb images (undistorted)
                if self._ui.show_colored_points:
                    self._camera_lidar_module.poll_colored_pointcloud(self._colored_pointcloud)

                # draw the labels on the images
                if self._labels_dirty and self._path_images:
                    self._labels_dirty = False
                    annotated = self._label_camera_mgr.draw_labels_on_camera(
                        self._path_images, self._labels.labels()
                    )
                    for i, (_cam_name, tex) in enumerate(self._ui._textures):
                        if i < len(annotated) and annotated[i] is not None:
                            tex.update(annotated[i])

                # Draw UI
                self._ui.draw()

                # Setup viewport + camera
                width, height = glfw.get_framebuffer_size(self._window)
                glViewport(0, 0, width, height)

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

                # draw the IPM plane based on the base footprint frame
                if self._ui.show_ipm and self._ipm_plane.has_texture:

                    self._ipm_plane.draw(view.T, proj.T, self.model_ipm_plane.T)
                    if self._hd_boundary_accumulator.get_polygons() is not None:
                        glUseProgram(self._program)
                        self._set_model_color(identity, 1.0, 1.0, 1.0, 1.0)  # white; change r,g,b to change color
                        glLineWidth(4.0)
                        for poly in self._hd_boundary_accumulator.get_polygons():
                            glBegin(GL_LINE_LOOP)
                            for p in poly:
                                glVertex3f(p[0], p[1], p[2] + 0.2)  # small lift above plane
                            glEnd()
                        glLineWidth(1.0)

                if self._ui.show_ipm and self._hd_grid_plane.has_texture:

                    grid_model = translate(
                        self._hd_grid_acc.origin_x + self._hd_grid_acc.size_x / 2.0,
                        self._hd_grid_acc.origin_y + self._hd_grid_acc.size_y / 2.0,
                        0.01
                    ) @ scale(
                        self._hd_grid_acc.size_x,
                        self._hd_grid_acc.size_y,
                        1.0
                    )

                    self._hd_grid_plane.draw(view.T, proj.T, grid_model.T)

                    if self._polys:
                        glUseProgram(self._program)
                        self._set_model_color(identity, 0.0, 1.0, 0.0, 1.0)
                        glLineWidth(4.0)

                        for poly in self._polys:
                            glBegin(GL_LINE_LOOP)
                            for p in poly:
                                glVertex3f(p[0], p[1], p[2] + 0.05)
                            glEnd()

                        glLineWidth(1.0)

                        self._hd_grid_acc.draw_polygon_spheres(view.T, proj.T)


                    if self._path_center is not None and len(self._path_center) >= 2:
                        z_lift = 0.05
                        glUseProgram(self._program)
                        glLineWidth(3.0)
                        center, left, right = self._path_center, self._path_left, self._path_right
                        # centerline: yellow
                        self._set_model_color(identity, 1.0, 1.0, 0.0, 1.0)
                        glBegin(GL_LINE_STRIP)
                        for p in center:
                            glVertex3f(p[0], p[1], p[2] + z_lift)
                        glEnd()
                        if left is not None:
                            self._set_model_color(identity, 0.0, 1.0, 1.0, 1.0)
                            glBegin(GL_LINE_STRIP)
                            for p in left:
                                glVertex3f(p[0], p[1], p[2] + z_lift)
                            glEnd()
                        if right is not None:
                            self._set_model_color(identity, 1.0, 0.0, 1.0, 1.0)
                            glBegin(GL_LINE_STRIP)
                            for p in right:
                                glVertex3f(p[0], p[1], p[2] + z_lift)
                            glEnd()
                        glLineWidth(1.0)

                # Point Cloud: transformed to world space via ego-pose; labels stay at lidar origin
                glUseProgram(self._program_points)
                glUniformMatrix4fv(self._proj_loc_points, 1, GL_FALSE, proj.T)
                glUniformMatrix4fv(self._view_loc_points, 1, GL_FALSE, view.T)
                glUniformMatrix4fv(self._model_loc_points, 1, GL_FALSE, self._lidar_model.T)
                self._pointcloud.draw()
                glUseProgram(self._program)

                # Axes
                if self._ui.show_axes:
                    draw_axes(self._cube, self._set_model_color, identity)

                # draw camera axes based on the lidar frame
                self._camera_lidar_module.draw_cameras_lidar_frame_axes(self._cube, self._lidar_model, view.T, proj.T, self._set_model_color)

                # draw car axes based on the lidar frame
                self._car_model.draw_axes(self._cube, self._lidar_model, self._set_model_color, identity)

                # draw camera-colored LiDAR point cloud (uploaded once per scene on reload)
                glUseProgram(self._program_points)
                glUniformMatrix4fv(self._proj_loc_points, 1, GL_FALSE, proj.T)
                glUniformMatrix4fv(self._view_loc_points, 1, GL_FALSE, view.T)
                glUniformMatrix4fv(self._model_loc_points, 1, GL_FALSE, self._lidar_model.T)
                if self._ui.show_colored_points:
                    self._colored_pointcloud.draw()
                glUseProgram(self._program)

                # Labels: stored in lidar frame, rendered in world space via lidar_model
                self._labels.draw(self._cube, self._set_model_color, view.T, proj.T, self._lidar_model)
                glUseProgram(self._program)

                # Ego pose
                self._pose.draw(view.T, proj.T, show_pose=self._ui.show_pose)
                glUseProgram(self._program)

                if self._ui.consume_add_request():
                    self._labels.add_label(
                        center=[0, 0, 1],
                        size=[2, 1, 1.5],
                        yaw=0.0,
                        label_type=self._ui.selected_label_type()
                    )
                    self._labels_dirty = True

                if self._ui.consume_remove_request():
                    self._labels.remove_last()
                    self._labels_dirty = True

                if self._ui.consume_delete_selected_request():
                    self._labels.remove_selected()
                    self._labels_dirty = True

            else:
                # Draw UI
                self._ui.draw()

                # Setup viewport + camera
                width, height = glfw.get_framebuffer_size(self._window)
                glViewport(0, 0, width, height)

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

                

            # Render ImGui ON TOP
            imgui.render()
            self._impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self._window)

        # Shutdown cleanly
        self._impl.shutdown()
        glfw.terminate()

if __name__ == "__main__":

    dataset = SyncDataset(Path("../dataset-sdv-feb23"))
    total = dataset.num_scenes()
    print(f"Total scenes: {total}")

    icons = {
        "car": Image.open("icons/car.png"),
        "person": Image.open("icons/person.png"),
        "bicycle": Image.open("icons/bike.png"),
        "bus": Image.open("icons/bus.png"),
    }

    config_dir = Path("camera_configs")

    dataset.build_camera_array(config_dir)
    dataset.print_camera_info()

    viz = Viz(900, 700, "3D Grid", dataset, icons)
    viz.run()
