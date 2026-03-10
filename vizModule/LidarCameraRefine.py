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
import yaml

from ModelsGBLModule import ModelUpload
from shaderModules import create_shader_program, create_pointcloud_shader_program
from dataLoaderModule import SyncDataset
from labelManager import LabelManager, LabelCameraManager
import cv2
from poseManager import PoseManager
from UlitysModule import Cube, draw_axes, Grid, PointCloud, ArcCameraControl, ImageTexture, perspective, translate, rotate_x, rotate_y, rotate_z
from CameraLidarModule import CameraLidarModule
from carModelModule import CarModel
from personDetectionModule import PersonDetectionModule
from ipmModule import IpmModule

_R_CV_TO_ROBOT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

def _euler_to_rotation(ax, ay, az):
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx

def _axis_rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    elif axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _depth_to_rainbow(depth_norm):
    """Map 0..1 depth values to rainbow RGB (red=close, blue=far)."""
    h = (depth_norm * 270.0).astype(np.float32)
    hsv = np.zeros((len(h), 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = (h * 0.5).astype(np.uint8)  # OpenCV hue is 0..180
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)
    return rgb


def _project_lidar_on_image(img, xyz, R, t, K, max_depth=50.0):
    """Project lidar points onto a camera image as rainbow depth-colored dots.
    Uses pinhole K; image is expected undistorted so projection matches."""
    pts = xyz.astype(np.float64)
    cam = (R @ pts.T).T + t.ravel().reshape(1, 3)

    mask = cam[:, 2] > 0.1
    cam = cam[mask]
    if len(cam) == 0:
        return img

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = cam[:, 2]
    u = (fx * cam[:, 0] / z + cx).astype(np.int32)
    v = (fy * cam[:, 1] / z + cy).astype(np.int32)

    h, w = img.shape[:2]
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inside], v[inside], z[inside]

    if len(u) == 0:
        return img

    depth_norm = np.clip(z / max_depth, 0.0, 1.0)
    colors = _depth_to_rainbow(depth_norm)

    out = img.copy()
    for i in range(len(u)):
        cv2.circle(out, (int(u[i]), int(v[i])), 5,
                   (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])), -1)
    return out


# ImGui
class SceneUI:
    """ImGui-based control panel for scene navigation and rendering options."""

    def __init__(self, dataset, label_manager):
        self.dataset = dataset
        self.scene_indices = dataset.indices()
        self.current_scene = 0
        self._goto_scene_text = "0"

        self.show_grid = True
        self.show_axes = True
        self.show_colored_points = False
        self.show_pose = False
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

        self.selected_cam_idx = 0
        self._extrinsic_changed = False
        self._save_extrinsic_request = False
        self._restore_extrinsic_request = False
        self.show_lidar_overlay = True


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

        imgui.text(f"Scene: {self.current_scene} / {len(self.scene_indices) - 1}")

        imgui.push_item_width(80)
        changed, self._goto_scene_text = imgui.input_text(
            "##goto", self._goto_scene_text, 16,
            imgui.INPUT_TEXT_CHARS_DECIMAL
        )
        imgui.pop_item_width()
        imgui.same_line()
        if imgui.button("Go"):
            try:
                idx = int(self._goto_scene_text)
                idx = max(0, min(idx, len(self.scene_indices) - 1))
                self.current_scene = idx
                self._goto_scene_text = str(idx)
                self._needs_reload = True
            except ValueError:
                pass

        if imgui.button("Live Topic Trigger"):
            self._live_topic_mode = not self._live_topic_mode

        imgui.separator()

        changed, self.show_grid = imgui.checkbox("Show Grid", self.show_grid)
        changed, self.show_axes = imgui.checkbox("Show Axes", self.show_axes)
        changed, self.show_colored_points = imgui.checkbox("Show Colored Points", self.show_colored_points)
        if changed:
            self._needs_reload = True
        _, self.show_pose = imgui.checkbox("Show Pose", self.show_pose)
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
            width = 450
            height = width * aspect

            imgui.image(tex.texture_id, width, height, (0, 0), (1, 1))
            imgui.separator()

        imgui.end()

        self.draw_extrinsic_panel()

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

    # --- Extrinsic refinement panel ---

    def draw_extrinsic_panel(self):
        imgui.begin("Calibration Refinement", True)

        extrinsics = self.dataset.extrinsics_array
        intrinsics = self.dataset.camera_array
        num_cams = len(extrinsics)
        if num_cams == 0:
            imgui.text("No cameras loaded.")
            imgui.end()
            return

        cam_names = [f"Camera {i}" for i in range(num_cams)]
        _, self.selected_cam_idx = imgui.combo(
            "Camera", self.selected_cam_idx, cam_names
        )

        ext = extrinsics[self.selected_cam_idx]
        if ext is None:
            imgui.text("No extrinsic data.")
            imgui.end()
            return

        _, self.show_lidar_overlay = imgui.checkbox(
            "Show LiDAR Points", self.show_lidar_overlay
        )

        t_step = 0.01
        r_step = math.radians(0.1)

        # Camera position in lidar frame (so translation moves along lidar X/Y/Z)
        cam_center_lidar = (-ext.R_opencv.T @ ext.t_opencv).ravel()

        imgui.separator()
        imgui.text("Translation (0.01 m/step, lidar frame)")

        for i, name in enumerate(["lidar X", "lidar Y", "lidar Z"]):
            if imgui.button(f"- ##neg_{name}"):
                delta = np.zeros((3, 1), dtype=np.float64)
                delta[i, 0] = t_step
                ext.t_opencv = ext.t_opencv + ext.R_opencv @ delta
                ext.t_robot = _R_CV_TO_ROBOT @ ext.t_opencv
                self._extrinsic_changed = True
            imgui.same_line()
            imgui.text(f"{name}: {cam_center_lidar[i]:+.4f}")
            imgui.same_line()
            if imgui.button(f"+ ##pos_{name}"):
                delta = np.zeros((3, 1), dtype=np.float64)
                delta[i, 0] = t_step
                ext.t_opencv = ext.t_opencv - ext.R_opencv @ delta
                ext.t_robot = _R_CV_TO_ROBOT @ ext.t_opencv
                self._extrinsic_changed = True

        imgui.separator()
        imgui.text("Rotation (0.1 deg/step, in place)")

        for i, name in enumerate(["rx", "ry", "rz"]):
            if imgui.button(f"- ##neg_{name}"):
                R_delta = _axis_rotation(i, -r_step)
                ext.R_opencv = R_delta @ ext.R_opencv
                ext.t_opencv = R_delta @ ext.t_opencv
                ext.R_robot = _R_CV_TO_ROBOT @ ext.R_opencv
                ext.t_robot = _R_CV_TO_ROBOT @ ext.t_opencv
                self._extrinsic_changed = True
            imgui.same_line()
            imgui.text(f"{name}")
            imgui.same_line()
            if imgui.button(f"+ ##pos_{name}"):
                R_delta = _axis_rotation(i, r_step)
                ext.R_opencv = R_delta @ ext.R_opencv
                ext.t_opencv = R_delta @ ext.t_opencv
                ext.R_robot = _R_CV_TO_ROBOT @ ext.R_opencv
                ext.t_robot = _R_CV_TO_ROBOT @ ext.t_opencv
                self._extrinsic_changed = True

        cam = (intrinsics[self.selected_cam_idx]
               if self.selected_cam_idx < len(intrinsics) else None)
        if cam is not None:
            K = cam.get_K()
            f_step = 1.0

            imgui.separator()
            imgui.text("Intrinsics (1 px/step)")

            if imgui.button("- ##neg_fx"):
                K[0, 0] -= f_step
                self._extrinsic_changed = True
            imgui.same_line()
            imgui.text(f"fx: {K[0, 0]:.1f}")
            imgui.same_line()
            if imgui.button("+ ##pos_fx"):
                K[0, 0] += f_step
                self._extrinsic_changed = True

            if imgui.button("- ##neg_fy"):
                K[1, 1] -= f_step
                self._extrinsic_changed = True
            imgui.same_line()
            imgui.text(f"fy: {K[1, 1]:.1f}")
            imgui.same_line()
            if imgui.button("+ ##pos_fy"):
                K[1, 1] += f_step
                self._extrinsic_changed = True

            if imgui.button("- ##neg_cx"):
                K[0, 2] -= f_step
                self._extrinsic_changed = True
            imgui.same_line()
            imgui.text(f"cx: {K[0, 2]:.1f}")
            imgui.same_line()
            if imgui.button("+ ##pos_cx"):
                K[0, 2] += f_step
                self._extrinsic_changed = True

            if imgui.button("- ##neg_cy"):
                K[1, 2] -= f_step
                self._extrinsic_changed = True
            imgui.same_line()
            imgui.text(f"cy: {K[1, 2]:.1f}")
            imgui.same_line()
            if imgui.button("+ ##pos_cy"):
                K[1, 2] += f_step
                self._extrinsic_changed = True

        imgui.separator()
        if imgui.button("Save Extrinsic"):
            self._save_extrinsic_request = True
        imgui.same_line()
        if imgui.button("Restore"):
            self._restore_extrinsic_request = True

        imgui.end()

    def consume_extrinsic_changed(self):
        if self._extrinsic_changed:
            self._extrinsic_changed = False
            return True
        return False

    def consume_save_extrinsic_request(self):
        if self._save_extrinsic_request:
            self._save_extrinsic_request = False
            return True
        return False

    def consume_restore_extrinsic_request(self):
        if self._restore_extrinsic_request:
            self._restore_extrinsic_request = False
            return True
        return False


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

        self._program = create_shader_program()
        self._program_points = create_pointcloud_shader_program()
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

        self._ipm_module = IpmModule(dataset)

        self._camera_lidar_module = CameraLidarModule()
        self._camera_lidar_module.load_camera_lidar_parameters(dataset)

        self._label_camera_mgr = LabelCameraManager()
        self._label_camera_mgr.load_camera_lidar_parameters(dataset)

        self._lidar_model = np.identity(4, dtype=np.float32)
        self._basefootprint_model = np.identity(4, dtype=np.float32)
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

    def _overlay_lidar_on_images(self, images, xyz):
        """Project lidar points onto each camera image as depth-colored dots."""
        extrinsics = self._dataset.extrinsics_array
        intrinsics = self._dataset.camera_array
        result = []
        for cam_idx, img in enumerate(images):
            if img is None:
                result.append(None)
                continue
            ext = extrinsics[cam_idx] if cam_idx < len(extrinsics) else None
            cam = intrinsics[cam_idx] if cam_idx < len(intrinsics) else None
            if ext is None or cam is None:
                result.append(img)
                continue
            R = ext.R_opencv.astype(np.float64)
            t = ext.t_opencv.astype(np.float64)
            K = cam.get_K().astype(np.float64)
            result.append(_project_lidar_on_image(img, xyz, R, t, K))
        return result

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

    def _save_current_extrinsic(self):
        cam_idx = self._ui.selected_cam_idx
        ext = self._dataset.extrinsics_array[cam_idx]
        if ext is None:
            print(f"Camera {cam_idx}: no extrinsic to save.")
            return

        lr = ext.lidar_rotation
        R_lidar = _euler_to_rotation(lr['axis_x'], lr['axis_y'], lr['axis_z'])

        R_opencv_save = ext.R_opencv @ R_lidar
        t_opencv_save = ext.t_opencv.copy()
        R_robot_save = _R_CV_TO_ROBOT @ R_opencv_save
        t_robot_save = _R_CV_TO_ROBOT @ t_opencv_save

        def _fmt(arr):
            return [round(float(x), 10) for x in arr.flatten()]

        data = {
            'extrinsics': {
                'opencv_frame': {
                    'R': {'rows': 3, 'cols': 3, 'data': _fmt(R_opencv_save)},
                    't': {'rows': 3, 'cols': 1, 'data': _fmt(t_opencv_save)},
                },
                'robot_frame': {
                    'R': {'rows': 3, 'cols': 3, 'data': _fmt(R_robot_save)},
                    't': {'rows': 3, 'cols': 1, 'data': _fmt(t_robot_save)},
                },
            }
        }

        path = Path("camera_configs") / f"LidartoCam{cam_idx}_refined.yaml"
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=None, sort_keys=False)
        print(f"Saved refined extrinsic for Camera {cam_idx} to {path}")

    def _restore_current_extrinsic(self):
        cam_idx = self._ui.selected_cam_idx
        ext = self._dataset.extrinsics_array[cam_idx]
        if ext is None:
            print(f"Camera {cam_idx}: no extrinsic to restore.")
            return

        path = Path("camera_configs") / f"LidartoCam{cam_idx}.yaml"
        if not path.exists():
            print(f"File not found: {path}")
            return

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        opencv = data["extrinsics"]["opencv_frame"]
        robot = data["extrinsics"]["robot_frame"]

        R_opencv = np.array(opencv["R"]["data"], dtype=np.float64).reshape(3, 3)
        t_opencv = np.array(opencv["t"]["data"], dtype=np.float64).reshape(3, 1)
        R_robot = np.array(robot["R"]["data"], dtype=np.float64).reshape(3, 3)
        t_robot = np.array(robot["t"]["data"], dtype=np.float64).reshape(3, 1)

        lr = ext.lidar_rotation
        R_lidar = _euler_to_rotation(lr['axis_x'], lr['axis_y'], lr['axis_z'])

        ext.R_opencv = R_opencv @ R_lidar.T
        ext.t_opencv = t_opencv
        ext.R_robot = R_robot @ R_lidar.T
        ext.t_robot = t_robot

        if self._current_images:
            self._path_images = self._label_camera_mgr.bake_path_on_images(
                self._current_images, self._pose.path_positions,
                self._lidar_model, self._pose.path_width
            )
        self._labels_dirty = True
        if self._ui.show_colored_points and self._current_xyz is not None:
            self._camera_lidar_module.start_colored_pointcloud(
                lidar_xyz=self._current_xyz,
                images=self._current_images,
            )

        print(f"Restored extrinsic for Camera {cam_idx} from {path}")

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
                # LEFT PRESS
                # -----------------------------
                if left_pressed and not self._mouse_pressed:
                    self._mouse_pressed = True
                    self._mouse_press_pos = np.array([x, y])

                # -----------------------------
                # LEFT RELEASE (click select)
                # -----------------------------
                elif not left_pressed and self._mouse_pressed:
                    release_pos = np.array([x, y])
                    delta = np.linalg.norm(release_pos - self._mouse_press_pos)

                    if delta < self._click_threshold:
                        inv_proj = np.linalg.inv(proj)
                        inv_view = np.linalg.inv(view)
                        cam_pos_world = inv_view[:3, 3]

                        cam_pos, ray_dir = self.screen_to_world_ray(
                            x, y, width, height,
                            inv_proj, inv_view, cam_pos_world
                        )

                        if not self._ui._live_topic_mode:
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

                            if closest_label is not None:
                                self._labels.select(closest_label)
                                self._labels_dirty = True

                    self._mouse_pressed = False

                # -----------------------------
                # DRAG behavior (camera orbit/pan)
                # -----------------------------
                if left_pressed:
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
                    self._current_xyz = xyz
                    # load images for scene
                    self._ui.load_images_for_scene(idx)
                    self._current_images = self._ui._undistorted_images
                    if self._ui.show_colored_points:
                        self._camera_lidar_module.start_colored_pointcloud(
                            lidar_xyz=xyz,
                            images=self._current_images,
                        )
                    # Lidar is world frame: no per-scene pose, keep identity
                    self._lidar_model = np.identity(4, dtype=np.float32)
                    self._basefootprint_model = np.identity(4, dtype=np.float32)
                    self._labels_dirty = True

                    # draw the path on the images
                    self._path_images = self._label_camera_mgr.bake_path_on_images(
                        self._current_images, self._pose.path_positions,
                        self._lidar_model, self._pose.path_width
                    )

                    # person detection
                    self._person_detection_module.get_camera_images(self._current_images)

                # extrinsic refinement: re-project when user adjusts extrinsic
                if self._ui.consume_extrinsic_changed():
                    if self._current_images:
                        self._path_images = self._label_camera_mgr.bake_path_on_images(
                            self._current_images, self._pose.path_positions,
                            self._lidar_model, self._pose.path_width
                        )
                    self._labels_dirty = True
                    if self._ui.show_colored_points and self._current_xyz is not None:
                        self._camera_lidar_module.start_colored_pointcloud(
                            lidar_xyz=self._current_xyz,
                            images=self._current_images,
                        )

                if self._ui.consume_save_extrinsic_request():
                    self._save_current_extrinsic()

                if self._ui.consume_restore_extrinsic_request():
                    self._restore_current_extrinsic()

                # only if true, color the pointcloud with the rgb images (undistorted)
                if self._ui.show_colored_points:
                    self._camera_lidar_module.poll_colored_pointcloud(self._colored_pointcloud)

                # draw the labels (and optional lidar overlay) on the images
                if self._labels_dirty and self._path_images:
                    self._labels_dirty = False
                    base_imgs = self._path_images
                    if (self._ui.show_lidar_overlay
                            and self._current_xyz is not None):
                        base_imgs = self._overlay_lidar_on_images(
                            self._path_images, self._current_xyz
                        )
                    annotated = self._label_camera_mgr.draw_labels_on_camera(
                        base_imgs, self._labels.labels()
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

    dataset = SyncDataset(Path("../dataset-sdv-feb28"))
    total = dataset.num_scenes()
    print(f"Total scenes: {total}")

    icons = {
        "car": Image.open("icons/car.png"),
        "person": Image.open("icons/person.png"),
        "bicycle": Image.open("icons/bike.png"),
        "bus": Image.open("icons/bus.png"),
        "bicycle_lane": Image.open("icons/bike_lane.png"),
    }

    config_dir = Path("camera_configs")

    dataset.build_camera_array(config_dir)
    dataset.print_camera_info()

    viz = Viz(900, 700, "3D Grid", dataset, icons)
    viz.run()
