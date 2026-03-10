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

from ModelsGBLModule import ModelUpload
from shaderModules import create_shader_program, create_pointcloud_shader_program
from dataLoaderModule import SyncDataset
from labelManager import LabelManager, LabelCameraManager
from poseManager import PoseManager
from UlitysModule import Cube, draw_axes, Grid, PointCloud, ArcCameraControl, ImageTexture, perspective, translate, scale, rotate_x, rotate_y, rotate_z
from CameraLidarModule import CameraLidarModule
from carModelModule import CarModel
from ipmModule import IpmModule, TexturedPlane, HDMapBoundaryAccumulator, HDMapGridAccumulator
from personDetectionModule import PersonDetectionModule
from hdMapIO import HDMapIO, HDMapData, HDMapRenderer

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
        self.show_hdmap_texture = False  # show HD grid plane + all overlays on top of the IPM plane
        self.show_edit_polygon = False  # mapping complete: extract polygon + show spheres for editing
        self._needs_polygon_extract = False  # one-shot: extract when user checks show_edit_polygon
        self.show_hdmap_panel = False  # toggle the HD Map Settings floating window

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

        self._add_polygon_point_request = False
        self._erase_polygon_point_request = False
        self._polygon_mode = None    # None | "add" | "erase"
        self._centerline_mode = None # None | "add" | "erase"
        self._bike_lane_mode = None  # None | "add" | "erase"

        self.show_bike_lane = False
        self.bike_lane_width = 1.5   # metres
        self.cl_ribbon_width = 1.0   # metres (centerline ribbon width)

        self._crosswalk_mode = None  # None | "add" | "erase"
        self.show_crosswalk = False
        self.crosswalk_width = 3.0   # metres

        self._building_mode = None   # None | "add" | "erase"
        self.show_buildings = False

        # HD-map persistence
        self._hdmap_save_path = "hdmap_export.json"
        self._hdmap_save_requested = False
        self._hdmap_load_requested = False
        self._hdmap_status = ""         # last save/load status message
        self.show_hdmap_render = False  # draw the loaded HD-map snapshot

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
        if imgui.button("HD Map Settings"):
            self.show_hdmap_panel = not self.show_hdmap_panel

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

        # ── HD Map Settings floating panel ────────────────────────────────────
        if self.show_hdmap_panel:
            imgui.begin("HD Map Settings", True)

            changed, self.show_ipm = imgui.checkbox("Show IPM", self.show_ipm)
            if changed:
                self._needs_reload = True

            _, self.show_hdmap_texture = imgui.checkbox(
                "Show HD Map Texture", self.show_hdmap_texture
            )

            changed, self.show_edit_polygon = imgui.checkbox(
                "Mapping complete - Edit polygon", self.show_edit_polygon
            )
            if changed and self.show_edit_polygon:
                self._needs_polygon_extract = True

            imgui.separator()
            imgui.text("Polygon Edit")

            if imgui.button("Add Vertex"):
                self._polygon_mode = "add"
                self._centerline_mode = None

            imgui.same_line()

            if imgui.button("Erase Vertex"):
                self._polygon_mode = "erase"
                self._centerline_mode = None

            imgui.separator()
            imgui.text("Centerline Edit")

            cl_w_changed, new_cl_w = imgui.slider_float("CL Width (m)", self.cl_ribbon_width, 0.1, 5.0)
            if cl_w_changed:
                self.cl_ribbon_width = new_cl_w

            if imgui.button("Add CL Vertex"):
                self._centerline_mode = "add"
                self._polygon_mode = None
                self._bike_lane_mode = None

            imgui.same_line()

            if imgui.button("Erase CL Vertex"):
                self._centerline_mode = "erase"
                self._polygon_mode = None
                self._bike_lane_mode = None

            imgui.separator()
            imgui.text("Bike Lane")

            _, self.show_bike_lane = imgui.checkbox("Show Bike Lane", self.show_bike_lane)

            width_changed, new_width = imgui.slider_float("BL Width (m)", self.bike_lane_width, 0.5, 6.0)
            if width_changed:
                self.bike_lane_width = new_width

            if imgui.button("Add BL Vertex"):
                self._bike_lane_mode = "add"
                self._polygon_mode = None
                self._centerline_mode = None

            imgui.same_line()

            if imgui.button("Erase BL Vertex"):
                self._bike_lane_mode = "erase"
                self._polygon_mode = None
                self._centerline_mode = None

            if imgui.button("Store BL Segment"):
                self._bike_lane_mode = "store"

            imgui.same_line()

            if imgui.button("Clear All BL"):
                self._bike_lane_mode = "clear_all"

            # ── Buildings ───────────────────────────────────────────────────
            imgui.separator()
            imgui.text("Buildings")

            _, self.show_buildings = imgui.checkbox("Show Buildings", self.show_buildings)

            if imgui.button("Add Bld Vertex"):
                self._building_mode = "add"
                self._polygon_mode = None
                self._centerline_mode = None
                self._bike_lane_mode = None
                self._crosswalk_mode = None

            imgui.same_line()

            if imgui.button("Erase Bld Vertex"):
                self._building_mode = "erase"
                self._polygon_mode = None
                self._centerline_mode = None
                self._bike_lane_mode = None
                self._crosswalk_mode = None

            if imgui.button("Store Building"):
                self._building_mode = "store"

            imgui.same_line()

            if imgui.button("Clear All Bld"):
                self._building_mode = "clear_all"

            # ── Crosswalk ───────────────────────────────────────────────────
            imgui.separator()
            imgui.text("Crosswalk")

            _, self.show_crosswalk = imgui.checkbox("Show Crosswalks", self.show_crosswalk)

            cw_w_changed, cw_w_new = imgui.slider_float("CW Width (m)", self.crosswalk_width, 1.0, 10.0)
            if cw_w_changed:
                self.crosswalk_width = cw_w_new

            if imgui.button("Add Crosswalk"):
                self._crosswalk_mode = "add"
                self._polygon_mode = None
                self._centerline_mode = None
                self._bike_lane_mode = None

            imgui.same_line()

            if imgui.button("Erase Crosswalk"):
                self._crosswalk_mode = "erase"
                self._polygon_mode = None
                self._centerline_mode = None
                self._bike_lane_mode = None

            if imgui.button("Clear All CW"):
                self._crosswalk_mode = "clear_cw"

            # crosswalk pending indicator
            if self._crosswalk_mode == "add":
                imgui.text_colored("Click map: pt 1 (then pt 2)", 1.0, 0.9, 0.2, 1.0)

            # ── Render loaded snapshot ───────────────────────────────────────
            imgui.separator()
            _, self.show_hdmap_render = imgui.checkbox(
                "Show Loaded HD Map", self.show_hdmap_render
            )

            # ── Save / Load ──────────────────────────────────────────────────
            imgui.separator()
            imgui.text("HD Map File")

            changed, new_path = imgui.input_text("##hdmap_path", self._hdmap_save_path, 512)
            if changed:
                self._hdmap_save_path = new_path

            if imgui.button("Save HD Map"):
                self._hdmap_save_requested = True

            imgui.same_line()

            if imgui.button("Load HD Map"):
                self._hdmap_load_requested = True

            if self._hdmap_status:
                imgui.text_colored(self._hdmap_status, 0.4, 1.0, 0.4, 1.0)

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

    def consume_polygon_mode(self):
        mode = self._polygon_mode
        return mode

    def consume_centerline_mode(self):
        mode = self._centerline_mode
        return mode

    def consume_bike_lane_mode(self):
        mode = self._bike_lane_mode
        return mode

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

        # Standalone renderer for the last loaded / saved HD-map snapshot
        self._hd_map_renderer = HDMapRenderer(
            cl_width=self._ui.cl_ribbon_width,
            bl_width=self._ui.bike_lane_width,
        )

        self._hd_map_renderer.load_bike_lane_icon(icons)

        # Bike-lane visuals (orange centre + left/right boundaries)
        self._bike_lane_center = None          # active segment smooth curve
        self._bike_lane_left = None
        self._bike_lane_right = None
        # stored segments: list of (center, left, right) np.ndarray tuples
        self._bike_lane_stored: list = []
        self._bike_lane_width_prev = self._ui.bike_lane_width

        # Building polygon visuals (smooth closed curves)
        self._bld_active_smooth = None         # active building smooth polygon
        self._bld_stored_smooth: list = []     # list of smooth np.ndarray per stored building

        self._lidar_model = np.identity(4, dtype=np.float32)
        self._basefootprint_model = np.identity(4, dtype=np.float32)
        self.model_ipm_plane = np.identity(4, dtype=np.float32)
        self.model_floor_plane = np.identity(4, dtype=np.float32)
        self._car_model = CarModel(dataset.car_settings)
        self.model_floor_plane = self._car_model.get_basefootprint_frame(self._lidar_model)
        
        self._mouse_pressed = False
        self._mouse_press_pos = None
        self._click_threshold = 5  # pixels
        self._labels_dirty = True

        self._last_size = (0, 0)
        self._proj = None

        self._person_detection_module = PersonDetectionModule()
        self._person_detection_module.load_camera_lidar_parameters(dataset)

        self._model_upload = ModelUpload()
        self._glb_shader = self._model_upload.create_glb_shader()
        self.mesh = self._model_upload.load_glb(dataset.car_model_file)
        

    def _set_model_color(self, model_matrix, r, g, b, a=1.0):
        glUseProgram(self._program)
        glUniformMatrix4fv(self._model_loc, 1, GL_FALSE, model_matrix.T)
        glUniform4f(self._color_loc, r, g, b, a)

    def _ray_to_model_space(self, cam_pos, ray_dir, model):
        """Transform a world-space ray into the local space of *model*."""
        inv_m = np.linalg.inv(model)
        cam_m = (inv_m @ np.append(cam_pos, 1.0))[:3]
        dir_m = (inv_m @ np.append(ray_dir, 0.0))[:3]
        n = np.linalg.norm(dir_m)
        if n > 1e-8:
            dir_m /= n
        return cam_m, dir_m

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

    def _sync_cl_ribbon(self):
        """Push the current _path_center into the purple centerline ribbon renderer."""
        self._hd_grid_acc._cl_ribbon_width = self._ui.cl_ribbon_width
        self._hd_grid_acc.update_cl_ribbon(self._path_center)

    def _refresh_bike_lane(self):
        """Recompute smooth curves + boundaries for active segment and all stored segments."""
        acc = self._hd_grid_acc
        w = self._ui.bike_lane_width

        # Active segment
        if acc._bike_lane_pts is not None and len(acc._bike_lane_pts) >= 2:
            smooth = acc.get_smooth_bike_lane()
            self._bike_lane_center = smooth
            if smooth is not None and len(smooth) >= 2:
                self._bike_lane_left, self._bike_lane_right = \
                    acc.compute_left_right_from_centerline(smooth, w)
            else:
                self._bike_lane_left = self._bike_lane_right = None
            acc.update_bl_active_ribbon(smooth)
        else:
            self._bike_lane_center = None
            self._bike_lane_left = self._bike_lane_right = None
            acc.update_bl_active_ribbon(None)

        # Stored segments
        self._bike_lane_stored = []
        for seg_pts in acc._bike_lane_segments:
            if len(seg_pts) < 2:
                continue
            smooth = acc._interpolate_open_curve(seg_pts)
            if smooth is not None and len(smooth) >= 2:
                left, right = acc.compute_left_right_from_centerline(smooth, w)
            else:
                left = right = None
            self._bike_lane_stored.append((smooth, left, right))

    def _refresh_buildings(self):
        """Recompute smooth closed curves for the active building and all stored ones."""
        acc = self._hd_grid_acc

        # Active building polygon
        if acc._bld_pts is not None and len(acc._bld_pts) >= 3:
            self._bld_active_smooth = acc.get_smooth_building()
        else:
            self._bld_active_smooth = acc._bld_pts  # raw (may be 1-2 pts)

        # Stored buildings
        self._bld_stored_smooth = []
        for seg_pts in acc._bld_segments:
            if len(seg_pts) < 3:
                continue
            smooth = acc._interpolate_closed_curve(seg_pts)
            self._bld_stored_smooth.append(smooth)

    # ------------------------------------------------------------------
    # HD-map persistence helpers
    # ------------------------------------------------------------------

    def _collect_hdmap_data(self) -> HDMapData:
        """Snapshot current editable HD-map state into an HDMapData object."""
        acc = self._hd_grid_acc
        return HDMapData(
            polygons=[p.copy() for p in acc._polygons_editable] if acc._polygons_editable else [],
            centerline=acc._centerline_pts.copy() if acc._centerline_pts is not None else None,
            bike_lane_segments=[s.copy() for s in acc._bike_lane_segments],
            bike_lane_active=acc._bike_lane_pts.copy() if acc._bike_lane_pts is not None else None,
            bike_lane_width=float(self._ui.bike_lane_width),
            crosswalks=[c.copy() for c in acc._crosswalk_pts],
            crosswalk_width=float(self._ui.crosswalk_width),
            buildings=[s.copy() for s in acc._bld_segments] +
                      ([acc._bld_pts.copy()] if acc._bld_pts is not None and len(acc._bld_pts) >= 3 else []),
        )

    def _apply_hdmap_data(self, data: HDMapData) -> None:
        """Restore editable HD-map state from an HDMapData object."""
        acc = self._hd_grid_acc

        # ── Polygons ──────────────────────────────────────────────────────
        if data.polygons:
            valid = [p for p in data.polygons if p is not None and len(p) >= 3]
            acc._polygons_editable = [p.copy() for p in valid]
            acc._polygons_edited = True
            acc.rebuild_spheres_from_editable()
            if acc._edit_snapshot_conf is not None:
                acc.rasterize_edited_polygons_to_grid()
                rgba = acc.to_rgba(alpha_scale=220)
                self._hd_grid_plane.set_texture_rgba(rgba)
            self._polys = acc.get_editable_polygons() or []
            self._ui.show_edit_polygon = True

        # ── Centerline ────────────────────────────────────────────────────
        if data.centerline is not None and len(data.centerline) >= 2:
            acc._centerline_pts = data.centerline.copy()
            acc._centerline_edited = True
            acc.rebuild_centerline_spheres()
            cl_smooth = acc.get_smooth_centerline()
            self._path_center = cl_smooth if cl_smooth is not None else data.centerline.copy()
            self._sync_cl_ribbon()
            if self._polys and self._path_center is not None and len(self._path_center) >= 2:
                left, right = acc.split_polygon_left_right_from_centerline(
                    polygon_world=self._polys[0],
                    centerline_world=self._path_center,
                )
                self._path_left, self._path_right = left, right

        # ── Bike-lane segments ────────────────────────────────────────────
        acc._bike_lane_segments = [
            s.copy() for s in data.bike_lane_segments if s is not None and len(s) >= 2
        ]

        # ── Bike-lane active segment ──────────────────────────────────────
        if data.bike_lane_active is not None and len(data.bike_lane_active) >= 2:
            acc._bike_lane_pts = data.bike_lane_active.copy()
            acc.rebuild_bike_lane_spheres()
        else:
            acc._bike_lane_pts = None
            acc._bike_lane_spheres.build_from_positions_direct([])

        # ── Width + visual refresh ────────────────────────────────────────
        self._ui.bike_lane_width = data.bike_lane_width
        self._bike_lane_width_prev = data.bike_lane_width
        acc._bike_lane_width = data.bike_lane_width
        self._refresh_bike_lane()
        acc.rebuild_bike_lane_ribbons()   # create GPU ribbons for all loaded stored segments

        if data.bike_lane_segments or data.bike_lane_active is not None:
            self._ui.show_bike_lane = True

        # ── Crosswalks ────────────────────────────────────────────────────
        acc._crosswalk_pts = [
            c.copy() for c in data.crosswalks
            if c is not None and np.asarray(c).shape == (2, 3)
        ]
        acc._crosswalk_pending = None
        acc.rebuild_crosswalk_spheres()
        acc._crosswalk_width = data.crosswalk_width
        self._ui.crosswalk_width = data.crosswalk_width
        if acc._crosswalk_pts:
            self._ui.show_crosswalk = True

        # ── Buildings ────────────────────────────────────────────────────
        acc._bld_segments = [
            b.copy() for b in data.buildings
            if b is not None and len(b) >= 3
        ]
        acc._bld_pts = None
        acc._bld_spheres.build_from_positions_direct([])
        self._refresh_buildings()
        if acc._bld_segments:
            self._ui.show_buildings = True

        # Make sure the HD-map panel is visible so the user sees what loaded
        self._ui.show_hdmap_panel = True

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

    def _draw_ipm_layers(self, view: np.ndarray, proj: np.ndarray) -> None:
        self._hd_grid_acc.draw_all_layers(
            view, proj,
            program=self._program,
            set_model_color=self._set_model_color,
            model_ipm_plane=self.model_ipm_plane,
            model_floor_plane=self.model_floor_plane,
            ipm_plane=self._ipm_plane,
            hd_grid_plane=self._hd_grid_plane,
            hd_boundary_accumulator=self._hd_boundary_accumulator,
            polys=self._polys,
            path_center=self._path_center,
            path_left=self._path_left,
            path_right=self._path_right,
            bld_active_smooth=self._bld_active_smooth,
            bld_stored_smooth=self._bld_stored_smooth,
            bike_lane_center=self._bike_lane_center,
            bike_lane_left=self._bike_lane_left,
            bike_lane_right=self._bike_lane_right,
            bike_lane_stored=self._bike_lane_stored,
            show_ipm=self._ui.show_ipm,
            show_hdmap_texture=self._ui.show_hdmap_texture,
            show_bike_lane=self._ui.show_bike_lane,
            show_buildings=self._ui.show_buildings,
            show_crosswalk=self._ui.show_crosswalk,
        )

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

            # Recompute bike-lane boundaries whenever the width slider changes
            if self._ui.show_bike_lane and self._ui.bike_lane_width != self._bike_lane_width_prev:
                self._bike_lane_width_prev = self._ui.bike_lane_width
                self._hd_grid_acc._bike_lane_width = self._ui.bike_lane_width
                self._refresh_bike_lane()
                self._hd_grid_acc.rebuild_bike_lane_ribbons()  # update stored-segment ribbon widths

            # Recompute centerline ribbon whenever its width slider changes
            if self._ui.cl_ribbon_width != self._hd_grid_acc._cl_ribbon_width:
                self._sync_cl_ribbon()

            # Handle immediate bike-lane button actions (store / clear all)
            bl_mode = self._ui._bike_lane_mode
            if bl_mode == "store":
                self._hd_grid_acc.store_bike_lane_segment()
                self._refresh_bike_lane()
                self._ui._bike_lane_mode = None
            elif bl_mode == "clear_all":
                self._hd_grid_acc.clear_all_bike_lane_segments()
                self._refresh_bike_lane()
                self._ui._bike_lane_mode = None

            # Handle immediate building button actions (store / clear all)
            bld_mode = self._ui._building_mode
            if bld_mode == "store":
                self._hd_grid_acc.store_building_segment()
                self._refresh_buildings()
                self._ui._building_mode = None
            elif bld_mode == "clear_all":
                self._hd_grid_acc.clear_all_buildings()
                self._refresh_buildings()
                self._ui._building_mode = None

            # Handle crosswalk width slider changes
            if self._ui.show_crosswalk and self._ui.crosswalk_width != self._hd_grid_acc._crosswalk_width:
                self._hd_grid_acc._crosswalk_width = self._ui.crosswalk_width

            # Handle "Clear All Crosswalks"
            if self._ui._crosswalk_mode == "clear_cw":
                self._hd_grid_acc._crosswalk_pts = []
                self._hd_grid_acc.rebuild_crosswalk_spheres()
                self._ui._crosswalk_mode = None

            # Handle HD-map save / load requests
            if self._ui._hdmap_save_requested:
                self._ui._hdmap_save_requested = False
                try:
                    data = self._collect_hdmap_data()
                    HDMapIO.save(self._ui._hdmap_save_path, data)
                    self._hd_map_renderer.update(
                        data,
                        cl_width=self._ui.cl_ribbon_width,
                        bl_width=self._ui.bike_lane_width,
                    )
                    self._ui._hdmap_status = f"Saved  ({HDMapIO.summary(data)})"
                except Exception as exc:
                    # HDMapRenderer.update() may have left the GL program unbound
                    # if it raised mid-execution; restore it so the render loop
                    # can continue without GL_INVALID_OPERATION errors.
                    glUseProgram(self._program)
                    self._ui._hdmap_status = f"Save error: {exc}"

            if self._ui._hdmap_load_requested:
                self._ui._hdmap_load_requested = False
                try:
                    data = HDMapIO.load(self._ui._hdmap_save_path)
                    self._apply_hdmap_data(data)
                    self._hd_map_renderer.update(
                        data,
                        cl_width=self._ui.cl_ribbon_width,
                        bl_width=self._ui.bike_lane_width,
                    )
                    self._ui.show_hdmap_render = True   # auto-show after load
                    self._ui._hdmap_status = f"Loaded  ({HDMapIO.summary(data)})"
                except Exception as exc:
                    glUseProgram(self._program)
                    self._ui._hdmap_status = f"Load error: {exc}"

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
                        best_t    = float("inf")
                        best_kind = None
                        best_idx  = -1

                        cam_pos_m, ray_dir_m = self._ray_to_model_space(cam_pos, ray_dir, self.model_floor_plane)

                        if self._ui.show_edit_polygon:
                            poly_res = self._hd_grid_acc._polygon_spheres.intersect_ray(cam_pos_m, ray_dir_m)
                            cl_res   = self._hd_grid_acc._centerline_spheres.intersect_ray(cam_pos_m, ray_dir_m)
                            if poly_res is not None and poly_res[1] < best_t:
                                best_t = poly_res[1]; best_kind = "polygon"; best_idx = poly_res[0]
                            if cl_res is not None and cl_res[1] < best_t:
                                best_t = cl_res[1]; best_kind = "centerline"; best_idx = cl_res[0]

                        if self._ui.show_bike_lane:
                            bl_res = self._hd_grid_acc._bike_lane_spheres.intersect_ray(cam_pos_m, ray_dir_m)
                            if bl_res is not None and bl_res[1] < best_t:
                                best_t = bl_res[1]; best_kind = "bike_lane"; best_idx = bl_res[0]

                        if self._ui.show_crosswalk:
                            cw_res = self._hd_grid_acc._crosswalk_spheres.intersect_ray(cam_pos_m, ray_dir_m)
                            if cw_res is not None and cw_res[1] < best_t:
                                best_t = cw_res[1]; best_kind = "crosswalk"; best_idx = cw_res[0]

                        if self._ui.show_buildings:
                            bld_res = self._hd_grid_acc._bld_spheres.intersect_ray(cam_pos_m, ray_dir_m)
                            if bld_res is not None and bld_res[1] < best_t:
                                best_t = bld_res[1]; best_kind = "building"; best_idx = bld_res[0]

                        if best_kind == "polygon":
                            self._hd_grid_acc._polygon_spheres.select(best_idx)
                            self._hd_grid_acc._polygon_spheres.begin_drag(cam_pos_m, ray_dir_m)
                        elif best_kind == "centerline":
                            self._hd_grid_acc._centerline_spheres.select(best_idx)
                            self._hd_grid_acc._centerline_spheres.begin_drag(cam_pos_m, ray_dir_m)
                        elif best_kind == "bike_lane":
                            self._hd_grid_acc._bike_lane_spheres.select(best_idx)
                            self._hd_grid_acc._bike_lane_spheres.begin_drag(cam_pos_m, ray_dir_m)
                        elif best_kind == "crosswalk":
                            self._hd_grid_acc._crosswalk_spheres.select(best_idx)
                            self._hd_grid_acc._crosswalk_spheres.begin_drag(cam_pos_m, ray_dir_m)
                        elif best_kind == "building":
                            self._hd_grid_acc._bld_spheres.select(best_idx)
                            self._hd_grid_acc._bld_spheres.begin_drag(cam_pos_m, ray_dir_m)

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

                            handled_polygon_mode = False
                            if self._ui.show_edit_polygon:
                                polygon_mode = self._ui.consume_polygon_mode()

                                if polygon_mode is not None:

                                    if polygon_mode == "add":
                                        if abs(ray_dir[2]) > 1e-6:
                                            t = -cam_pos[2] / ray_dir[2]
                                            hit = cam_pos + t * ray_dir
                                        else:
                                            hit = None

                                        if hit is not None:
                                            self._hd_grid_acc.add_vertex_to_selected_polygon(hit)

                                    elif polygon_mode == "erase":
                                        self._hd_grid_acc.erase_selected_vertex()

                                    self._hd_grid_acc.rebuild_spheres_from_editable()
                                    self._hd_grid_acc.rasterize_edited_polygons_to_grid()
                                    # upload updated grid texture so the green fill reflects the new polygon
                                    rgba = self._hd_grid_acc.to_rgba(alpha_scale=220)
                                    self._hd_grid_plane.set_texture_rgba(rgba)
                                    # refresh green polygon line and left/right path lines
                                    self._polys = self._hd_grid_acc.get_editable_polygons() or []
                                    if self._polys and self._pose.path_positions is not None and len(self._pose.path_positions) >= 2:
                                        left, right = self._hd_grid_acc.split_polygon_left_right_from_centerline(
                                            polygon_world=self._polys[0],
                                            centerline_world=np.asarray(self._pose.path_positions, dtype=np.float32)
                                        )
                                        self._path_left, self._path_right = left, right
                                    else:
                                        self._path_left, self._path_right = None, None

                                    self._ui._polygon_mode = None
                                    handled_polygon_mode = True

                            # --- Centerline add / erase ---
                            if not handled_polygon_mode and self._ui.show_edit_polygon:
                                centerline_mode = self._ui.consume_centerline_mode()

                                if centerline_mode is not None:

                                    if centerline_mode == "add":
                                        if abs(ray_dir[2]) > 1e-6:
                                            t = -cam_pos[2] / ray_dir[2]
                                            hit = cam_pos + t * ray_dir
                                        else:
                                            hit = None
                                        if hit is not None:
                                            self._hd_grid_acc.add_vertex_to_centerline(hit)

                                    elif centerline_mode == "erase":
                                        self._hd_grid_acc.erase_selected_centerline_vertex()

                                    self._hd_grid_acc.rebuild_centerline_spheres()
                                    # recompute smooth line and left/right split
                                    cl_smooth = self._hd_grid_acc.get_smooth_centerline()
                                    if cl_smooth is not None:
                                        self._path_center = cl_smooth
                                    self._sync_cl_ribbon()
                                    if self._polys and self._path_center is not None and len(self._path_center) >= 2:
                                        left, right = self._hd_grid_acc.split_polygon_left_right_from_centerline(
                                            polygon_world=self._polys[0],
                                            centerline_world=self._path_center
                                        )
                                        self._path_left, self._path_right = left, right
                                    else:
                                        self._path_left, self._path_right = None, None

                                    self._ui._centerline_mode = None
                                    handled_polygon_mode = True

                                # Bike-lane add / erase
                                bike_lane_mode = self._ui.consume_bike_lane_mode()
                                if bike_lane_mode is not None and self._ui.show_bike_lane:
                                    if bike_lane_mode == "add":
                                        if abs(ray_dir[2]) > 1e-6:
                                            t = -cam_pos[2] / ray_dir[2]
                                            hit = cam_pos + t * ray_dir
                                        else:
                                            hit = None
                                        if hit is not None:
                                            self._hd_grid_acc.add_vertex_to_bike_lane(hit)

                                    elif bike_lane_mode == "erase":
                                        self._hd_grid_acc.erase_selected_bike_lane_vertex()

                                    self._hd_grid_acc.rebuild_bike_lane_spheres()
                                    self._refresh_bike_lane()
                                    self._ui._bike_lane_mode = None
                                    handled_polygon_mode = True

                                # Building add / erase
                                bld_mode_click = self._ui._building_mode
                                if bld_mode_click in ("add", "erase") and self._ui.show_buildings:
                                    if bld_mode_click == "add":
                                        if abs(ray_dir[2]) > 1e-6:
                                            t_gnd = -cam_pos[2] / ray_dir[2]
                                            hit = cam_pos + t_gnd * ray_dir
                                        else:
                                            hit = None
                                        if hit is not None:
                                            self._hd_grid_acc.add_vertex_to_building(hit)

                                    elif bld_mode_click == "erase":
                                        self._hd_grid_acc.erase_selected_building_vertex()

                                    self._hd_grid_acc.rebuild_building_spheres()
                                    self._refresh_buildings()
                                    self._ui._building_mode = None
                                    handled_polygon_mode = True

                                # Crosswalk add / erase
                                cw_mode = self._ui._crosswalk_mode
                                if cw_mode is not None and self._ui.show_crosswalk and cw_mode in ("add", "erase"):
                                    if cw_mode == "add":
                                        if abs(ray_dir[2]) > 1e-6:
                                            t_gnd = -cam_pos[2] / ray_dir[2]
                                            hit = cam_pos + t_gnd * ray_dir
                                        else:
                                            hit = None
                                        if hit is not None:
                                            completed = self._hd_grid_acc.add_crosswalk_point(hit)
                                            if completed:
                                                self._ui._crosswalk_mode = None
                                            # else: keep "add" mode — waiting for second click

                                    elif cw_mode == "erase":
                                        cam_pos_m, ray_dir_m = self._ray_to_model_space(cam_pos, ray_dir, self.model_floor_plane)
                                        cw_res = self._hd_grid_acc._crosswalk_spheres.intersect_ray(cam_pos_m, ray_dir_m)
                                        if cw_res is not None:
                                            self._hd_grid_acc._crosswalk_spheres.select(cw_res[0])
                                            self._hd_grid_acc.erase_selected_crosswalk()
                                        self._ui._crosswalk_mode = None

                                    handled_polygon_mode = True

                            if not handled_polygon_mode:

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

                                # polygon sphere pick — only when edit-polygon mode
                                if self._ui.show_edit_polygon:
                                    cam_pos_m, ray_dir_m = self._ray_to_model_space(cam_pos, ray_dir, self.model_floor_plane)
                                    polygon_result = self._hd_grid_acc._polygon_spheres.intersect_ray(cam_pos_m, ray_dir_m)
                                    if polygon_result is not None:
                                        polygon_idx, polygon_t = polygon_result
                                    else:
                                        polygon_idx, polygon_t = None, float("inf")
                                else:
                                    polygon_idx, polygon_t = None, float("inf")

                                # selection priority
                                if polygon_idx is not None and polygon_t < min_label_t:
                                    self._hd_grid_acc._polygon_spheres.select(polygon_idx)
                                elif closest_label is not None:
                                    self._labels.select(closest_label)
                                    self._labels_dirty = True

                    # Finish polygon drag (if any) + bake grid once on release (only when releasing)
                    if not self._ui._live_topic_mode:
                        drag_idx = self._hd_grid_acc._polygon_spheres._selected_index
                        was_dragging = self._hd_grid_acc._polygon_spheres._dragging
                        self._hd_grid_acc._polygon_spheres.end_drag()

                        cl_drag_idx = self._hd_grid_acc._centerline_spheres._selected_index
                        was_cl_dragging = self._hd_grid_acc._centerline_spheres._dragging
                        self._hd_grid_acc._centerline_spheres.end_drag()

                        bl_drag_idx = self._hd_grid_acc._bike_lane_spheres._selected_index
                        was_bl_dragging = self._hd_grid_acc._bike_lane_spheres._dragging
                        self._hd_grid_acc._bike_lane_spheres.end_drag()

                        cw_drag_idx = self._hd_grid_acc._crosswalk_spheres._selected_index
                        was_cw_dragging = self._hd_grid_acc._crosswalk_spheres._dragging
                        self._hd_grid_acc._crosswalk_spheres.end_drag()

                        bld_drag_idx = self._hd_grid_acc._bld_spheres._selected_index
                        was_bld_dragging = self._hd_grid_acc._bld_spheres._dragging
                        self._hd_grid_acc._bld_spheres.end_drag()

                        if self._ui.show_edit_polygon and was_dragging and drag_idx >= 0:
                            self._hd_grid_acc.sync_sphere_to_polygon(drag_idx)
                            self._hd_grid_acc.rasterize_edited_polygons_to_grid()
                            rgba = self._hd_grid_acc.to_rgba(alpha_scale=220)
                            self._hd_grid_plane.set_texture_rgba(rgba)
                            self._polys = self._hd_grid_acc.get_editable_polygons() or []
                            centerline = self._path_center if self._path_center is not None and len(self._path_center) >= 2 else None
                            if self._polys and centerline is not None:
                                left, right = self._hd_grid_acc.split_polygon_left_right_from_centerline(
                                    polygon_world=self._polys[0],
                                    centerline_world=centerline
                                )
                                self._path_left, self._path_right = left, right
                            else:
                                self._path_left, self._path_right = None, None

                        if self._ui.show_edit_polygon and was_cl_dragging and cl_drag_idx >= 0:
                            self._hd_grid_acc.sync_centerline_sphere(cl_drag_idx)
                            cl_smooth = self._hd_grid_acc.get_smooth_centerline()
                            if cl_smooth is not None:
                                self._path_center = cl_smooth
                            self._sync_cl_ribbon()
                            if self._polys and self._path_center is not None and len(self._path_center) >= 2:
                                left, right = self._hd_grid_acc.split_polygon_left_right_from_centerline(
                                    polygon_world=self._polys[0],
                                    centerline_world=self._path_center
                                )
                                self._path_left, self._path_right = left, right
                            else:
                                self._path_left, self._path_right = None, None

                        if self._ui.show_bike_lane and was_bl_dragging and bl_drag_idx >= 0:
                            self._hd_grid_acc.sync_bike_lane_sphere(bl_drag_idx)
                            self._refresh_bike_lane()

                        if self._ui.show_crosswalk and was_cw_dragging and cw_drag_idx >= 0:
                            self._hd_grid_acc.sync_crosswalk_sphere(cw_drag_idx)

                        if self._ui.show_buildings and was_bld_dragging and bld_drag_idx >= 0:
                            self._hd_grid_acc.sync_building_sphere(bld_drag_idx)
                            self._refresh_buildings()

                    self._mouse_pressed = False

                # -----------------------------
                # DRAG behavior (either sphere drag OR camera orbit/pan)
                # -----------------------------
                if left_pressed:
                    if self._hd_grid_acc._polygon_spheres._dragging and not self._ui._live_topic_mode and self._ui.show_edit_polygon:
                        # Drag a polygon boundary sphere
                        inv_proj = np.linalg.inv(proj)
                        inv_view = np.linalg.inv(view)
                        cam_pos_world = inv_view[:3, 3]

                        cam_pos, ray_dir = self.screen_to_world_ray(
                            x, y, width, height,
                            inv_proj, inv_view, cam_pos_world
                        )
                        cam_pos_m, ray_dir_m = self._ray_to_model_space(cam_pos, ray_dir, self.model_floor_plane)

                        self._hd_grid_acc._polygon_spheres.drag(cam_pos_m, ray_dir_m)
                        idx = self._hd_grid_acc._polygon_spheres._selected_index
                        if idx >= 0:
                            self._hd_grid_acc.sync_sphere_to_polygon(idx)
                            self._polys = self._hd_grid_acc.get_editable_polygons() or []
                            centerline = self._path_center if self._path_center is not None and len(self._path_center) >= 2 else None
                            if self._polys and centerline is not None:
                                left, right = self._hd_grid_acc.split_polygon_left_right_from_centerline(
                                    polygon_world=self._polys[0],
                                    centerline_world=centerline
                                )
                                self._path_left, self._path_right = left, right
                            else:
                                self._path_left, self._path_right = None, None

                    elif self._hd_grid_acc._centerline_spheres._dragging and not self._ui._live_topic_mode and self._ui.show_edit_polygon:
                        # Drag a centerline sphere
                        inv_proj = np.linalg.inv(proj)
                        inv_view = np.linalg.inv(view)
                        cam_pos_world = inv_view[:3, 3]

                        cam_pos, ray_dir = self.screen_to_world_ray(
                            x, y, width, height,
                            inv_proj, inv_view, cam_pos_world
                        )
                        cam_pos_m, ray_dir_m = self._ray_to_model_space(cam_pos, ray_dir, self.model_floor_plane)

                        self._hd_grid_acc._centerline_spheres.drag(cam_pos_m, ray_dir_m)
                        idx = self._hd_grid_acc._centerline_spheres._selected_index
                        if idx >= 0:
                            self._hd_grid_acc.sync_centerline_sphere(idx)
                            cl_smooth = self._hd_grid_acc.get_smooth_centerline()
                            if cl_smooth is not None:
                                self._path_center = cl_smooth
                            self._sync_cl_ribbon()
                            if self._polys and self._path_center is not None and len(self._path_center) >= 2:
                                left, right = self._hd_grid_acc.split_polygon_left_right_from_centerline(
                                    polygon_world=self._polys[0],
                                    centerline_world=self._path_center
                                )
                                self._path_left, self._path_right = left, right
                            else:
                                self._path_left, self._path_right = None, None

                    elif self._hd_grid_acc._bike_lane_spheres._dragging and not self._ui._live_topic_mode and self._ui.show_bike_lane:
                        # Drag a bike-lane sphere
                        inv_proj = np.linalg.inv(proj)
                        inv_view = np.linalg.inv(view)
                        cam_pos_world = inv_view[:3, 3]

                        cam_pos, ray_dir = self.screen_to_world_ray(
                            x, y, width, height,
                            inv_proj, inv_view, cam_pos_world
                        )
                        cam_pos_m, ray_dir_m = self._ray_to_model_space(cam_pos, ray_dir, self.model_floor_plane)

                        self._hd_grid_acc._bike_lane_spheres.drag(cam_pos_m, ray_dir_m)
                        idx = self._hd_grid_acc._bike_lane_spheres._selected_index
                        if idx >= 0:
                            self._hd_grid_acc.sync_bike_lane_sphere(idx)
                            self._refresh_bike_lane()

                    elif self._hd_grid_acc._crosswalk_spheres._dragging and not self._ui._live_topic_mode and self._ui.show_crosswalk:
                        # Drag a crosswalk endpoint sphere
                        inv_proj = np.linalg.inv(proj)
                        inv_view = np.linalg.inv(view)
                        cam_pos_world = inv_view[:3, 3]

                        cam_pos, ray_dir = self.screen_to_world_ray(
                            x, y, width, height,
                            inv_proj, inv_view, cam_pos_world
                        )
                        cam_pos_m, ray_dir_m = self._ray_to_model_space(cam_pos, ray_dir, self.model_floor_plane)

                        self._hd_grid_acc._crosswalk_spheres.drag(cam_pos_m, ray_dir_m)
                        idx = self._hd_grid_acc._crosswalk_spheres._selected_index
                        if idx >= 0:
                            self._hd_grid_acc.sync_crosswalk_sphere(idx)

                    elif self._hd_grid_acc._bld_spheres._dragging and not self._ui._live_topic_mode and self._ui.show_buildings:
                        # Drag a building polygon sphere
                        inv_proj = np.linalg.inv(proj)
                        inv_view = np.linalg.inv(view)
                        cam_pos_world = inv_view[:3, 3]

                        cam_pos, ray_dir = self.screen_to_world_ray(
                            x, y, width, height,
                            inv_proj, inv_view, cam_pos_world
                        )
                        cam_pos_m, ray_dir_m = self._ray_to_model_space(cam_pos, ray_dir, self.model_floor_plane)

                        self._hd_grid_acc._bld_spheres.drag(cam_pos_m, ray_dir_m)
                        idx = self._hd_grid_acc._bld_spheres._selected_index
                        if idx >= 0:
                            self._hd_grid_acc.sync_building_sphere(idx)
                            self._refresh_buildings()

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

                    # ESC: release polygon edit (deselect sphere, end drag) — only when edit-polygon mode
                    if self._ui.show_edit_polygon and glfw.get_key(self._window, glfw.KEY_ESCAPE) == glfw.PRESS:
                        self._hd_grid_acc._polygon_spheres.end_drag()
                        self._hd_grid_acc._polygon_spheres.select(-1)
                        self._hd_grid_acc.clear_polygon_edits()
                        self._hd_grid_acc._centerline_spheres.end_drag()
                        self._hd_grid_acc._centerline_spheres.select(-1)
                        self._hd_grid_acc._centerline_edited = False
                        self._ui._polygon_mode = None
                        self._ui._centerline_mode = None

                    if self._ui.show_bike_lane and glfw.get_key(self._window, glfw.KEY_ESCAPE) == glfw.PRESS:
                        self._hd_grid_acc._bike_lane_spheres.end_drag()
                        self._hd_grid_acc._bike_lane_spheres.select(-1)
                        self._ui._bike_lane_mode = None

                    if self._ui.show_crosswalk and glfw.get_key(self._window, glfw.KEY_ESCAPE) == glfw.PRESS:
                        # Cancel crosswalk placement (pending first point is also cleared)
                        self._hd_grid_acc._crosswalk_pending = None
                        self._hd_grid_acc._rebuild_cw_spheres(include_pending=False)
                        self._ui._crosswalk_mode = None

                    if self._ui.show_buildings and glfw.get_key(self._window, glfw.KEY_ESCAPE) == glfw.PRESS:
                        self._hd_grid_acc._bld_spheres.end_drag()
                        self._hd_grid_acc._bld_spheres.select(-1)
                        self._ui._building_mode = None

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

                    bev, valid_mask, bev_drivable = self._ipm_module.warp_images(self._current_images, self._current_images_mask)

                    self._ipm_plane.set_texture(bev, valid_mask)

                    if bev_drivable is not None:
                        # poligon set
                        self._hd_boundary_accumulator.update(bev_drivable, self.model_ipm_plane)
                        # grid accumulation (always; polygon extraction only when checkbox on)
                        self._hd_grid_acc.update(
                            bev_drivable,
                            bev,                 # <-- pass colored BEV image
                            self.model_ipm_plane
                        )

                        rgba = self._hd_grid_acc.to_rgba(alpha_scale=220)
                        self._hd_grid_plane.set_texture_rgba(rgba)

                        center = np.array(self._pose.path_positions, dtype=np.float32)
                        if center.ndim == 1:
                            center = center.reshape(-1, 2)
                        if center.shape[1] == 2:
                            center = np.concatenate([center, np.zeros((len(center), 1), dtype=np.float32)], axis=1)

                        if self._ui.show_edit_polygon:
                            if self._hd_grid_acc._polygons_edited:
                                self._polys = self._hd_grid_acc.get_editable_polygons() or []
                            else:
                                self._polys = self._hd_grid_acc.extract_polygons()
                                self._hd_grid_acc.update_polygon_spheres(self._polys)
                            # reset centerline spheres only if the user hasn't manually edited them
                            if not self._hd_grid_acc._centerline_edited:
                                self._hd_grid_acc.update_centerline_spheres(center)
                            _cl_smooth = self._hd_grid_acc.get_smooth_centerline()
                            self._path_center = _cl_smooth if _cl_smooth is not None else center
                            self._sync_cl_ribbon()
                            left, right = None, None
                            if self._polys and len(self._path_center) >= 2:
                                poly = self._polys[0]
                                left, right = self._hd_grid_acc.split_polygon_left_right_from_centerline(
                                    polygon_world=poly,
                                    centerline_world=self._path_center
                                )
                            self._path_left = left
                            self._path_right = right
                        else:
                            self._polys = []
                            self._path_center = center
                            self._sync_cl_ribbon()
                            self._path_left = None
                            self._path_right = None

                    # person detection
                    self._person_detection_module.get_camera_images(self._current_images)

                # When user checks "Edit polygon" without reloading: extract once from current grid
                if self._ui._needs_polygon_extract and self._ui.show_edit_polygon and self._hd_grid_plane.has_texture:
                    self._ui._needs_polygon_extract = False
                    if self._hd_grid_acc._polygons_edited:
                        self._polys = self._hd_grid_acc.get_editable_polygons() or []
                    else:
                        self._polys = self._hd_grid_acc.extract_polygons()
                        self._hd_grid_acc.update_polygon_spheres(self._polys)
                    left, right = None, None
                    if self._polys and self._pose.path_positions is not None and len(self._pose.path_positions) >= 2:
                        center = np.array(self._pose.path_positions, dtype=np.float32)
                        if center.ndim == 1:
                            center = center.reshape(-1, 2)
                        if center.shape[1] == 2:
                            center = np.concatenate([center, np.zeros((len(center), 1), dtype=np.float32)], axis=1)
                        if not self._hd_grid_acc._centerline_edited:
                            self._hd_grid_acc.update_centerline_spheres(center)
                        _cl_smooth = self._hd_grid_acc.get_smooth_centerline()
                        self._path_center = _cl_smooth if _cl_smooth is not None else center
                        self._sync_cl_ribbon()
                        left, right = self._hd_grid_acc.split_polygon_left_right_from_centerline(
                            polygon_world=self._polys[0],
                            centerline_world=self._path_center
                        )
                        self._path_left, self._path_right = left, right
                    else:
                        self._path_left = None
                        self._path_right = None

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

                # IPM plane + HD grid overlays
                self._draw_ipm_layers(view, proj)

                # Loaded HD-map snapshot — independent of IPM / grid texture
                if self._ui.show_hdmap_render:
                    self._hd_map_renderer.draw(view.T, proj.T, model=self.model_floor_plane.T)
                    glUseProgram(self._program)  # restore main program after renderer

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

                prev = glGetIntegerv(GL_CURRENT_PROGRAM)

                glUseProgram(self._glb_shader)

                self._model_upload.view_matrix = view.T
                self._model_upload.proj_matrix = proj.T

                model_transform = self._basefootprint_model @ rotate_x(-math.pi/2)

                self.mesh.local_transform = model_transform.T

                self._model_upload.render_glb_mesh(self.mesh, self._glb_shader)

                glUseProgram(prev)

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
