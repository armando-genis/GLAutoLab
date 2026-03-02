import numpy as np
import cv2
import yaml
import argparse
from tqdm import tqdm
import os
from dataclasses import dataclass
from typing import Optional, List
from CameraModule import IpmCameraConfig
from PathRendererModule import PathSphereMarkerRenderer
from typing import List, Tuple, Optional


from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

class IpmModule:
    def __init__(self, dataset):
        self.dataset = dataset

        self.interpMode = cv2.INTER_LINEAR
        self.ipm_camera_configs = self.dataset.load_ipm_camera_configs()

        self.drone_camera = None
        self.set_drone_camera()

        self.input_resolution = None
        self.set_input_resolution()

        self.outputRes = None
        self.pxPerM = None
        self.IPMs = []
        self.masks = []

        self.M = None
        self.calculate_output_shape()

        # print(f"M: {self.M}")

        self.calculate_ipm()
        self.invalid_mask()

        self.mask_resolution = self.dataset.mask_resolution

    def set_input_resolution(self):
        # get the first image from the dataset (samples are keyed by scene id, not 0-based index)
        indices = self.dataset.indices()
        if not indices:
            raise ValueError("Dataset has no samples")
        images = self.dataset.load_images(indices[0])
        # load_images returns dict of cam_name -> array; use first image for resolution
        first_img = next(iter(images.values()))
        self.input_resolution = first_img.shape[:2]
        # print(f"Input resolution: {self.input_resolution}")

    def set_drone_camera(self):
        # drone camera parameters
        # camera calibration matrix K
        fx = 682.578
        fy = 682.578
        px = 482.0
        py = 302.0

        # rotation matrix R (in deg) (drone is assumed to be at 90° pitch, 90° roll)
        yaw = 0.0
        pitch = 90.0
        roll = 90.0

        # vehicle coords of camera origin
        XCam = 0.0
        YCam = 0.0
        ZCam = 40.0

        self.drone_camera = IpmCameraConfig(fx, fy, px, py, yaw, pitch, roll, XCam, YCam, ZCam)

    def calculate_output_shape(self):
        c = self.drone_camera
        # Read from IpmCameraConfig: K has fx, fy, px, py; position from t = -R @ [XCam, YCam, ZCam]
        fx, fy = c.K[0, 0], c.K[1, 1]
        px, py = c.K[0, 2], c.K[1, 2]
        x_cam, y_cam, z_cam = (-c.R.T @ c.t).flatten()

        self.outputRes = (int(2 * py), int(2 * px))
        dx = self.outputRes[1] / fx * z_cam
        dy = self.outputRes[0] / fy * z_cam
        self.pxPerM = (self.outputRes[0] / dy, self.outputRes[1] / dx)

        # setup mapping from street/top-image plane to world coords (use outputRes so center = principal point, same as drone_camera.py)
        shift = (self.outputRes[0] / 2.0, self.outputRes[1] / 2.0)
        shift = shift[0] + y_cam * self.pxPerM[0], shift[1] - x_cam * self.pxPerM[1]
        self.M = np.array([
            [1.0 / self.pxPerM[1], 0.0, -shift[1] / self.pxPerM[1]],
            [0.0, -1.0 / self.pxPerM[0], shift[0] / self.pxPerM[0]],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

    def calculate_ipm(self):
        for config in self.ipm_camera_configs:
            if config is None:
                continue
            ipm = np.linalg.inv(config.P.dot(self.M))
            config.homography_matrix = ipm
            self.IPMs.append(ipm)
            print(f"OpenCV homography:")
            print(ipm.tolist())

    def invalid_mask(self):
        x_cam, y_cam, _ = (-self.drone_camera.R.T @ self.drone_camera.t).flatten()

        j_coords = np.arange(self.outputRes[0])
        i_coords = np.arange(self.outputRes[1])
        i_grid, j_grid = np.meshgrid(i_coords, j_coords)

        y_offset = -j_grid + self.outputRes[0] / 2 - y_cam * self.pxPerM[0]
        x_offset = i_grid - self.outputRes[1] / 2 + x_cam * self.pxPerM[1]
        theta = np.rad2deg(np.arctan2(y_offset, x_offset))

        self.masks = [None] * len(self.ipm_camera_configs)

        for i, config in enumerate(self.ipm_camera_configs):
            if config is None:
                continue

            yaw = config.yaw
            diff = (theta - yaw + 180) % 360 - 180
            diff = np.abs(diff)

            mask_2d = diff > 90
            mask = np.stack([mask_2d, mask_2d, mask_2d], axis=-1)

            self.masks[i] = mask

    def warp_images(self, images: list[np.ndarray | None],
                    images_mask: list[np.ndarray | None] | None = None):

        warped_images = []
        warped_masks = []
        if images_mask is not None:
            bev_sidewalk = np.zeros((self.outputRes[0], self.outputRes[1]), dtype=np.uint8)
        else:
            bev_sidewalk = None

        for i, config in enumerate(self.ipm_camera_configs):

            if config is None:
                continue

            if i >= len(images) or images[i] is None:
                continue

            img = images[i]

            # Optional overlay (for visualization only)
            if images_mask is not None and i < len(images_mask) and images_mask[i] is not None:
                color_mask, mask_resized = self.colorize_mask(images_mask[i])
                # img = self.overlay_mask(img, color_mask, alpha=0.4)

                sidewalk_mask = (mask_resized == 1).astype(np.uint8)

                warped_mask = cv2.warpPerspective(
                    sidewalk_mask,
                    config.homography_matrix,
                    (self.outputRes[1], self.outputRes[0]),
                    flags=cv2.INTER_NEAREST
                )

                # Apply invalid mask aligned by camera index
                if i < len(self.masks) and self.masks[i] is not None:
                    warped_mask[self.masks[i][..., 0]] = 0

                # Fuse using OR
                bev_sidewalk = np.maximum(bev_sidewalk, warped_mask)

                warped_masks.append(warped_mask)
                
            # Warp to BEV
            warped = cv2.warpPerspective(
                img,
                config.homography_matrix,
                (self.outputRes[1], self.outputRes[0]),
                flags=self.interpMode
            )

            # Apply invalid mask (aligned by camera index)
            if i < len(self.masks) and self.masks[i] is not None:
                warped[self.masks[i]] = 0

            warped_images.append(warped)

        if not warped_images:
            return None, None, None

        # ---------- Stitch ----------
        H, W = self.outputRes
        birdsEyeView = np.zeros((H, W, 3), dtype=np.float32)
        count = np.zeros((H, W), dtype=np.float32)

        for warped in warped_images:
            valid = np.any(warped != 0, axis=-1)
            birdsEyeView[valid] += warped[valid]
            count[valid] += 1

        valid_mask_final = count > 0
        count[count == 0] = 1  # avoid div0

        birdsEyeView = birdsEyeView / count[..., None]
        birdsEyeView = np.clip(birdsEyeView, 0, 255).astype(np.uint8)


        if not warped_masks:
            return birdsEyeView, valid_mask_final, None

        birdsEyeView_drivable = (bev_sidewalk > 0).astype(np.uint8) * 255

        return birdsEyeView, valid_mask_final, birdsEyeView_drivable

    def colorize_mask(self, mask: np.ndarray):
        """
        Convert class-id mask into BGR color mask.
        """
        # Resize mask to input resolution
        mask_resized = cv2.resize(
            mask,
            (self.input_resolution[1], self.input_resolution[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Create empty color image
        color_mask = np.zeros((mask_resized.shape[0], mask_resized.shape[1], 3), dtype=np.uint8)

        # Class 1: sidewalk (BGR: [255,0,0])
        color_mask[mask_resized == 1] = (255, 0, 0)

        # Class 2: grass (BGR: [0,255,0])
        color_mask[mask_resized == 2] = (0, 255, 0)

        return color_mask, mask_resized

    def overlay_mask(self, image: np.ndarray, color_mask: np.ndarray, alpha=0.4):
        """
        Blend color mask over image.
        """
        return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

class TexturedPlane:
    """
    Metric ground plane with texture (IPM BEV).
    - Shader compiled internally
    - Own VAO/VBO
    - Own texture
    - Draws on Z=0 plane
    """

    VERTEX_SHADER = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTex;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    out vec2 TexCoord;

    void main() {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        TexCoord = aTex;
    }
    """

    FRAGMENT_SHADER = """
    #version 330 core
    in vec2 TexCoord;
    out vec4 FragColor;

    uniform sampler2D u_texture;
    uniform float u_alpha;

    void main() {
        vec4 texColor = texture(u_texture, TexCoord);

        // if texture alpha == 0 → discard fragment
        if (texColor.a < 0.01)
            discard;

        FragColor = vec4(texColor.rgb, texColor.a * u_alpha);
    }
    """

    def __init__(self, alpha=0.4, flip_y=True):
        self.flip_y = bool(flip_y)
        self.alpha = float(alpha)
        # ---------- Shader ----------
        self.shader = compileProgram(
            compileShader(self.VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(self.FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )

        glUseProgram(self.shader)

        self.loc_model = glGetUniformLocation(self.shader, "model")
        self.loc_view = glGetUniformLocation(self.shader, "view")
        self.loc_proj = glGetUniformLocation(self.shader, "projection")
        self.loc_tex = glGetUniformLocation(self.shader, "u_texture")
        self.loc_alpha = glGetUniformLocation(self.shader, "u_alpha")

        # ---------- Geometry (unit quad centered at origin) ----------
        # 2 triangles (pos3 + uv2)
        self.vertices = np.array([
            -0.5, -0.5, 0.0,  0.0, 0.0,
             0.5, -0.5, 0.0,  1.0, 0.0,
             0.5,  0.5, 0.0,  1.0, 1.0,

            -0.5, -0.5, 0.0,  0.0, 0.0,
             0.5,  0.5, 0.0,  1.0, 1.0,
            -0.5,  0.5, 0.0,  0.0, 1.0,
        ], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes,
                     self.vertices, GL_STATIC_DRAW)

        stride = 5 * 4

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                              stride, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                              stride, ctypes.c_void_p(12))

        glBindVertexArray(0)

        # ---------- Texture ----------
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glBindTexture(GL_TEXTURE_2D, 0)

        self.has_texture = False


    # --------------------------------------------------
    # Texture upload / update
    # --------------------------------------------------
    def set_texture(self, image_rgb: np.ndarray, valid_mask: np.ndarray):
        """
        Upload BEV image with alpha channel.
        valid_mask: boolean mask of valid pixels.
        """

        if image_rgb is None:
            return

        # Flip for OpenGL
        if self.flip_y:
            image_rgb = cv2.flip(image_rgb, 0)
            valid_mask = np.flipud(valid_mask)

        h, w = image_rgb.shape[:2]

        # Build RGBA
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = image_rgb

        # Alpha = 255 where valid, 0 where invalid
        rgba[:, :, 3] = (valid_mask.astype(np.uint8) * 255)

        rgba = np.ascontiguousarray(rgba)

        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        if not self.has_texture:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                        w, h, 0,
                        GL_RGBA, GL_UNSIGNED_BYTE,
                        rgba)
            self.has_texture = True
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0,
                            0, 0, w, h,
                            GL_RGBA, GL_UNSIGNED_BYTE,
                            rgba)

        glBindTexture(GL_TEXTURE_2D, 0)


    def set_texture_rgba(self, rgba: np.ndarray):
        """
        Upload RGBA directly (alpha per-pixel).
        Expects rgba uint8 (H,W,4).
        """
        if rgba is None:
            return

        if self.flip_y:
            rgba = cv2.flip(rgba, 0)

        rgba = np.ascontiguousarray(rgba)

        h, w = rgba.shape[:2]

        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        if not self.has_texture:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                        GL_RGBA, GL_UNSIGNED_BYTE, rgba)
            self.has_texture = True
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
                            GL_RGBA, GL_UNSIGNED_BYTE, rgba)

        glBindTexture(GL_TEXTURE_2D, 0)

    # --------------------------------------------------
    # Draw
    # --------------------------------------------------

    def draw(self, view, projection, model_matrix):
        if not self.has_texture:
            return

        prev_program = glGetIntegerv(GL_CURRENT_PROGRAM)
        prev_vao = glGetIntegerv(GL_VERTEX_ARRAY_BINDING)

        # ----- Use our shader -----
        glUseProgram(self.shader)

        glUniformMatrix4fv(self.loc_view, 1, GL_FALSE, view)
        glUniformMatrix4fv(self.loc_proj, 1, GL_FALSE, projection)
        glUniformMatrix4fv(self.loc_model, 1, GL_FALSE, model_matrix)

        glUniform1f(self.loc_alpha, float(self.alpha))

        # ----- Proper blending setup -----
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Do NOT write depth (so lidar/grid still render correctly)
        glDepthMask(GL_FALSE)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        # ----- Bind texture -----
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glUniform1i(self.loc_tex, 0)

        # ----- Draw plane -----
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # ----- Restore state -----
        glBindVertexArray(prev_vao)
        glBindTexture(GL_TEXTURE_2D, 0)

        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

        glUseProgram(prev_program)
        

class HDMapBoundaryAccumulator:
    """
    Extract polygons from BEV drivable mask and map them onto the SAME plane
    you render the BEV texture on.

    IMPORTANT:
    - This expects model_ipm_plane = basefootprint_model @ scale(meters_x, meters_y, 1)
      (i.e., already scaled).
    - Therefore, we convert BEV pixels -> quad local coords in [-0.5, 0.5],
      NOT meters (otherwise you'd scale twice).
    """

    def __init__(self, min_area_px: float = 300.0, simplify_eps_px: float = 2.0, accumulate: bool = False):
        self.min_area_px = float(min_area_px)
        self.simplify_eps_px = float(simplify_eps_px)
        self.accumulate = bool(accumulate)
        self.global_polygons: list[np.ndarray] = []

    def reset(self):
        self.global_polygons = []

    def update(self, bev_drivable, model_ipm_plane):
        if bev_drivable is None:
            return

        if not self.accumulate:
            self.global_polygons = []

        # --- ensure 8UC1 binary ---
        mask = bev_drivable
        if mask.ndim == 3:
            # if someone accidentally passes BGR mask, take one channel
            mask = mask[..., 0]
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        mask = (mask > 0).astype(np.uint8) * 255  # 0/255

        H, W = mask.shape

        # --- clean ---
        kernel = np.ones((5, 5), np.uint8)
        clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_px:
                continue

            if self.simplify_eps_px > 0:
                cnt = cv2.approxPolyDP(cnt, self.simplify_eps_px, True)

            pts = cnt.reshape(-1, 2)  # (N,2) px,py

            world_poly = []

            for px, py in pts:
                # pixel -> normalized [0..1]
                u = px / (W - 1.0)

                # IMPORTANT: your texture is flipped vertically in set_texture()
                # so we must also flip the BEV y when mapping to plane coords.
                v = 1.0 - (py / (H - 1.0))

                # normalized -> plane local quad coords [-0.5..0.5]
                x_local = u - 0.5
                y_local = v - 0.5

                local = np.array([x_local, y_local, 0.0, 1.0], dtype=np.float32)
                world = model_ipm_plane @ local
                world_poly.append(world[:3])

            if len(world_poly) >= 3:
                self.global_polygons.append(np.asarray(world_poly, dtype=np.float32))

    def get_polygons(self):
        return self.global_polygons


class HDMapGridAccumulator:
    """
    TRUE world-coordinate persistent COLOR occupancy grid.

    - Fixed global world frame
    - Not attached to ego
    - Floor-based spatial binning (stable)
    - Safe bounds handling
    """

    def __init__(
        self,
        size_x_m,
        size_y_m,
        resolution,
        origin_x,
        origin_y,
        decay=0.995,
    ):
        self.res = float(resolution)
        self.decay = float(decay)

        self.size_x = float(size_x_m)
        self.size_y = float(size_y_m)

        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)

        # Grid dimensions
        self.W = int(np.ceil(self.size_x / self.res))
        self.H = int(np.ceil(self.size_y / self.res))

        # Confidence (alpha)
        self.conf = np.zeros((self.H, self.W), dtype=np.float32)

        # RGB color grid
        self.color = np.zeros((self.H, self.W, 3), dtype=np.float32)

        self._polygon_spheres = PathSphereMarkerRenderer(radius = 0.20, color = (0.0, 1.0, 0.3, 0.9), drag_enabled = True)

        # Editable polygon representation: list of (N,3) arrays (2m-sampled points per polygon)
        self._polygons_editable: list[np.ndarray] = []
        # For each sphere index: (poly_idx, point_idx within that polygon's sampled points)
        self._sphere_to_poly: list[tuple[int, int]] = []
        self._polygons_edited = False

        self._last_polygon_hash = None

    # --------------------------------------------------
    # Reset map
    # --------------------------------------------------
    def reset(self):
        self.conf.fill(0.0)
        self.color.fill(0.0)

    # --------------------------------------------------
    # Update with BEV observation
    # --------------------------------------------------
    def update(self, bev_mask, bev_color, model_ipm_plane):

        # Always decay (persistent fading)
        self.conf *= self.decay

        if bev_mask is None or bev_color is None:
            return

        mask = bev_mask
        if mask.ndim == 3:
            mask = mask[..., 0]

        mask = mask > 0
        if not np.any(mask):
            return

        H_bev, W_bev = mask.shape
        ys, xs = np.where(mask)

        # ---------------------------
        # BEV pixel -> local quad
        # ---------------------------
        u = xs / (W_bev - 1.0)
        v = ys / (H_bev - 1.0)

        x_local = u - 0.5
        y_local = 0.5 - v

        locals = np.stack(
            [x_local, y_local, np.zeros_like(x_local), np.ones_like(x_local)],
            axis=1,
        )

        # ---------------------------
        # Local -> world
        # ---------------------------
        world_pts = (model_ipm_plane @ locals.T).T
        wx = world_pts[:, 0]
        wy = world_pts[:, 1]

        # ---------------------------
        # World -> grid index (FLOOR!)
        # ---------------------------
        gx = np.floor((wx - self.origin_x) / self.res).astype(np.int32)
        gy = np.floor((wy - self.origin_y) / self.res).astype(np.int32)

        valid = (
            (gx >= 0) & (gx < self.W) &
            (gy >= 0) & (gy < self.H)
        )

        if not np.any(valid):
            return

        gx = gx[valid]
        gy = gy[valid]

        sampled_colors = bev_color[ys[valid], xs[valid]].astype(np.float32)

        # For each cell, accumulate using np.add.at
        for c in range(3):
            np.add.at(self.color[..., c], (gy, gx), sampled_colors[:, c])

        np.add.at(self.conf, (gy, gx), 1.0)

    # --------------------------------------------------
    # Convert to RGBA texture
    # --------------------------------------------------
    def to_rgba(self, alpha_scale=255):

        rgba = np.zeros((self.H, self.W, 4), dtype=np.uint8)

        # rgba[..., :3] = np.clip(self.color, 0, 255).astype(np.uint8)

        color_norm = np.zeros_like(self.color)

        valid = self.conf > 0
        color_norm[valid] = self.color[valid] / self.conf[valid, None]

        rgba[..., :3] = np.clip(color_norm, 0, 255).astype(np.uint8)
        rgba[..., 3] = np.clip(self.conf * alpha_scale, 0, 255).astype(np.uint8)

        return rgba

    def extract_polygons(self, min_area_cells=10, simplify_eps_cells=1.5):
        """
        Extract world-space polygons from the persistent HD grid.
        Returns list of Nx3 world coordinate arrays.
        """

        # -----------------------------
        # Build binary occupancy mask
        # -----------------------------
        mask = (self.conf > 0.1).astype(np.uint8) * 255

        if not np.any(mask):
            return []

        # -----------------------------
        # Clean small holes
        # -----------------------------
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons_world = []

        for cnt in contours:

            area = cv2.contourArea(cnt)
            if area < min_area_cells:
                continue

            if simplify_eps_cells > 0:
                cnt = cv2.approxPolyDP(cnt, simplify_eps_cells, True)

            pts = cnt.reshape(-1, 2)  # (N,2) grid indices

            world_poly = []

            for gx, gy in pts:

                # Grid index → world meters
                wx = self.origin_x + (gx + 0.5) * self.res
                wy = self.origin_y + (gy + 0.5) * self.res

                world_poly.append([wx, wy, 0.0])

            if len(world_poly) >= 3:
                polygons_world.append(np.asarray(world_poly, dtype=np.float32))

        return polygons_world

    def sample_polygon_every(self, poly: np.ndarray, step_m: float = 2.0):

        if len(poly) < 3:
            return []

        pts = [np.asarray(p, dtype=np.float32) for p in poly]
        pts.append(pts[0].copy())

        # compute perimeter
        lengths = []
        total_length = 0.0

        for i in range(len(pts) - 1):
            seg_len = np.linalg.norm(pts[i+1] - pts[i])
            lengths.append(seg_len)
            total_length += seg_len

        if total_length < step_m:
            return [pts[0]]

        out = []
        dist = step_m
        accum = 0.0

        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i+1]
            seg = p1 - p0
            seg_len = lengths[i]

            while accum + seg_len >= dist:
                t = (dist - accum) / seg_len
                out.append(p0 + seg * t)
                dist += step_m

            accum += seg_len

        return out

    def _polygons_hash(self, polygons):
        if not polygons:
            return 0
        total = 0
        for p in polygons:
            total += int(np.sum(p) * 1000)
        return total

    def update_polygon_spheres(self, polygons: list[np.ndarray]):
        """Build sphere markers from polygons. Stores editable 2m-sampled points and sphere→(poly, point) mapping."""
        all_marker_points = []
        self._polygons_editable = []
        self._sphere_to_poly = []

        for poly in polygons:
            pts = self.sample_polygon_every(poly, step_m=2.0)
            if not pts:
                continue
            # Store as (N,3) for this polygon (world x,y,z; z=0)
            pts_arr = np.asarray(pts, dtype=np.float32)
            if pts_arr.ndim == 1:
                pts_arr = np.expand_dims(pts_arr, 0)
            self._polygons_editable.append(pts_arr)
            for j in range(len(pts)):
                self._sphere_to_poly.append((len(self._polygons_editable) - 1, j))
            all_marker_points.extend(pts)

        # Use positions as-is so each sphere matches one polygon point (no re-sampling
        # as a single polyline, which would join across polygons and misplace spheres)
        self._polygon_spheres.build_from_positions_direct(all_marker_points)

    def sync_sphere_to_polygon(self, sphere_index: int):
        """Push the current sphere position back to the editable polygon so the polygon moves with the drag."""
        if sphere_index < 0 or sphere_index >= len(self._sphere_to_poly):
            return
        poly_idx, point_idx = self._sphere_to_poly[sphere_index]
        if poly_idx >= len(self._polygons_editable) or point_idx >= len(self._polygons_editable[poly_idx]):
            return
        center = self._polygon_spheres._centers[sphere_index].copy()
        center[2] = 0.0  # keep polygon in z=0 plane
        self._polygons_editable[poly_idx][point_idx] = center
        self._polygons_edited = True

    def sync_all_spheres_to_polygons(self):
        """Rebuild editable polygon vertices from current sphere positions so the mesh matches spheres (neighbors stay correct)."""
        if not self._polygons_editable or len(self._sphere_to_poly) != len(self._polygon_spheres._centers):
            return
        for sphere_index in range(len(self._sphere_to_poly)):
            poly_idx, point_idx = self._sphere_to_poly[sphere_index]
            if poly_idx >= len(self._polygons_editable) or point_idx >= len(self._polygons_editable[poly_idx]):
                continue
            center = self._polygon_spheres._centers[sphere_index].copy()
            center[2] = 0.0
            self._polygons_editable[poly_idx][point_idx] = center

    def get_editable_polygons(self) -> Optional[List[np.ndarray]]:
        """Return editable polygon list for drawing; None if not in edited state or empty."""
        if not self._polygons_edited or not self._polygons_editable:
            return None
        self.sync_all_spheres_to_polygons()
        return self._polygons_editable

    def rasterize_edited_polygons_to_grid(self, color=(0.0, 1.0, 0.3), conf_value=1.0):
        """
        Draw current editable polygons into the grid so the texture reflects the moved shape.

        FIX:
        - Prevent "painting accumulation" by restoring a snapshot of the grid before edits,
        then re-drawing the edited polygons fresh each call.
        """
        if not self._polygons_editable:
            return

        # Make sure polygon vertices reflect current sphere positions
        self.sync_all_spheres_to_polygons()

        # -----------------------------
        # 1) Snapshot base grid once (when editing begins)
        # -----------------------------
        # Lazily create snapshot buffers the first time we rasterize during an edit session.
        if not hasattr(self, "_edit_snapshot_conf"):
            self._edit_snapshot_conf = None
            self._edit_snapshot_color = None

        if self._edit_snapshot_conf is None or self._edit_snapshot_color is None:
            # Snapshot the grid as it was BEFORE the editable polygon overlay is drawn
            self._edit_snapshot_conf = self.conf.copy()
            self._edit_snapshot_color = self.color.copy()

        # -----------------------------
        # 2) Restore snapshot every time (erases previous rasterization footprint)
        # -----------------------------
        self.conf[...] = self._edit_snapshot_conf
        self.color[...] = self._edit_snapshot_color

        # -----------------------------
        # 3) Draw current (latest) edited polygons
        # -----------------------------
        # Single mask reused (avoid realloc per polygon)
        mask = np.zeros((self.H, self.W), dtype=np.uint8)

        r = float(color[0]) * 255.0
        g = float(color[1]) * 255.0
        b = float(color[2]) * 255.0

        for poly in self._polygons_editable:
            if poly is None or len(poly) < 3:
                continue

            # World -> grid
            gx = np.floor((poly[:, 0] - self.origin_x) / self.res).astype(np.int32)
            gy = np.floor((poly[:, 1] - self.origin_y) / self.res).astype(np.int32)

            # Clip to grid bounds
            gx = np.clip(gx, 0, self.W - 1)
            gy = np.clip(gy, 0, self.H - 1)

            pts = np.column_stack((gx, gy)).astype(np.int32)

            # Clear and fill mask for this polygon
            mask.fill(0)
            cv2.fillPoly(mask, [pts], 255)

            sel = mask > 0
            self.conf[sel] = conf_value
            self.color[sel, 0] = r
            self.color[sel, 1] = g
            self.color[sel, 2] = b

    def clear_polygon_edits(self):
        """Clear edited state so next update_polygon_spheres will overwrite from grid again."""
        self._polygons_edited = False
        if hasattr(self, "_edit_snapshot_conf"):
            self._edit_snapshot_conf = None
            self._edit_snapshot_color = None

    def draw_polygon_spheres(self, view: np.ndarray, projection: np.ndarray):
        self._polygon_spheres.draw(view, projection)

    def compute_left_right_from_centerline(
        self,
        centerline: np.ndarray,
        lane_width: float
    ):
        """
        centerline: Nx3 world coordinates
        lane_width: full width in meters

        Returns:
            left_line, right_line (Nx3)
        """

        if len(centerline) < 2:
            return None, None

        pts = centerline[:, :2]

        # Tangent using gradient
        d = np.gradient(pts, axis=0)
        norm = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
        t = d / norm

        # Normal (rotate 90 deg)
        n = np.stack([-t[:, 1], t[:, 0]], axis=1)

        half = lane_width * 0.5

        left_xy  = pts + n * half
        right_xy = pts - n * half

        left  = np.concatenate([left_xy,  np.zeros((len(left_xy), 1))], axis=1)
        right = np.concatenate([right_xy, np.zeros((len(right_xy), 1))], axis=1)

        return left.astype(np.float32), right.astype(np.float32)