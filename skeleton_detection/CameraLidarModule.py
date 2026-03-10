import threading
import numpy as np
import cv2
from UlitysModule import Cube, draw_axes
from scipy.spatial import cKDTree

class CameraLidarModule:
    def __init__(self):
        self.lidar_camera_extrinsics_array = []
        self.camera_array_intrinsics = []
        self.ipm_camera_configs = []
        self._worker = None
        self._pending_result = None
        self._lock = threading.Lock()

        self.print_ipm_camera_configs()

    def load_camera_lidar_parameters(self, dataset):
        self.lidar_camera_extrinsics_array = dataset.load_camera_lidar_extrinsics_array()
        self.camera_array_intrinsics = dataset.load_camera_array_intrinsics()
        self.ipm_camera_configs = dataset.load_ipm_camera_configs()

    def print_ipm_camera_configs(self):
        for config in self.ipm_camera_configs:
            # print the homografic matrix   
            print(config.homography_matrix)

    def transform_cv_to_robot_transform(self, R_ext, t_ext):

            R_cv_to_robot = np.array([
                [ 0,  0,  1],
                [-1,  0,  0],
                [ 0, -1,  0],
            ], dtype=np.float64)

            R_ext_robot = R_cv_to_robot @ R_ext
            t_ext_robot = R_cv_to_robot @ t_ext
            return R_ext_robot, t_ext_robot

    def draw_cameras_lidar_frame_axes(self, cube=None, lidar_frame=None, view_matrix=None, projection_matrix=None, set_model_color=None):
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

            draw_axes(cube, set_model_color, cam_world)

    def _compute_colored(self, lidar_xyz, images, stride):
        """Pure NumPy — safe to call from any thread."""
        pts = np.ascontiguousarray(lidar_xyz[::stride], dtype=np.float32)
        N = pts.shape[0]

        pts_h = np.empty((4, N), dtype=np.float32)
        pts_h[:3] = pts.T
        pts_h[3] = 1.0

        all_xyz = []
        all_colors = []

        for i, ext in enumerate(self.lidar_camera_extrinsics_array):
            if ext is None or i >= len(images) or images[i] is None:
                continue
            if i >= len(self.camera_array_intrinsics) or self.camera_array_intrinsics[i] is None:
                continue

            K = self.camera_array_intrinsics[i].get_K()
            image = images[i]
            h, w = image.shape[:2]

            Rt = np.empty((3, 4), dtype=np.float32)
            Rt[:, :3] = ext.R_opencv
            Rt[:, 3] = ext.t_opencv.ravel()

            P_cam = Rt @ pts_h                          # (3, N)

            front = P_cam[2] > 0.1
            P_cam_f = P_cam[:, front]

            if P_cam_f.shape[1] == 0:
                continue

            inv_z = np.float32(1.0) / P_cam_f[2]
            fx = np.float32(K[0, 0])
            fy = np.float32(K[1, 1])
            cx = np.float32(K[0, 2])
            cy = np.float32(K[1, 2])

            u = (fx * P_cam_f[0] * inv_z + cx + 0.5).astype(np.int32)
            v = (fy * P_cam_f[1] * inv_z + cy + 0.5).astype(np.int32)

            vis = (u >= 0) & (u < w) & (v >= 0) & (v < h)

            u = u[vis]
            v = v[vis]
            if u.shape[0] == 0:
                continue

            all_colors.append(image[v, u])

            front_idx = np.where(front)[0]
            all_xyz.append(pts[front_idx[vis]])

        if not all_xyz:
            return None

        return (
            np.concatenate(all_xyz),
            np.concatenate(all_colors).astype(np.uint8),
        )

    def start_colored_pointcloud(self, lidar_xyz, images, stride=1):
        """Kick off background computation. Non-blocking."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._worker_fn,
            args=(lidar_xyz, images, stride),
            daemon=True,
        )
        self._worker.start()

    def _worker_fn(self, lidar_xyz, images, stride):
        result = self._compute_colored(lidar_xyz, images, stride)
        with self._lock:
            self._pending_result = result

    def poll_colored_pointcloud(self, colored_pointcloud):
        """Call every frame from the render loop. Uploads when the result is ready."""
        with self._lock:
            result = self._pending_result
            self._pending_result = None
        if result is not None:
            colored_pointcloud.update_colored(*result)
            return True
        return False

    def upload_colored_pointcloud(self, lidar_xyz=None, images=None, colored_pointcloud=None, stride=1):
        """Synchronous fallback — blocks until done."""
        if lidar_xyz is None or not images or colored_pointcloud is None:
            return
        result = self._compute_colored(lidar_xyz, images, stride)
        if result is not None:
            colored_pointcloud.update_colored(*result)

    def draw_lidar_on_image(self,
                            image,
                            lidar_xyz,
                            R_ext,
                            t_ext,
                            K):

        img = image.copy()
        h, w = img.shape[:2]

        lidar_xyz = np.asarray(lidar_xyz, dtype=np.float64)

        # --- Transform to camera frame ---
        P_cam = (R_ext @ lidar_xyz.T).T + t_ext.reshape(1,3)

        # Keep only points in front
        mask_front = P_cam[:,2] > 0
        P_cam = P_cam[mask_front]

        if P_cam.shape[0] == 0:
            return img

        # --- Project ---
        rvec_ext, _ = cv2.Rodrigues(R_ext)
        D_zero = np.zeros((4,1), dtype=np.float64)

        proj_pts, _ = cv2.projectPoints(
            lidar_xyz[mask_front],
            rvec_ext,
            t_ext.reshape(3,1),
            K,
            D_zero
        )

        proj_pts = proj_pts.reshape(-1,2)

        u = np.round(proj_pts[:,0]).astype(np.int32)
        v = np.round(proj_pts[:,1]).astype(np.int32)

        # Keep inside image
        mask_img = (
            (u >= 0) & (u < w) &
            (v >= 0) & (v < h)
        )

        u = u[mask_img]
        v = v[mask_img]

        # --- Paint pixels ---
        depth = P_cam[mask_img, 2]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        colors = (depth_norm * 255).astype(np.uint8)

        for ui, vi, ci in zip(u, v, colors):
            img[vi, ui] = (ci, 255-ci, 0)  # gradient

        return img

class LidarKeypointAssociator:

    def __init__(self):
        self.trees = []
        self.xyz_cache = []

    def build(self, lidar_xyz, extrinsics_array, intrinsics_array):

        # store references for query()
        self.extrinsics = extrinsics_array
        self.camera_intrinsics = intrinsics_array

        self.trees.clear()
        self.xyz_cache.clear()

        for cam_idx, ext in enumerate(extrinsics_array):

            cam = intrinsics_array[cam_idx] if cam_idx < len(intrinsics_array) else None

            if ext is None or cam is None:
                self.trees.append(None)
                self.xyz_cache.append(None)
                continue

            R = ext.R_opencv
            t = ext.t_opencv.reshape(3)

            K = cam.get_K()
            fx, fy = K[0,0], K[1,1]
            cx, cy = K[0,2], K[1,2]

            # LiDAR → camera
            cam_pts = (R @ lidar_xyz.T).T + t

            z = cam_pts[:,2]
            valid = z > 0.1

            cam_pts = cam_pts[valid]
            xyz = lidar_xyz[valid]

            u = fx * cam_pts[:,0] / cam_pts[:,2] + cx
            v = fy * cam_pts[:,1] / cam_pts[:,2] + cy

            pixels = np.stack([u,v], axis=1)

            tree = cKDTree(pixels)

            self.trees.append(tree)
            self.xyz_cache.append(xyz)

    def query(self, cam_idx, u, v, max_ray_dist=0.2):

        tree = self.trees[cam_idx]
        xyz = self.xyz_cache[cam_idx]

        if tree is None or xyz is None:
            return None

        # --- get ~20 closest pixel candidates ---
        dist, idx = tree.query([u, v], k=20)

        candidates = xyz[idx]

        # --- camera intrinsics ---
        cam = self.camera_intrinsics[cam_idx]
        K = cam.get_K()

        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]

        # --- ray in camera frame ---
        ray_cam = np.array([
            (u - cx) / fx,
            (v - cy) / fy,
            1.0
        ])

        ray_cam /= np.linalg.norm(ray_cam)

        # --- convert ray to lidar frame ---
        ext = self.extrinsics[cam_idx]
        R = ext.R_opencv
        t = ext.t_opencv.reshape(3)

        ray_lidar = R.T @ ray_cam
        cam_origin = -R.T @ t

        # --- distance of candidates to ray ---
        vec = candidates - cam_origin
        proj = vec @ ray_lidar
        closest = cam_origin + np.outer(proj, ray_lidar)

        dist_ray = np.linalg.norm(candidates - closest, axis=1)

        mask = dist_ray < max_ray_dist
        if not np.any(mask):
            return None

        candidates = candidates[mask]
        depth = proj[mask]

        # choose closest along ray
        idx = np.argmin(depth)

        return candidates[idx]


    def box2d_to_3d(self, cam_idx, bbox, lidar_xyz, extrinsics, intrinsics):

        x1, y1, x2, y2 = bbox

        ext = extrinsics[cam_idx]
        cam = intrinsics[cam_idx]

        R = ext.R_opencv
        t = ext.t_opencv.reshape(3)

        K = cam.get_K()
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]

        cam_pts = (R @ lidar_xyz.T).T + t

        z = cam_pts[:,2]
        valid = z > 0.1

        cam_pts = cam_pts[valid]
        xyz = lidar_xyz[valid]

        u = fx * cam_pts[:,0] / cam_pts[:,2] + cx
        v = fy * cam_pts[:,1] / cam_pts[:,2] + cy

        mask = (
            (u >= x1) & (u <= x2) &
            (v >= y1) & (v <= y2)
        )

        pts = xyz[mask]

        result = self.fit_oriented_box_pca(pts)

        if result is None:
            return None

        center, size, yaw = result

        return center, size, yaw

    def fit_oriented_box_pca(self, pts):
        """
        Fit an oriented 3D bounding box using PCA.

        Parameters
        ----------
        pts : np.ndarray (N,3)
            LiDAR points belonging to the object

        Returns
        -------
        center : np.ndarray (3,)
        size   : np.ndarray (3,)
        yaw    : float
        """

        if pts.shape[0] < 5:
            return None

        # --- use XY plane for orientation ---
        xy = pts[:, :2]

        mean_xy = xy.mean(axis=0)
        xy_centered = xy - mean_xy

        cov = xy_centered.T @ xy_centered / xy_centered.shape[0]

        eigvals, eigvecs = np.linalg.eigh(cov)

        # largest eigenvector = main direction
        main_dir = eigvecs[:, np.argmax(eigvals)]

        yaw = np.arctan2(main_dir[1], main_dir[0])

        # --- rotate points into object frame ---
        c = np.cos(-yaw)
        s = np.sin(-yaw)

        R = np.array([
            [c, -s],
            [s,  c]
        ])

        xy_rot = (R @ xy_centered.T).T

        # compute extents
        min_xy = xy_rot.min(axis=0)
        max_xy = xy_rot.max(axis=0)

        size_x = max_xy[0] - min_xy[0]
        size_y = max_xy[1] - min_xy[1]

        min_z = pts[:,2].min()
        max_z = pts[:,2].max()

        size_z = max_z - min_z

        center_xy = (min_xy + max_xy) * 0.5
        center_xy = mean_xy + (R.T @ center_xy)

        center_z = (min_z + max_z) * 0.5

        center = np.array([center_xy[0], center_xy[1], center_z])
        size = np.array([size_x, size_y, size_z])

        return center, size, yaw

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

    def draw_skeletons_on_camera(self, images, skeletons):
        """
        Project 3D skeletons onto each camera image.
        """
        skel_by_cam: dict[int, list[dict]] = {}
        if skeletons is not None:
            for cam_entry in skeletons:
                skel_by_cam[cam_entry["cam"]] = cam_entry.get("persons", [])

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

