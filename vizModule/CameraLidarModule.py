import threading
import numpy as np
import cv2
from UlitysModule import Cube, draw_axes

class CameraLidarModule:
    def __init__(self):
        self.lidar_camera_extrinsics_array = []
        self.camera_array_intrinsics = []
        self._worker = None
        self._pending_result = None
        self._lock = threading.Lock()

    def load_camera_lidar_parameters(self, dataset):
        self.lidar_camera_extrinsics_array = dataset.load_camera_lidar_extrinsics_array()
        self.camera_array_intrinsics = dataset.load_camera_array_intrinsics()

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