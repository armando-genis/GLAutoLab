import os
import sys
import yaml
from pathlib import Path
import re
import numpy as np
import cv2
from PIL import Image, ImageOps
from collections import defaultdict

from CameraModule import CameraUndistorter, CameraLidarExtrinsics, IpmCameraConfig
from carModelModule import CarSettings

# PyYAML (YAML 1.1) treats zero-padded integers like 00010 as octal.
# Override both the resolver and constructor so they are always decimal.
class _DecimalSafeLoader(yaml.SafeLoader):
    pass

_INT_TAG = "tag:yaml.org,2002:int"
for _ch, _resolvers in list(_DecimalSafeLoader.yaml_implicit_resolvers.items()):
    _DecimalSafeLoader.yaml_implicit_resolvers[_ch] = [
        (tag, regexp) for tag, regexp in _resolvers if tag != _INT_TAG
    ]

_DecimalSafeLoader.add_implicit_resolver(
    _INT_TAG,
    re.compile(r"^[-+]?[0-9]+$"),
    list("-+0123456789"),
)
_DecimalSafeLoader.add_constructor(
    _INT_TAG,
    lambda loader, node: int(loader.construct_scalar(node)),
)

_IMAGE_RE = re.compile(r"(racecar_camera_camera_\d+_image_raw)_(\d{5})\.jpg")
# Mask files: racecar_camera_camera_N_image_raw_XXXXX_<hex8>_mask.png
_MASK_RE = re.compile(r"(racecar_camera_camera_\d+_image_raw)_(\d{5})_[0-9a-f]{8}_mask\.(?:png|jpg)", re.IGNORECASE)
_CAMERA_INDEX_RE = re.compile(r"camera_(\d+)_image_raw")
_CAM_FILE_RE = re.compile(r"cam(\d+)_(intrinsic)\.yaml")
_LIDAR_CAM_FILE_RE = re.compile(r"LidartoCam(\d+)\.yaml")

def _to_mat_3x3(block):
    if isinstance(block, dict) and "data" in block:
        return np.array(block["data"], dtype=np.float64).reshape(3, 3)
    return np.array(block, dtype=np.float64).reshape(3, 3)

def _to_vec(block):
    if isinstance(block, dict) and "data" in block:
        return np.array(block["data"], dtype=np.float64).reshape(-1, 1)
    return np.array(block, dtype=np.float64).reshape(-1, 1)

def _camera_index(cam_name: str) -> int:
    """Extract camera number from cam_name for consistent ordering (e.g. camera_0 -> 0)."""
    m = _CAMERA_INDEX_RE.search(cam_name)
    return int(m.group(1)) if m else 0

class SyncDataset:
    def __init__(self, root: Path):
        self.image_dir = root / "individual"
        self.lidar_dir = root / "lidar_bins"
        self.pose_dir = root / "tf_yaml"
        self.images_dir_mask = root / "individual_mask"
        self.skeleton_dir = root / "skeleton_person"

        self.mask_resolution = None  # (width, height) when mask images exist
        self.samples = self._index_samples()
        self._index_masks()
        self.camera_array = []
        self.extrinsics_array = []
        self.ipm_camera_configs = []
        self.lidar_config = {}
        self.car_settings = None
        self.world_offset_height = 0.0

        self.car_model_file = "rot_sdv"




    def _index_samples(self):
        samples = defaultdict(lambda: {"images": {}})

        for img_path in self.image_dir.glob("*.jpg"):
            m = _IMAGE_RE.match(img_path.name)
            if not m:
                continue
            cam_name = m.group(1)
            idx = int(m.group(2))
            samples[idx]["images"][cam_name] = img_path

        for bin_path in self.lidar_dir.glob("*.bin"):
            idx = int(bin_path.stem)
            samples[idx]["lidar"] = bin_path

        poses_path = self.pose_dir / "poses.yaml"
        if poses_path.exists():
            with open(poses_path, "r") as f:
                pose_data = yaml.load(f, Loader=_DecimalSafeLoader)
            for entry in pose_data.get("scenes", []):
                idx = int(entry["scene"])
                samples[idx]["pose"] = entry["pose"]

        skeleton_path = self.skeleton_dir / "detections.yaml"
        if skeleton_path.exists():
            with open(skeleton_path, "r") as f:
                skeleton_data = yaml.load(f, Loader=_DecimalSafeLoader)
            for entry in skeleton_data.get("scenes", []):
                idx = int(entry["scene"])
                samples[idx]["skeleton"] = entry.get("cameras", [])

        synced = {
            idx: s
            for idx, s in samples.items()
            if "lidar" in s and len(s["images"]) > 0
        }
        return dict(sorted(synced.items()))

    def _index_masks(self):
        """
        Index mask images from images_dir_mask if the folder exists and contains images.
        Only sets mask paths on samples when the folder exists and has mask images.
        """
        if not self.images_dir_mask.exists() or not self.images_dir_mask.is_dir():
            return
        mask_paths = list(self.images_dir_mask.glob("*.jpg")) + list(self.images_dir_mask.glob("*.png"))
        if not mask_paths:
            return
        for path in mask_paths:
            m = _MASK_RE.match(path.name)
            if not m:
                continue
            cam_name = m.group(1)
            idx = int(m.group(2))
            if idx in self.samples:
                self.samples[idx].setdefault("masks", {})[cam_name] = path

        # Store mask resolution from first available mask image
        if self.has_masks():
            first_path = None
            for s in self.samples.values():
                if "masks" in s and s["masks"]:
                    first_path = next(iter(s["masks"].values()))
                    break
            if first_path is not None:
                with Image.open(first_path) as pil_img:
                    self.mask_resolution = pil_img.size  # (width, height)

    def has_masks(self) -> bool:
        """True if any sample has mask images (mask folder existed and had images)."""
        return any("masks" in s and s["masks"] for s in self.samples.values())

    def load_masks(self, idx: int):
        """
        Load mask images for scene idx. Returns None if no masks are available
        for this scene; otherwise returns a dict cam_name -> np.ndarray (grayscale mask).
        """
        masks_entry = self.samples[idx].get("masks")
        if not masks_entry:
            return None
        items = sorted(
            masks_entry.items(),
            key=lambda x: _camera_index(x[0]),
        )
        masks = {}
        for cam_name, path in items:
            with Image.open(path) as pil_img:
                pil_img = pil_img.convert("L")
                mask = np.array(pil_img)
            masks[cam_name] = mask
        return masks if masks else None

    def indices(self):
        return list(self.samples.keys())

    def num_scenes(self) -> int:
        """Return the total number of scenes in the dataset."""
        return len(self.samples)

    def load_camera_lidar_extrinsics_array(self):
        return self.extrinsics_array

    def load_camera_array_intrinsics(self):
        return self.camera_array

    def load_ipm_camera_configs(self):
        return self.ipm_camera_configs

    def load_images(self, idx: int):
        # Build cam_index -> detections lookup for O(1) access per camera
        skel_by_cam: dict[int, list[dict]] = {}
        skeleton = self.load_skeleton(idx)
        if skeleton is not None:
            for cam_entry in skeleton:
                skel_by_cam[cam_entry["cam"]] = cam_entry.get("persons", [])

        items = sorted(
            self.samples[idx]["images"].items(),
            key=lambda x: _camera_index(x[0]),
        )
        imgs = {}
        for cam_name, path in items:
            with Image.open(path) as pil_img:
                pil_img = pil_img.convert("RGB")
                img = np.array(pil_img)

            cam_idx = _camera_index(cam_name)

            # Draw skeleton on raw pixels before undistortion
            if cam_idx in skel_by_cam:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = self._draw_detections(img, skel_by_cam[cam_idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            undistorter = (
                self.camera_array[cam_idx]
                if self.camera_array and cam_idx < len(self.camera_array)
                else None
            )
            if undistorter is not None:
                h, w = img.shape[:2]
                undistorter.ensure_size(w, h)
                img = undistorter.undistort(img)

            imgs[cam_name] = img

        return imgs

    # COCO skeleton connections (0-indexed keypoint pairs)
    _SKELETON_PAIRS = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    _KPT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    _CONF_THR   = 0.3
    _KPT_COLOR  = (0, 255, 0)
    _LIMB_COLOR = (0, 165, 255)
    _BOX_COLOR  = (255, 0, 0)
    _TEXT_COLOR = (255, 255, 255)

    def _draw_detections(self, img: np.ndarray, persons: list[dict]) -> np.ndarray:
        """Draw bounding boxes and skeletons for all persons onto a BGR image (in-place)."""
        for person in persons:
            b = person["box"]
            cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]),
                          self._BOX_COLOR, 2, cv2.LINE_AA)
            cv2.putText(img, f"person {b['conf']:.2f}",
                        (b["x1"], b["y1"] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._TEXT_COLOR, 1, cv2.LINE_AA)

            kpts = person.get("keypoints", {})

            # Limbs
            for a, b_idx in self._SKELETON_PAIRS:
                kpa = kpts.get(self._KPT_NAMES[a])
                kpb = kpts.get(self._KPT_NAMES[b_idx])
                if kpa is None or kpb is None:
                    continue
                if kpa["conf"] < self._CONF_THR or kpb["conf"] < self._CONF_THR:
                    continue
                cv2.line(img,
                         (int(kpa["x"]), int(kpa["y"])),
                         (int(kpb["x"]), int(kpb["y"])),
                         self._LIMB_COLOR, 2, cv2.LINE_AA)

            # Keypoints
            for kp in kpts.values():
                if kp["conf"] < self._CONF_THR:
                    continue
                cv2.circle(img, (int(kp["x"]), int(kp["y"])),
                           4, self._KPT_COLOR, -1, cv2.LINE_AA)

        return img

    def load_raw_images(self, idx: int):
        items = sorted(
            self.samples[idx]["images"].items(),
            key=lambda x: _camera_index(x[0]),
        )
        imgs = {}
        for cam_name, path in items:
            with Image.open(path) as pil_img:
                pil_img = pil_img.convert("RGB")
                img = np.array(pil_img)
            imgs[cam_name] = img
        return imgs

    def load_lidar(self, idx: int):
        bin_path = self.samples[idx]["lidar"]
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]
        valid_mask = np.isfinite(xyz).all(axis=1)
        return xyz[valid_mask]

    def load_pose(self, idx: int):
        pose = self.samples[idx].get("pose")
        if pose is None:
            return None
        xyz     = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)
        heading = float(pose["heading"])
        roll    = float(pose.get("roll",  0.0))
        pitch   = float(pose.get("pitch", 0.0))
        return xyz, heading, roll, pitch

    def load_skeleton(self, idx: int) -> list[dict] | None:
        """
        Return skeleton detections for scene `idx`, or None if unavailable.

        Returns a list of camera dicts (one per camera):
            [
                {
                    "cam": int,
                    "persons": [
                        {
                            "box": {"x1", "y1", "x2", "y2", "conf"},
                            "keypoints": {
                                "nose":          {"x", "y", "conf"},
                                "left_shoulder": {"x", "y", "conf"},
                                ...
                            }
                        },
                        ...
                    ]
                },
                ...
            ]
        """
        return self.samples[idx].get("skeleton") or None

    # Camera Loader Utility
    def _is_fisheye_yaml(self, data: dict) -> bool:
        camera_type = str(data.get("camera_type", "")).strip().lower()
        if camera_type == "fisheye":
            return True
        dist_model = str(data.get("distortion_model", "")).strip().lower()
        return dist_model in ("equidistant", "fisheye")

    def _load_undistorter(self, yaml_path: str) -> CameraUndistorter:

        if not os.path.exists(yaml_path):
            self.get_logger().warn(f"cam_yaml not found: {yaml_path}. Undistortion disabled.")
            K = np.eye(3, dtype=np.float64)
            D = np.zeros((4, 1), dtype=np.float64)
            self._K = K
            self._D = D
            return CameraUndistorter(K, D, (0, 0))

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        w = int(data.get("image_width", 0))
        h = int(data.get("image_height", 0))

        K = _to_mat_3x3(data["camera_matrix"])
        D = _to_vec(data["distortion_coefficients"])
        is_fisheye = self._is_fisheye_yaml(data)

        print(f"Loaded {yaml_path} | size=({w}x{h}) | fisheye={is_fisheye} | D_len={D.shape[0]}")

        return CameraUndistorter(K, D, (w, h))

    def _load_car_config(self, calib_dir: Path) -> dict:
        car_config_path = calib_dir / "carConfig.yaml"
        if not car_config_path.exists():
            print(f"carConfig.yaml not found at {car_config_path}. Using default car config.")
            return {}
        with open(car_config_path, "r") as f:
            data = yaml.safe_load(f)
        print(f"Loaded car config from {car_config_path}")
        return data

    def _load_car_settings(self, calib_dir: Path) -> CarSettings:

        car_settings_path = calib_dir / "carConfig.yaml"

        if not car_settings_path.exists():
            raise FileNotFoundError(f"carConfig.yaml not found at {car_settings_path}")

        with open(car_settings_path, "r") as f:
            data = yaml.safe_load(f)

        width = data["vehicle_config"]["width"]
        height = data["vehicle_config"]["height"]
        length = data["vehicle_config"]["length"]
        self.car_model_file = data["vehicle_config"]["model_file"]


        R_Base_to_Lidar = _to_mat_3x3(data["base_link_to_lidar"]["R"])
        t_Base_to_Lidar = _to_vec(data["base_link_to_lidar"]["t"])
        R_Center_to_Base = _to_mat_3x3(data["center_link_to_base_link"]["R"])
        t_Center_to_Base = _to_vec(data["center_link_to_base_link"]["t"])

        base_to_lidar = self.lidar_config.get("base_footprint_to_lidar", {})
        R_Basefootprint_to_lidar = _to_mat_3x3(base_to_lidar["R"])
        t_Basefootprint_to_lidar = _to_vec(base_to_lidar["t"])

        return CarSettings(
            width,
            height,
            length,
            R_Base_to_Lidar,
            t_Base_to_Lidar,
            R_Center_to_Base,
            t_Center_to_Base,
            R_Basefootprint_to_lidar,
            t_Basefootprint_to_lidar
        )

    def _load_lidar_config(self, calib_dir: Path) -> dict:
        lidar_config_path = calib_dir / "lidarConfig.yaml"
        if not lidar_config_path.exists():
            print(f"lidarConfig.yaml not found at {lidar_config_path}. Using default lidar rotation.")
            return {}
        with open(lidar_config_path, "r") as f:
            data = yaml.safe_load(f)
        print(f"Loaded lidar config from {lidar_config_path}")
        return data

    def _load_extrinsics(self, yaml_path: str) -> CameraLidarExtrinsics:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        ext    = data["extrinsics"]
        opencv = ext["opencv_frame"]
        robot  = ext["robot_frame"]

        R_opencv = _to_mat_3x3(opencv["R"])
        t_opencv = _to_vec(opencv["t"])
        R_robot  = _to_mat_3x3(robot["R"])
        t_robot  = _to_vec(robot["t"])

        lr = self.lidar_config.get("lidar_rotation", {})
        lidar_rotation = {
            "axis_x": float(lr.get("axis_x", 0.0)),
            "axis_y": float(lr.get("axis_y", 0.0)),
            "axis_z": float(lr.get("axis_z", 0.0)),
        }

        print(f"Loaded extrinsics {yaml_path}")
        return CameraLidarExtrinsics(R_opencv, t_opencv, R_robot, t_robot, lidar_rotation)

    def _load_camera_configs(self, calib_dir: Path):
        configs = {}

        for file in calib_dir.glob("cam*_*.yaml"):
            m = _CAM_FILE_RE.match(file.name)
            if not m:
                continue
            cam_idx = int(m.group(1))
            file_type = m.group(2)
            configs.setdefault(cam_idx, {})
            configs[cam_idx][file_type] = file

        for file in calib_dir.glob("LidartoCam*.yaml"):
            m = _LIDAR_CAM_FILE_RE.match(file.name)
            if not m:
                continue
            cam_idx = int(m.group(1))
            configs.setdefault(cam_idx, {})
            configs[cam_idx]["extrinsic"] = file

        return configs

    def build_camera_array(self, calib_dir: Path):
        """
        Returns:
            list[CameraUndistorter | None]
        """

        # Collect camera indices used in dataset
        used_cam_indices = set()

        for idx in self.indices():
            for cam_name in self.samples[idx]["images"].keys():
                used_cam_indices.add(_camera_index(cam_name))

        configs = self._load_camera_configs(calib_dir)
        self.lidar_config = self._load_lidar_config(calib_dir)

        base_to_lidar = self.lidar_config.get("base_footprint_to_lidar", {})
        R_Basefootprint_to_lidar = _to_mat_3x3(base_to_lidar["R"])
        t_Basefootprint_to_lidar = _to_vec(base_to_lidar["t"])

        self.world_offset_height = -t_Basefootprint_to_lidar[2, 0] # negative because the height is in the negative z direction

        print(f"World offset height: {self.world_offset_height}")

        self.car_settings = self._load_car_settings(calib_dir)

        if not configs and used_cam_indices:
            print("No camera calibration files found.")

        max_index = max(
            max(used_cam_indices) if used_cam_indices else 0,
            max(configs.keys()) if configs else 0,
        )

        self.camera_array = []
        self.extrinsics_array = []
        self.ipm_camera_configs = []

        for cam_idx in range(max_index + 1):

            intrinsic_file = configs.get(cam_idx, {}).get("intrinsic")
            extrinsic_file = configs.get(cam_idx, {}).get("extrinsic")

            used_in_data = cam_idx in used_cam_indices

            # Case 1 — Intrinsic calibration exists
            if intrinsic_file:

                cam = self._load_undistorter(str(intrinsic_file))

                self.camera_array.append(cam)

                if not used_in_data:
                    print(
                        f"Camera {cam_idx} has calibration but is not used in dataset."
                    )

            # Case 2 — Missing calibration
            else:
                if used_in_data:
                    print(
                        f"Camera {cam_idx} used in dataset but calibration missing."
                    )

                self.camera_array.append(None)

            # Extrinsics (Lidar → Camera)
            ext = self._load_extrinsics(str(extrinsic_file)) if extrinsic_file else None
            self.extrinsics_array.append(ext)

            # IPM Camera Config
            if intrinsic_file:
                self.ipm_camera_configs.append(self._load_ipm_camera_config(str(intrinsic_file), ext))
            else:
                self.ipm_camera_configs.append(None)

    def _load_ipm_camera_config(self, intrinsic_file: str, ext: CameraLidarExtrinsics) -> IpmCameraConfig:

        with open(intrinsic_file, 'r') as f:
            data = yaml.safe_load(f)

            fx = data["camera_matrix"]["data"][0]
            fy = data["camera_matrix"]["data"][4]
            px = data["camera_matrix"]["data"][2]
            py = data["camera_matrix"]["data"][5]

        # get the base link to lidar from lidarConfig
        base_to_lidar = self.lidar_config.get("base_footprint_to_lidar", {})
        R_base_to_lidar = _to_mat_3x3(base_to_lidar["R"])
        t_base_to_lidar = _to_vec(base_to_lidar["t"])

        # Inverse of the extrinsic: camera pose expressed in LiDAR frame (robot convention)
        R_cam_in_lidar = ext.R_robot.T                          # (3, 3)
        t_cam_in_lidar = (-ext.R_robot.T @ ext.t_robot) # (3, 1)

        R_cam_in_base = R_base_to_lidar @ R_cam_in_lidar
        t_cam_in_base = t_base_to_lidar + (R_base_to_lidar @ t_cam_in_lidar)

        # --- Euler ZYX (yaw, pitch, roll) from R_cam_in_base ---
        # yaw   = atan2(r21, r11)
        # pitch = atan2(-r31, sqrt(r32^2 + r33^2))
        # roll  = atan2(r32, r33)
        yaw = np.degrees(np.arctan2(R_cam_in_base[1, 0], R_cam_in_base[0, 0]))
        pitch = np.degrees(np.arctan2(-R_cam_in_base[2, 0],
                                        np.sqrt(R_cam_in_base[2, 1]**2 + R_cam_in_base[2, 2]**2)))
        roll = np.degrees(np.arctan2(R_cam_in_base[2, 1], R_cam_in_base[2, 2]))

        # --- Store translation components ---
        XCam = float(t_cam_in_base[0, 0])
        YCam = float(t_cam_in_base[1, 0])
        ZCam = float(t_cam_in_base[2, 0])

        print(f"Camera {intrinsic_file} in base frame:")
        print("R_cam_in_base:\n", R_cam_in_base)
        print("t_cam_in_base:", t_cam_in_base.ravel())
        print("--------------------------------")
        # print fx, fy, px, py
        print(f"fx: {fx:.6f}, fy: {fy:.6f}, px: {px:.6f}, py: {py:.6f}")
        print(f"yaw: {yaw:.6f} deg, pitch: {pitch:.6f} deg, roll: {roll:.6f} deg")
        print(f"XCam: {XCam:.6f}, YCam: {YCam:.6f}, ZCam: {ZCam:.6f}")

        return IpmCameraConfig(fx, fy, px, py, yaw, pitch, roll, XCam, YCam, ZCam)


    def print_camera_info(self):
        """
        Pretty print camera calibration info.
        """
        print("\n================ CAMERA INFO ================\n")
        for idx, cam in enumerate(self.camera_array):

            print(f"--- Camera {idx} ---")

            if cam is None:
                print("Intrinsics: MISSING")
            else:
                print("Intrinsics: LOADED")
                print(f"\nFrame size (w x h): {cam.frame_size}")
                print("\nIntrinsics (K):")
                print(cam.K)
                print("\nDistortion (D):")
                print(cam.D)

            ext = self.extrinsics_array[idx] if idx < len(self.extrinsics_array) else None
            if ext is None:
                print("Extrinsics: MISSING")
            else:
                print("\nExtrinsics (Lidar → Camera): LOADED")
                print("\nR (opencv_frame):")
                print(ext.R_opencv)
                print("\nt (opencv_frame):")
                print(ext.t_opencv)
                print("\nR (robot_frame):")
                print(ext.R_robot)
                print("\nt (robot_frame):")
                print(ext.t_robot)
                print(f"\nLidar rotation: {ext.lidar_rotation}")

            print("\n---------------------------------------------\n")

            ipm_cam = self.ipm_camera_configs[idx] if idx < len(self.ipm_camera_configs) else None
            if ipm_cam is None:
                print("IPM Camera Config: MISSING")
            else:
                print("IPM Camera Config: LOADED")
                print(f"\nK: {ipm_cam.K}")
                print(f"\nP: {ipm_cam.P}")


