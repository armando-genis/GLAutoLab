"""
Standalone IPM (Inverse Perspective Mapping) script.

Loads camera configs (front, right) expressed in vehicle/base_footprint frame
  x → forward, y → left, z → up
and warps each image onto a ground-plane BEV using a virtual drone camera.
"""

import numpy as np
import cv2
import yaml
import os

# ──────────────────────────────────────────────────────────────
# Camera model (same math as vizModule/CameraModule.py)
# ──────────────────────────────────────────────────────────────

class IpmCameraConfig:
    def __init__(self, fx, fy, px, py, yaw, pitch, roll, XCam, YCam, ZCam):
        self.K = np.zeros((3, 3))
        self.R = np.zeros((3, 3))
        self.t = np.zeros((3, 1))
        self.P = np.zeros((3, 4))
        self.yaw = yaw
        self.homography_matrix = None

        self._set_K(fx, fy, px, py)
        self._set_R(np.deg2rad(yaw), np.deg2rad(pitch), np.deg2rad(roll))
        self._set_T(XCam, YCam, ZCam)
        self._update_P()

    def _set_K(self, fx, fy, px, py):
        self.K[0, 0] = fx
        self.K[1, 1] = fy
        self.K[0, 2] = px
        self.K[1, 2] = py
        self.K[2, 2] = 1.0

    def _set_R(self, y, p, r):
        # Intrinsic ZYX (yaw-pitch-roll): pitch/roll are relative to the
        # camera's local frame AFTER yawing.  Matrix form of the body
        # rotation is Rz(yaw)·Ry(pitch)·Rx(roll); its inverse (vehicle→body)
        # is Rx(-roll)·Ry(-pitch)·Rz(-yaw).
        # Rs then switches from vehicle axes to OpenCV camera axes.
        Rz = np.array([[ np.cos(-y), -np.sin(-y), 0.0],
                        [ np.sin(-y),  np.cos(-y), 0.0],
                        [        0.0,         0.0, 1.0]])
        Ry = np.array([[ np.cos(-p), 0.0, np.sin(-p)],
                        [        0.0, 1.0,        0.0],
                        [-np.sin(-p), 0.0, np.cos(-p)]])
        Rx = np.array([[1.0,        0.0,         0.0],
                        [0.0, np.cos(-r), -np.sin(-r)],
                        [0.0, np.sin(-r),  np.cos(-r)]])
        Rs = np.array([[ 0.0, -1.0, 0.0],
                        [ 0.0,  0.0,-1.0],
                        [ 1.0,  0.0, 0.0]])
        self.R = Rs @ Rx @ Ry @ Rz

    def _set_T(self, XCam, YCam, ZCam):
        X = np.array([XCam, YCam, ZCam])
        self.t = -self.R @ X

    def _update_P(self):
        Rt = np.zeros((3, 4))
        Rt[:3, :3] = self.R
        Rt[:3, 3] = self.t
        self.P = self.K @ Rt


# ──────────────────────────────────────────────────────────────
# Fisheye undistortion
# ──────────────────────────────────────────────────────────────

def load_intrinsic(yaml_path):
    """Load fisheye intrinsic calibration (K + distortion D) from YAML."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    K = np.array(cfg["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    D = np.array(cfg["distortion_coefficients"]["data"], dtype=np.float64).reshape(4, 1)
    w = cfg["image_width"]
    h = cfg["image_height"]
    return K, D, (w, h)


def undistort_fisheye(img, K, D):
    """Remove fisheye distortion using the equidistant model."""
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
    )
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def load_camera_config(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return IpmCameraConfig(
        cfg["fx"], cfg["fy"], cfg["px"], cfg["py"],
        cfg["yaw"], cfg["pitch"], cfg["roll"],
        cfg["XCam"], cfg["YCam"], cfg["ZCam"],
    )


def build_drone_camera(yaml_path=None):
    """Virtual top-down drone camera used to define the BEV output grid."""
    if yaml_path and os.path.exists(yaml_path):
        return load_camera_config(yaml_path)
    return IpmCameraConfig(
        fx=682.578, fy=682.578, px=482.0, py=302.0,
        yaw=0.0, pitch=90.0, roll=90.0,
        XCam=0.0, YCam=0.0, ZCam=40.0,
    )


# ──────────────────────────────────────────────────────────────
# IPM computation
# ──────────────────────────────────────────────────────────────

def compute_output_shape(drone):
    """Returns (outputRes, pxPerM, M) where M maps BEV pixel → world-z=0."""
    fx, fy = drone.K[0, 0], drone.K[1, 1]
    px, py = drone.K[0, 2], drone.K[1, 2]
    cam_world = (-drone.R.T @ drone.t).flatten()
    x_cam, y_cam, z_cam = cam_world

    outputRes = (int(2 * py), int(2 * px))
    dx = outputRes[1] / fx * z_cam
    dy = outputRes[0] / fy * z_cam
    pxPerM = (outputRes[0] / dy, outputRes[1] / dx)

    shift = (outputRes[0] / 2.0, outputRes[1] / 2.0)
    shift = (shift[0] + y_cam * pxPerM[0],
             shift[1] - x_cam * pxPerM[1])

    M = np.array([
        [ 1.0 / pxPerM[1], 0.0, -shift[1] / pxPerM[1]],
        [ 0.0, -1.0 / pxPerM[0],  shift[0] / pxPerM[0]],
        [ 0.0,              0.0,                    0.0],
        [ 0.0,              0.0,                    1.0],
    ])
    return outputRes, pxPerM, M


def compute_homographies(cam_configs, M):
    """H = inv(P @ M) for each camera — maps image pixels → BEV pixels."""
    homographies = []
    for cfg in cam_configs:
        H = np.linalg.inv(cfg.P @ M)
        cfg.homography_matrix = H
        homographies.append(H)
    return homographies


def compute_invalid_masks(cam_configs, drone, outputRes, pxPerM):
    """Mask out BEV regions behind each camera (> 90° from viewing direction)."""
    cam_world = (-drone.R.T @ drone.t).flatten()
    x_cam, y_cam = cam_world[0], cam_world[1]

    j_coords = np.arange(outputRes[0])
    i_coords = np.arange(outputRes[1])
    i_grid, j_grid = np.meshgrid(i_coords, j_coords)

    y_offset = -j_grid + outputRes[0] / 2 - y_cam * pxPerM[0]
    x_offset =  i_grid - outputRes[1] / 2 + x_cam * pxPerM[1]
    theta = np.rad2deg(np.arctan2(y_offset, x_offset))

    masks = []
    for cfg in cam_configs:
        diff = (theta - cfg.yaw + 180) % 360 - 180
        mask_2d = np.abs(diff) > 90
        masks.append(np.stack([mask_2d] * 3, axis=-1))
    return masks


def warp_and_stitch(images, cam_configs, masks, outputRes):
    """Warp each image to BEV with distance-based feathering at seams."""
    H, W = outputRes

    warped_list = []
    weight_list = []

    for i, (img, cfg) in enumerate(zip(images, cam_configs)):
        warped = cv2.warpPerspective(img, cfg.homography_matrix,
                                     (W, H), flags=cv2.INTER_LINEAR)
        if i < len(masks):
            warped[masks[i]] = 0

        valid = np.any(warped != 0, axis=-1).astype(np.uint8)

        # Distance transform: each valid pixel gets a weight proportional
        # to how far it is from the nearest invalid/border pixel.
        dist = cv2.distanceTransform(valid, cv2.DIST_L2, 5).astype(np.float32)

        warped_list.append(warped)
        weight_list.append(dist)

    accum = np.zeros((H, W, 3), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    for warped, w in zip(warped_list, weight_list):
        accum += warped.astype(np.float32) * w[..., None]
        weight_sum += w

    weight_sum[weight_sum == 0] = 1
    bev = np.clip(accum / weight_sum[..., None], 0, 255).astype(np.uint8)
    return bev, warped_list


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "camera_configs")
    image_dir  = os.path.join(script_dir, "images")

    camera_names = ["front", "right"]
    # intrinsic filenames may not match exactly (e.g. "rigth_intrinsic.yaml")
    intrinsic_names = {"front": "front_intrinsic.yaml",
                       "right": "rigth_intrinsic.yaml"}
    cam_configs = []
    images = []

    for name in camera_names:
        cfg_path = os.path.join(config_dir, f"{name}.yaml")
        img_path = os.path.join(image_dir, f"{name}.jpg")

        if not os.path.exists(cfg_path):
            print(f"[WARN] Config not found: {cfg_path}")
            continue
        if not os.path.exists(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue

        cam_configs.append(load_camera_config(cfg_path))
        img = cv2.imread(img_path)

        intr_path = os.path.join(config_dir, intrinsic_names.get(name, ""))
        if os.path.exists(intr_path):
            K_intr, D_intr, _ = load_intrinsic(intr_path)
            img = undistort_fisheye(img, K_intr, D_intr)
            print(f"Loaded camera '{name}' — image {img.shape[:2]}  (undistorted)")
        else:
            print(f"Loaded camera '{name}' — image {img.shape[:2]}  (no intrinsic, raw)")

        images.append(img)

    if not cam_configs:
        print("No cameras loaded, exiting.")
        return

    drone_yaml = os.path.join(config_dir, "drone.yaml")
    drone = build_drone_camera(drone_yaml)

    outputRes, pxPerM, M = compute_output_shape(drone)
    print(f"BEV output: {outputRes[1]}x{outputRes[0]} px  "
          f"({pxPerM[1]:.2f}, {pxPerM[0]:.2f}) px/m")

    homographies = compute_homographies(cam_configs, M)
    for name, H in zip(camera_names, homographies):
        print(f"\nHomography [{name}]:")
        print(np.array2string(H, precision=6, suppress_small=True))

    masks = compute_invalid_masks(cam_configs, drone, outputRes, pxPerM)
    bev, warped_list = warp_and_stitch(images, cam_configs, masks, outputRes)

    out_path = os.path.join(script_dir, "bev_result.jpg")
    cv2.imwrite(out_path, bev)
    print(f"\nBEV saved to {out_path}")

    cv2.imshow("BEV (stitched)", bev)
    for name, w in zip(camera_names, warped_list):
        cv2.imshow(f"warped_{name}", w)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
