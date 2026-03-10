import numpy as np
import cv2

class CameraUndistorter:
    def __init__(self, K, D, frame_size_wh):
        self.K = K.astype(np.float64)
        self.D = D.astype(np.float64).reshape(4, 1)
        self.frame_size = tuple(frame_size_wh)

        self.map1 = None
        self.map2 = None
        self._compute_maps()

    def _compute_maps(self):
        w, h = self.frame_size
        R = np.eye(3)

        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, R, self.K, (w, h), cv2.CV_16SC2
        )

    def ensure_size(self, w, h):
        if (w, h) != self.frame_size:
            self.frame_size = (w, h)
            self._compute_maps()

    def undistort(self, img):
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

    def get_K(self):
        return self.K

    def get_zero_distortion(self):
        return np.zeros((5, 1), dtype=np.float64)

class CameraLidarExtrinsics:
    @staticmethod
    def _euler_to_rotation(ax, ay, az):
        cx, sx = np.cos(ax), np.sin(ax)
        cy, sy = np.cos(ay), np.sin(ay)
        cz, sz = np.cos(az), np.sin(az)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
        return Rz @ Ry @ Rx

    def __init__(self, R_opencv, t_opencv, R_robot, t_robot, lidar_rotation: dict):
        self.lidar_rotation = lidar_rotation

        R_lidar = self._euler_to_rotation(
            lidar_rotation["axis_x"],
            lidar_rotation["axis_y"],
            lidar_rotation["axis_z"],
        )

        self.R_opencv = (R_opencv.astype(np.float64).reshape(3, 3) @ R_lidar.T)
        self.t_opencv = t_opencv.astype(np.float64).reshape(3, 1)
        self.R_robot  = (R_robot.astype(np.float64).reshape(3, 3) @ R_lidar.T)
        self.t_robot  = t_robot.astype(np.float64).reshape(3, 1)

class IpmCameraConfig:
  def __init__(self, fx, fy, px, py, yaw, pitch, roll, XCam, YCam, ZCam):
    self.K = np.zeros([3, 3])
    self.R = np.zeros([3, 3])
    self.t = np.zeros([3, 1])
    self.P = np.zeros([3, 4])
    self.yaw = yaw  # store for invalid_mask (camera viewing direction in BEV, degrees)
    self.homography_matrix = None  # set by IpmModule.calculate_ipm(): image -> BEV warp

    self.setK(fx, fy, px, py)
    self.setR(np.deg2rad(yaw), np.deg2rad(pitch), np.deg2rad(roll))
    self.setT(XCam, YCam, ZCam)
    self.updateP()

  def setK(self, fx, fy, px, py):
    self.K[0, 0] = fx
    self.K[1, 1] = fy
    self.K[0, 2] = px
    self.K[1, 2] = py
    self.K[2, 2] = 1.0

  def setR(self, y, p, r):
    # Intrinsic ZYX: body rotation = Rz(yaw)·Ry(pitch)·Rx(roll)
    # Inverse (vehicle → body) = Rx(-roll)·Ry(-pitch)·Rz(-yaw)
    # Rs converts vehicle axes → OpenCV camera axes
    Rz = np.array([[np.cos(-y), -np.sin(-y), 0.0], [np.sin(-y), np.cos(-y), 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[np.cos(-p), 0.0, np.sin(-p)], [0.0, 1.0, 0.0], [-np.sin(-p), 0.0, np.cos(-p)]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(-r), -np.sin(-r)], [0.0, np.sin(-r), np.cos(-r)]])
    Rs = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    self.R = Rs.dot(Rx.dot(Ry.dot(Rz)))

  def setT(self, XCam, YCam, ZCam):
    X = np.array([XCam, YCam, ZCam])
    self.t = -self.R.dot(X)

  def updateP(self):
    Rt = np.zeros([3, 4])
    Rt[0:3, 0:3] = self.R
    Rt[0:3, 3] = self.t
    self.P = self.K.dot(Rt)