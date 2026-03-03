import numpy as np

class PersonDetectionModule:
    def __init__(self):
        self.lidar_camera_extrinsics_array = []
        self.camera_array_intrinsics = []
        self.ipmCameras = []

    def load_camera_lidar_parameters(self, dataset):
        self.lidar_camera_extrinsics_array = dataset.load_camera_lidar_extrinsics_array()
        self.camera_array_intrinsics = dataset.load_camera_array_intrinsics()
        self.ipmCameras = dataset.load_ipm_camera_configs()

    def get_camera_images(self, images: list[np.ndarray | None]):
        for i, config in enumerate(self.ipmCameras):
            if config is None:
                continue

            if i >= len(images) or images[i] is None:
                continue

            img = images[i]

            # print image shape
            print(f"------------------------> person Image shape: {img.shape}")
            