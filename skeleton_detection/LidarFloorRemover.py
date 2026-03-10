import numpy as np


class GroundPlaneRemover:

    def __init__(
        self,
        sensor_height=2.15,
        num_iter=15,
        num_lpr=400,
        th_seeds=0.3,
        th_dist=0.3
    ):

        self.sensor_height = sensor_height
        self.num_iter = num_iter
        self.num_lpr = num_lpr
        self.th_seeds = th_seeds
        self.th_dist = th_dist


    def remove_ground(self, xyz):

        pts = xyz.copy()

        # --------------------------------------------------
        # 1. sort by height
        # --------------------------------------------------
        pts_sorted = pts[np.argsort(pts[:, 2])]

        # remove points that are too low (noise)
        pts_sorted = pts_sorted[pts_sorted[:, 2] > -1.5 * self.sensor_height]

        if len(pts_sorted) == 0:
            return xyz

        # --------------------------------------------------
        # 2. compute LPR height
        # --------------------------------------------------
        lpr = np.mean(pts_sorted[:min(self.num_lpr, len(pts_sorted)), 2])

        # --------------------------------------------------
        # 3. extract seeds
        # --------------------------------------------------
        seeds = pts_sorted[pts_sorted[:, 2] < lpr + self.th_seeds]

        if len(seeds) < 3:
            return xyz

        # --------------------------------------------------
        # 4. iterative plane estimation
        # --------------------------------------------------
        ground = seeds

        for _ in range(self.num_iter):

            centroid = np.mean(ground, axis=0)

            cov = np.cov((ground - centroid).T)

            U, S, Vt = np.linalg.svd(cov)

            normal = U[:, 2]

            d = -normal.dot(centroid)

            dist = xyz @ normal + d

            ground = xyz[dist < self.th_dist]

            if len(ground) < 3:
                break

        # --------------------------------------------------
        # 5. remove ground
        # --------------------------------------------------
        dist = xyz @ normal + d

        mask = dist > self.th_dist

        return xyz[mask]