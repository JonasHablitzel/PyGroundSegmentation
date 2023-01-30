import numpy as np
import numpy.typing as npt
from .utils import sort_pcl_by_height, NDArrayFloat

np.set_printoptions(suppress=True)


class GroundPlaneFitting:
    def __init__(
        self,
        num_seg: int = 1,
        num_iter: int = 3,
        num_lpr: int = 250,
        th_seeds: float = 1.2,
        th_dist: float = 0.3,
        sensor_height: float = 2.0,
        sensor_height_factor: float = -1.5,
    ):

        self._num_seg = num_seg
        self._num_iter = num_iter
        self._num_lpr = num_lpr
        self._th_seeds = th_seeds
        self._th_dist = th_dist
        self._sensor_height = sensor_height
        self._sensor_height_factor = sensor_height_factor

    def _extract_initial_seeds(self, pointcloud: NDArrayFloat) -> NDArrayFloat:

        # mean height value
        lpr_height = pointcloud[: self._num_lpr, 2].mean()

        # Iterate pointcloud, filter those height is less than lpr_height+_th_seeds
        filter_func = pointcloud[:, 2] < lpr_height + self._th_seeds
        seed_pc = pointcloud[filter_func]
        return seed_pc

    def _estimatePlane(self, ground_pcl: NDArrayFloat) -> (NDArrayFloat, float):
        cov_matrix = np.cov(ground_pcl[:, :3], rowvar=False)
        pcl_mean = np.mean(ground_pcl[:, :3], axis=0)
        u, _, _ = np.linalg.svd(cov_matrix, full_matrices=True)

        normal_n = u[:, 2]

        d = -1 * np.inner(normal_n, pcl_mean)

        th_dist_d = self._th_dist - d

        return normal_n, th_dist_d

    def _error_point_removel(self, pointcloud: NDArrayFloat) -> NDArrayFloat:
        # Negative outlier error point removal.
        # As there might be some error mirror reflection under the ground
        # We define the outlier threshold times the height of the LiDAR sensor
        filer_func = pointcloud[:, 2] > (
            self._sensor_height_factor * self._sensor_height
        )
        return pointcloud[filer_func]

    def estimate_ground(self, pointcloud: NDArrayFloat) -> NDArrayFloat:

        sorted_pcl = sort_pcl_by_height(pointcloud)
        cleaned_pcl = self._error_point_removel(sorted_pcl)

        seed_pcl = self._extract_initial_seeds(cleaned_pcl)

        # Initialise the ground plane to the seed
        ground_pcl = seed_pcl

        for idx in range(self._num_iter):
            normal_n, th_dist_d = self._estimatePlane(ground_pcl)
            # print(normal_n, th_dist_d)

            # Ground plane model
            results = pointcloud[:, :3] @ normal_n

            ground_idxs = np.array(results < th_dist_d)
            ground_pcl = pointcloud[ground_idxs]

        return ground_idxs
