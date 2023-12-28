import numpy as np
import numpy.typing as npt
from typing import Tuple
from .utils import (
    sort_pcl_by_height,
    NDArrayFloat64,
    NDArrayFloat32,
    NDArrayBool,
    estimate_plane,
)

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

    def _extract_initial_seeds(self, pointcloud: NDArrayFloat32) -> NDArrayFloat32:
        # mean height value
        lpr_height = pointcloud[: self._num_lpr, 2].mean()

        # Iterate pointcloud, filter those height is less than lpr_height+_th_seeds
        filter_func = pointcloud[:, 2] < lpr_height + self._th_seeds
        seed_pc = pointcloud[filter_func]
        return seed_pc

    def _error_point_removel(self, pointcloud: NDArrayFloat32) -> NDArrayFloat32:
        # Negative outlier error point removal.
        # As there might be some error mirror reflection under the ground
        # We define the outlier threshold times the height of the LiDAR sensor
        filer_func = pointcloud[:, 2] > (
            self._sensor_height_factor * self._sensor_height
        )
        return pointcloud[filer_func]

    def estimate_ground(self, pointcloud: NDArrayFloat32) -> NDArrayBool:
        assert (
            pointcloud.dtype == np.float32
        ), f"Only arrays with type float32 are supported, we got {pointcloud.dtype}"

        sorted_pcl = sort_pcl_by_height(pointcloud)
        cleaned_pcl = self._error_point_removel(sorted_pcl)

        seed_pcl = self._extract_initial_seeds(cleaned_pcl)

        # Initialise the ground plane to the seed
        ground_pcl = seed_pcl

        for idx in range(self._num_iter):
            normal_n, th_dist_d = estimate_plane(ground_pcl, self._th_dist)
            # print(normal_n, th_dist_d)

            # Ground plane model
            results = pointcloud[:, :3] @ normal_n

            ground_idxs = np.array(results < th_dist_d)
            ground_pcl = pointcloud[ground_idxs]

        return ground_idxs
