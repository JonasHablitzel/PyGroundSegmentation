import numpy as np
from typing import Tuple
import numpy.typing as npt

NDArrayFloat32 = npt.NDArray[np.float32]
NDArrayFloat64 = npt.NDArray[np.float64]
NDArrayBool = npt.NDArray[np.bool_]


def sort_pcl_by_height(pointcloud: NDArrayFloat32) -> NDArrayFloat32:
    sorted_pcl = pointcloud[pointcloud[:, 2].argsort()]
    return sorted_pcl


def estimate_plane(
    ground_pcl: NDArrayFloat32, th_dist: float
) -> Tuple[NDArrayFloat64, float]:
    cov_matrix = np.cov(ground_pcl[:, :3], rowvar=False)
    pcl_mean = np.mean(ground_pcl[:, :3], axis=0)
    u, _, _ = np.linalg.svd(cov_matrix, full_matrices=True)

    normal_n = u[:, 2]

    d = -1 * np.inner(normal_n, pcl_mean)

    th_dist_d = th_dist - d

    return normal_n, th_dist_d
