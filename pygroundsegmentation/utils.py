import numpy as np
import numpy.typing as npt

NDArrayFloat32 = npt.NDArray[np.float32]
NDArrayFloat64 = npt.NDArray[np.float64]
NDArrayBool = npt.NDArray[np.bool_]


def sort_pcl_by_height(pointcloud: NDArrayFloat32) -> NDArrayFloat32:
    sorted_pcl = pointcloud[pointcloud[:, 2].argsort()]
    return sorted_pcl
