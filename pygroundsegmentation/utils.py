import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float32]


def sort_pcl_by_height(pointcloud: NDArrayFloat) -> NDArrayFloat:
    sorted_pcl = pointcloud[pointcloud[:, 2].argsort()]
    return sorted_pcl
