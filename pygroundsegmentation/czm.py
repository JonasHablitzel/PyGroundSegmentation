from dataclasses import dataclass
from typing import Tuple
import numpy as np
from .utils import NDArrayFloat32


@dataclass
class ConcentricZoneModel:
    num_min_pts: int = (
        10  # Minimum number of points to be estimated as ground plane in each patch.
    )
    max_range: float = 80.0  # max_range of ground estimation area
    min_range: float = 2.7  # min_range of ground estimation area

    pre_sort: bool = True  # If the sub arrays in the czm should be sorted or not
    num_zones: int = 4  # Setting of Concentric Zone Model(CZM)
    num_sectors_each_zone: Tuple[int] = (
        16,
        32,
        54,
        32,
    )  # Setting of Concentric Zone Model(CZM)
    num_rings_each_zone: Tuple[int] = (
        2,
        4,
        4,
        4,
    )  # Setting of Concentric Zone Model(CZM)

    def __post_init__(self):
        assert (
            len(self.num_sectors_each_zone) == 4
        ), f"num_sectors_each_zone must have the length for we got: {len(self.num_sectors_each_zone)}"
        assert (
            len(self.num_rings_each_zone) == 4
        ), f"num_rings_each_zone must have the length for we got: {len(self.num_rings_each_zone)}"

        self.num_sectors = len(self.num_rings_each_zone)
        self.min_range_z2 = (7 * self.min_range + self.max_range) / 8.0
        self.min_range_z3 = (3 * self.min_range + self.max_range) / 4.0
        self.min_range_z4 = (self.min_range + self.max_range) / 2.0
        self.min_ranges = (
            self.min_range,
            self.min_range_z2,
            self.min_range_z3,
            self.min_range_z4,
        )
        self.ring_sizes = (
            (self.min_range_z2 - self.min_range) / self.num_rings_each_zone[0],
            (self.min_range_z3 - self.min_range_z2) / self.num_rings_each_zone[1],
            (self.min_range_z4 - self.min_range_z3) / self.num_rings_each_zone[2],
            (self.max_range - self.min_range_z4) / self.num_rings_each_zone[3],
        )
        self.zone_sizes = (
            2 * np.pi / self.num_sectors_each_zone[0],
            2 * np.pi / self.num_sectors_each_zone[1],
            2 * np.pi / self.num_sectors_each_zone[2],
            2 * np.pi / self.num_sectors_each_zone[3],
        )

    def add_pointcloud(self, pointcloud: NDArrayFloat32):
        self._num_points = pointcloud.shape[0]
        pointcloud_x = pointcloud[:, 0]
        pointcloud_y = pointcloud[:, 1]
        pointcloud_z = pointcloud[:, 2]
        pointcloud_i = pointcloud[:, 3]

        self._pointcloud_x = pointcloud_x
        self._pointcloud_y = pointcloud_y
        self._pointcloud_z = pointcloud_z
        self._pointcloud_intensity = pointcloud_i

        double_pi = 2 * np.pi
        self._radius = np.sqrt(
            pointcloud_x * pointcloud_x + pointcloud_y * pointcloud_y
        )
        self._theta = np.arctan2(pointcloud_y, pointcloud_x)
        self._theta[self._theta < 0] += double_pi
        self.pointcloud_mask = np.full((self._num_points), True, dtype=bool)

        num_points = self._num_points
        radius = self._radius
        theta = self._theta

        within_range_mask = (
            (radius <= self.max_range)
            & (radius > self.min_range)
            & self.pointcloud_mask
        )

        subzone_1_mask = radius < self.min_ranges[1]
        subzone_2_mask = radius < self.min_ranges[2]
        subzone_3_mask = radius < self.min_ranges[3]

        zone_1_mask = (subzone_1_mask) & within_range_mask
        zone_2_mask = (subzone_2_mask & ~subzone_1_mask) & within_range_mask
        zone_3_mask = (subzone_3_mask & ~subzone_2_mask) & within_range_mask
        zone_4_mask = (
            ~(subzone_1_mask | subzone_2_mask | subzone_3_mask) & within_range_mask
        )

        self._zone_masks = (
            zone_1_mask,
            zone_2_mask,
            zone_3_mask,
            zone_4_mask,
        )
        self._ring_idxs = np.full(num_points, -1, dtype=np.intc)
        self._sector_idxs = np.full(num_points, -1, dtype=np.intc)

        # Calculate for the first RING
        for zone_idx in range(self.num_sectors):
            zone_mask = self._zone_masks[zone_idx]
            masked_r = radius[zone_mask]
            masked_theta = theta[zone_mask]
            self._ring_idxs[zone_mask] = np.minimum(
                (
                    (masked_r - self.min_ranges[zone_idx]) / self.ring_sizes[zone_idx]
                ).astype(int),
                self.num_rings_each_zone[zone_idx] - 1,
            )
            self._sector_idxs[zone_mask] = np.minimum(
                (masked_theta / self.zone_sizes[zone_idx]).astype(int),
                self.num_sectors_each_zone[zone_idx] - 1,
            )

        self._init_czm_maps_v1(pointcloud)

    def _init_czm_maps_v1(self, pointcloud):
        # Precompute the values
        _precomputed_ring_masks = []
        for ring_idx in range(max(self.num_rings_each_zone)):
            _precomputed_ring_masks.append(self._ring_idxs == ring_idx)
        _precomputed_ring_arr = np.row_stack(_precomputed_ring_masks)
        _precomputed_ring_arr = _precomputed_ring_arr.T
        self._precomputed_ring_arr = _precomputed_ring_arr

        _precomputed_sector_masks = []
        for sector_idx in range(max(self.num_sectors_each_zone)):
            _precomputed_sector_masks.append(self._sector_idxs == sector_idx)
        _precomputed_sector_arr = np.row_stack(_precomputed_sector_masks)
        _precomputed_sector_arr = _precomputed_sector_arr.T
        self._precomputed_sector_arr = _precomputed_sector_arr

        self._precomputed_ring_masks = _precomputed_ring_masks
        self._precomputed_sector_masks = _precomputed_sector_masks
        self.czm = {}

        for zone_idx in range(self.num_zones):
            zone_mask = self._zone_masks[zone_idx]
            for ring_idx in range(self.num_rings_each_zone[zone_idx]):
                ring_mask = _precomputed_ring_masks[ring_idx]
                for sector_idx in range(self.num_sectors_each_zone[zone_idx]):
                    mask_idxs = np.nonzero(
                        zone_mask & ring_mask & _precomputed_sector_masks[sector_idx]
                    )[0]
                    if mask_idxs.shape[0] < self.num_min_pts:
                        continue

                    sub_pcl = pointcloud[mask_idxs].copy()
                    if self.pre_sort:
                        sub_pcl_sorted = sub_pcl[sub_pcl[:, 2].argsort()]
                        self.czm[(zone_idx, ring_idx, sector_idx)] = sub_pcl_sorted
                    else:
                        self.czm[(zone_idx, ring_idx, sector_idx)] = sub_pcl
