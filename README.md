# PyGroundSegmentation

This libary includes some ground segmentation algorithems rewritten in python. There are no external C or C++ dependencies only pure python (numpy)

# Installation

```bash
pip install pygroundsegmentation
```

## Included Algorithems

- [x] [GPF](https://github.com/VincentCheungM/Run_based_segmentation) (Ground Plane Fitting)
- [ ] [Patchwork](https://github.com/LimHyungTae/patchwork) (ver.1)

## Example Usage

```python
from pygroundsegmentation import GroundPlaneFitting

ground_estimator = GroundPlaneFitting() #Instantiate one of Groundestimators

xyz_pointcloud = np.random.rand(1000,3) #Example Pointcloud
ground_idxs = ground_estimator.estimate_ground(xyz_pointcloud)
ground_pcl = xyz_pointcloud[ground_idxs]

```
