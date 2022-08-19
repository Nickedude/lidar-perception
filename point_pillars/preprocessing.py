from typing import Tuple

import numpy as np
from tqdm import tqdm


def create_pillars(points: np.ndarray, num_pillars: int, points_per_pillar: int, bin_width: float = 0.16) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a point cloud into a list of pillars.

    This is done by putting all points in a grid in the xy-plane. Each cell in the grid has size
    bin_width ** 2. At most N points will be kept in each pillar. If there are more than N points
    then N points are selected at random.

    Args:
        points: point cloud with shape Nx4, where N is the number of points and the four dimensions
            correspond to the x,y,z coordinates and the reflectance of each point.
        num_pillars: max number of pillars, P
        points_per_pillar: max number of points per pillar, N
        bin_width: width of each bin in meters, each cell will have an area of bin_width ** 2

    Returns:
        the pillars as a P x N x D array
    """
    assert len(points.shape) == 2 and points.shape[1] == 4, "Points must by a Nx4 array"

    max_range = 100
    dimensionality = 9
    all_pillars = np.zeros((max_range * max_range, points_per_pillar, 9))

    x_bins = np.linspace(-max_range / 2, max_range / 2, np.ceil(max_range / bin_width).astype(int))
    y_bins = np.linspace(0, max_range, np.ceil(max_range / bin_width).astype(int))

    x_indices = _get_bin_indices(points[:, 0], x_bins)
    y_indices = _get_bin_indices(points[:, 1], y_bins)

    print("k")


def _get_bin_indices(coordinates: np.array, bins: np.array):
    coordinates = coordinates.reshape((-1, 1))
    mins = bins[:-1]
    maxs = bins[1:]
    is_within_bounds = ((mins <= coordinates) & (coordinates <= maxs)).astype(int)
    indices = is_within_bounds.argmax(axis=1)
    indices[is_within_bounds.sum(axis=1) == 0] = -1
    return indices
