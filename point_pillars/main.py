"""Build, train and evaluate the PointPillars model."""
from model import build_model

import numpy as np


def create_dummy_indices(height: int, width: int, batch_idx: int):
    num_pillars = height * width
    indices = np.zeros((num_pillars, 3)).astype(int)
    p = 0

    for row in range(height):
        for col in range(height):
            indices[p, 0] = batch_idx
            indices[p, 1] = row
            indices[p, 2] = col
            p += 1

    return indices


def main():
    height = 8
    width = 8

    batch_size = 1
    point_dimensionality = 9  # D
    points_per_pillar = 16  # N
    number_of_pillars = height * width  # P

    dummy_data = np.random.rand(
        batch_size, number_of_pillars, points_per_pillar, point_dimensionality
    )
    indices = np.array(
        [create_dummy_indices(height, width, i) for i in range(batch_size)]
    )
    model = build_model(
        height, width, number_of_pillars, points_per_pillar, point_dimensionality
    )
    output = model((dummy_data, indices))
    print(output)


if __name__ == "__main__":
    main()
