"""Build, train and evaluate the PointPillars model."""
from model import build_model

import numpy as np


def main():
    batch_size = 8
    point_dimensionality = 9  # D
    points_per_pillar = 16  # N
    number_of_pillars = 64  # P

    dummy_data = np.random.rand(
        batch_size, number_of_pillars, points_per_pillar, point_dimensionality
    )
    model = build_model(number_of_pillars, points_per_pillar, point_dimensionality)
    output = model(dummy_data)
    print(output)


if __name__ == "__main__":
    main()
