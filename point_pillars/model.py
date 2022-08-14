"""Keras implementation of PointPillars."""
from typing import Tuple

import tensorflow as tf


class ReduceMax(tf.keras.layers.Layer):
    """Convenience class for wrapping reduce_max in a layer."""

    def __init__(self, axis: int = -1):
        """Constructor.

        Args:
            axis: the axis to reduce

        """
        super().__init__(trainable=False, name="reduce_max")
        self._axis = axis

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Apply max reduction to inputs."""
        return tf.math.reduce_max(inputs, self._axis)


def _build_pillar_feature_network(
    number_of_pillars: int,
    points_per_pillar: int,
    point_dimensionality: int,
    number_of_features: int = 128,
) -> tf.keras.Sequential:
    """Build pillar feature network.

    Assumes input tensor of shape B x P x N x D.
    Outputs tensor of shape B x P x C.
    Where B is the batch size.

    Args:
        number_of_pillars: number of pillars, P
        points_per_pillar: number of points per pillar, N
        point_dimensionality: number of features per point, D
        number_of_features: number of features, C

    """

    feature_network = tf.keras.models.Sequential()
    feature_network.add(
        tf.keras.Input(
            shape=(
                number_of_pillars,
                points_per_pillar,
                point_dimensionality,
            )
        )
    )
    feature_network.add(tf.keras.layers.Dense(units=number_of_features))
    feature_network.add(tf.keras.layers.BatchNormalization())
    feature_network.add(tf.keras.layers.ReLU())
    feature_network.add(ReduceMax(axis=2))

    return feature_network


def build_model(
    number_of_pillars: int, points_per_pillar: int, point_dimensionality: int
):
    """Build PointPillar network.

    Args:
        number_of_pillars: number of pillars, P
        points_per_pillar: number of points per pillar, N
        point_dimensionality: number of features per point, D

    """
    return _build_pillar_feature_network(
        number_of_pillars, points_per_pillar, point_dimensionality
    )
