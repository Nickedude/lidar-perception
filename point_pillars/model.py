"""Keras implementation of PointPillars."""
import tensorflow as tf


class FeatureNetwork(tf.keras.Sequential):
    def __init__(
        self,
        height: int,
        width: int,
        number_of_pillars: int,
        points_per_pillar: int,
        point_dimensionality: int,
        number_of_features: int = 128,
    ):
        """Build pillar feature network.

        Assumes input tensor of shape B x P x N x D.
        Outputs tensor of shape B x H x W x C.
        Where B is the batch size.

        Args:
            height: height of pseudo-image, H
            width: width of pseudo-image
            number_of_pillars: number of pillars, P
            points_per_pillar: number of points per pillar, N
            point_dimensionality: number of features per point, D
            number_of_features: number of features, C

        """
        super().__init__(name="feature_network")
        self._height = height
        self._width = width
        self._num_features = number_of_features

        self.add(
            tf.keras.Input(
                shape=(
                    number_of_pillars,
                    points_per_pillar,
                    point_dimensionality,
                )
            )
        )
        self.add(tf.keras.Input(shape=(number_of_pillars, 2)))
        # Fully connected layer is implemented using 1x1 convolution
        self.add(tf.keras.layers.Conv2D(filters=number_of_features, kernel_size=1))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.ReLU())

    def call(self, inputs, training=None, mask=None):
        data, indices = inputs
        batch_size, num_pillars, _, _ = data.shape
        assert indices.shape == (batch_size, num_pillars, 3)

        outputs = super().call(data, training, mask)  # B x P x N x C
        outputs = tf.math.reduce_max(
            outputs, axis=2
        )  # Max reduction across points -> B x P x C

        output_shape = (batch_size, self._height, self._width, self._num_features)
        outputs = tf.scatter_nd(
            indices, outputs, output_shape
        )  # Scatter to grid -> B x H x W x C

        return outputs


def build_model(
    height: int,
    width: int,
    number_of_pillars: int,
    points_per_pillar: int,
    point_dimensionality: int,
):
    """Build PointPillar network.

    Args:
        height: height of pseudo-image, H
        width: width of pseudo-image: W
        number_of_pillars: number of pillars, P
        points_per_pillar: number of points per pillar, N
        point_dimensionality: number of features per point, D

    """
    return FeatureNetwork(
        height, width, number_of_pillars, points_per_pillar, point_dimensionality
    )
