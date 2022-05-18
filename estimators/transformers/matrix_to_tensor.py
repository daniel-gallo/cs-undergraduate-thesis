import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from estimators.transformers.to_numpy import to_numpy


class MatrixToTensor(BaseEstimator, TransformerMixin):
    """
    Converts tabular data to tensor form.

    The features must be sorted first by channel, then by height and finally by width.
    The output has NHWC format (the output dimension is num_samples x height x width x channels).
    """
    def __init__(self, height: int, width: int, num_channels: int):
        self.height = height
        self.width = width
        self.num_channels = num_channels

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        x = to_numpy(x)

        num_samples, num_features = x.shape
        assert (num_features == self.height * self.width * self.num_channels)
        image_size = self.height * self.width

        x_new = np.zeros((num_samples, self.height, self.width, self.num_channels))
        for channel in range(self.num_channels):
            images = x[:, image_size * channel:image_size * (channel + 1)]
            images = images.reshape(-1, self.height, self.width)
            x_new[:, :, :, channel] = images

        return x_new
