from typing import Union, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from estimators.transformers.to_numpy import to_numpy

from sklearn.utils.validation import check_is_fitted


def _check_numpy(x):
    if type(x) != np.ndarray:
        raise Exception(f'x must be a NumPy array, but it is {type(x)}')


class TensorStandardScaler(TransformerMixin, BaseEstimator):
    """
    Similar to sklearn's StandardScaler but for tensors.

    It is needed to specify the axis or axes along which the means and
    standard deviations are computed.

    For example,
     >> x.shape = (n, h, w, c)
     >> scaler = TensorStandardScaler(axis=(0, 1, 2))
     >> z = scaler.fit_transform(x)
     >> np.mean(z[:, :, :, 0])  # 0 can be changed by any channel no.
        0
     >> np.std(z[:, :, :, 0])  # 0 can be changed by any channel no.
        1
    """
    def __init__(self, axis: Union[int, Tuple[int, ...]]):
        self.axis = axis

        self.means = None
        self.stds = None

    def __sklearn_is_fitted__(self):
        return self.means is not None and self.stds is not None

    def fit(self, x: np.ndarray, y=None):
        x = to_numpy(x)

        self.means = x.mean(axis=self.axis)
        self.stds = x.std(axis=self.axis)

        return self

    def transform(self, x: np.ndarray, y=None):
        check_is_fitted(self)
        x = to_numpy(x)

        return (x - self.means) / self.stds
