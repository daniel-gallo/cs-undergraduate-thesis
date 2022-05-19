import numpy as np

from estimators.predictors.mlp.activations.base import BaseActivation


class ReLU(BaseActivation):
    @staticmethod
    def sigma(x):
        return np.maximum(x, 0)

    @staticmethod
    def sigma_prime(x):
        derivative = np.ones_like(x)
        derivative[x < 0] = 0

        return derivative

    @staticmethod
    def get_initial_weights_std(fan_in: int, fan_out: int):
        return np.sqrt(4 / (fan_in + fan_out))
