import numpy as np

from estimators.predictors.mlp.activations.base import BaseActivation


class Identity(BaseActivation):
    @staticmethod
    def sigma(x):
        return x

    @staticmethod
    def sigma_prime(x):
        return np.ones_like(x)

    @staticmethod
    def get_initial_weights_std(fan_in: int, fan_out: int):
        return np.sqrt(2 / (fan_in + fan_out))
