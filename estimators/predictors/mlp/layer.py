from copy import copy

import numpy as np

from estimators.predictors.mlp.activations import BaseActivation
from estimators.predictors.mlp.optimizers import GradientDescent


class Layer:
    def __init__(
            self,
            fan_in: int,
            fan_out: int,
            activation: BaseActivation,
            alpha: float,
            optimizer: GradientDescent
    ):
        """
        Layer of an MLP.

        :param fan_in: number of input neurons.
        :param fan_out: number of output neurons.
        :param activation: activation function.
        :param alpha: L2 regularization parameter.
        :param optimizer: optimizer, e.g. gradient descent with momentum.
        """
        weight_std = activation.get_initial_weights_std(fan_in, fan_out)
        self.weights = np.random.randn(fan_in, fan_out) * weight_std
        self.weights_optimizer = copy(optimizer)
        self.biases = np.random.randn(fan_out) * weight_std
        self.biases_optimizer = copy(optimizer)

        self.activation = activation.sigma
        self.activation_prime = activation.sigma_prime

        self.alpha = alpha

        self.input_data = None
        self.z = None

    def forward(self, input_data: np.ndarray):
        """
        Applies the activation function to an affine transformation of the input. It also stores the input and the
        result of the affine transformation (before the activation).
        """
        self.input_data = input_data

        self.z = input_data @ self.weights + self.biases
        h = self.activation(self.z)

        return h

    def backprop(self, dl_dh: np.ndarray):
        """
        Receives dl_dh and returns dl_d(h - 1), updating the weights.

        :param dl_dh: the upstream gradient.
        :return: dl_d(h - 1), that will be the upstream gradient of the previous layer.
        """
        dl_dz = dl_dh * self.activation_prime(self.z)

        new_dl_dh = dl_dz @ self.weights.T
        dl_dw = self.input_data.T @ dl_dz
        dl_db = dl_dz.sum(axis=0)

        # L2 penalty
        weights_gradient = dl_dw + self.alpha * self.weights
        bias_gradient = dl_db + self.alpha * self.biases

        self.weights = self.weights_optimizer.get_updated_weights(weights_gradient, self.weights)
        self.biases = self.biases_optimizer.get_updated_weights(bias_gradient, self.biases)

        return new_dl_dh
