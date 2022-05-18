from math import ceil
from typing import Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error

from estimators.predictors.mlp.activations import ReLU, Identity, BaseActivation
from estimators.predictors.mlp.layer import Layer
from estimators.predictors.mlp.optimizers import Momentum, GradientDescent


class MLP(BaseEstimator):
    """
    Custom multilayer perceptron
    """
    def __init__(
            self,
            dim_in: int,
            hidden_layers: Tuple[int, ...],
            dim_out: int,
            alpha: float = 0.1,
            activation: BaseActivation = ReLU,
            final_activation: BaseActivation = Identity,
            optimizer: GradientDescent = Momentum(),
            mini_batch_size: int = 64,
            epochs: int = 200,
            metric=mean_absolute_error,
            verbose: bool = False
    ):
        """
        :param dim_in: number of features.
        :param hidden_layers: tuple whose ith element represent the number of units of the ith hidden layer.
        :param dim_out: number of outputs.
        :param alpha: L2 regularization parameter.
        :param activation: activation function for all but the last layer, e.g., tanh, relu...
        :param final_activation: activation function for the last layer, e.g., identity, softmax...
        :param optimizer: optimizer, e.g. gradient descent with momentum.
        :param mini_batch_size: mini batch size.
        :param epochs: number of epochs (there is no convergence test).
        :param metric: function that returns an error given the true values and the predicted ones. This is only used
            to display the loss on every epoch, the objective function uses MSE (+ L2 penalty).
        :param verbose: True to see the loss on every epoch.
        """
        self.dim_in = dim_in
        self.hidden_layers = hidden_layers
        self.dim_out = dim_out
        self.alpha = alpha
        self.activation = activation
        self.final_activation = final_activation
        self.optimizer = optimizer
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.metric = metric
        self.verbose = verbose

        self._create_layers()

        self.losses = []
        self.val_losses = []

    def _create_layers(self):
        dimensions = (self.dim_in, *self.hidden_layers, self.dim_out)
        self.layers = [
            Layer(
                d1,
                d2,
                self.activation,
                self.alpha,
                self.optimizer
            ) for d1, d2 in zip(dimensions[:-2], dimensions[1:-1])
        ]
        # The last layer may have a different activation function
        self.layers.append(
            Layer(
                self.hidden_layers[-1],
                self.dim_out,
                self.final_activation,
                self.alpha,
                self.optimizer
            )
        )

    def set_params(self, **params):
        for param in params:
            if hasattr(self, param):
                setattr(self, param, params[param])

        self._create_layers()

        return self

    def predict(self, x):
        h = x
        for layer in self.layers:
            h = layer.forward(h)

        return h

    def _compute_loss(self, x, y, validation_data, epoch):
        loss = self.metric(y, self.predict(x))
        self.losses.append(loss)
        if self.verbose:
            print(f'Epoch {epoch}: {loss}', end='')

        if validation_data is not None:
            x_val, y_val = validation_data
            val_loss = self.metric(y_val, self.predict(x_val))
            self.val_losses.append(val_loss)
            if self.verbose:
                print(f' ({val_loss})', end='')

        if self.verbose:
            print()

    def fit(self, x, y, validation_data=None):
        """
        If verbose is set to True and validation data is provided (as a tuple (X_val, y_val)),
        the validation loss will be shown on every epoch.
        """
        self.losses = []
        self.val_losses = []
        if y.ndim == 1:
            y = y[:, np.newaxis]

        for epoch in range(self.epochs):
            self._compute_loss(x, y, validation_data, epoch)

            idx = np.array(range(len(x)))
            np.random.shuffle(idx)
            num_mini_batches = ceil(len(x) // self.mini_batch_size)
            for i in range(num_mini_batches):
                x_mini = x[idx[self.mini_batch_size * i:self.mini_batch_size * (i + 1)]]
                y_mini = y[idx[self.mini_batch_size * i:self.mini_batch_size * (i + 1)]]

                y_hat = self.predict(x_mini)

                # Backward pass (using MSE)
                upstream_gradient = 2 * (y_hat - y_mini)
                for layer in reversed(self.layers):
                    upstream_gradient = layer.backprop(upstream_gradient)
