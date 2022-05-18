import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras_tuner import HyperParameters
from scikeras.wrappers import KerasRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from data_manager import DataManager
from estimators.predictors.mlp.mlp import MLP
from estimators.transformers.matrix_to_tensor import MatrixToTensor
from estimators.transformers.tensor_standard_scaler import TensorStandardScaler


def clip(get_regressor):
    def clip_wrapper():
        return TransformedTargetRegressor(
            regressor=get_regressor(),
            inverse_func=lambda x: np.clip(np.squeeze(x), 0, 1),
            check_inverse=False
        )

    return clip_wrapper


@clip
def get_ridge():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1400))
    ])


@clip
def get_lasso():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Lasso(alpha=2 ** -11, max_iter=2000))
    ])


@clip
def get_svr():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR(C=2 ** -3, epsilon=2 ** -9, gamma=2 ** -13))
    ])


@clip
def get_mlp():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', MLPRegressor(hidden_layer_sizes=(150, 150, 150), alpha=0.5))
    ])


@clip
def get_custom_mlp():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', MLP(3480, (100, 100), 1, alpha=1, epochs=100))
    ])


def _get_wrapper(get_model, epochs, min_delta=0.001, patience=30):
    return KerasRegressor(
        get_model,
        optimizer='adam',
        epochs=epochs,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=min_delta,
                patience=patience,
                restore_best_weights=True
            )
        ]
    )


relu_kwargs = {
    'activation': tf.keras.activations.relu,
    'kernel_initializer': tf.keras.initializers.HeUniform,
}


def _get_dropout_layers(rate, units, c=None):
    kwargs = {}
    if c is not None:
        kwargs['kernel_constraint'] = tf.keras.constraints.MaxNorm(max_value=c)

    layers = [tf.keras.layers.Dropout(rate), tf.keras.layers.Dense(units, **relu_kwargs, **kwargs)]

    return layers


def get_dropout_model_from_parameters(hp: kt.HyperParameters):
    input_rate = hp.Float('input_rate', 0, .5)
    hidden_rate = hp.Float('hidden_rate', 0, .5)
    num_layers = hp.Int('num_layers', 1, 3)
    units_per_layer = hp.Int('units_per_layer', 50, 300, 50)

    layers = []
    layers.extend(_get_dropout_layers(input_rate, units_per_layer))
    for _ in range(num_layers - 1):
        layers.extend(_get_dropout_layers(hidden_rate, units_per_layer))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(np.prod(DataManager.MAP_SHAPE),)),

        *layers,

        tf.keras.layers.Dense(1, activation='linear'),
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.mean_absolute_error
    )

    return model


def get_dropout_model():
    hp = kt.HyperParameters()
    hp.values['input_rate'] = 0.1
    hp.values['hidden_rate'] = 0.5
    hp.values['num_layers'] = 2
    hp.values['units_per_layer'] = 150

    return get_dropout_model_from_parameters(hp)


@clip
def get_dropout():
    model_wrapper = _get_wrapper(get_dropout_model, 50, patience=50)

    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model_wrapper)
    ])


def get_cnn_model_from_parameters(hp: HyperParameters):
    conv_alpha = hp.Float('conv_alpha', 1e-6, 1e4, sampling='log')
    filter_size = hp.Choice('filter_size', [3, 5])
    filter_size_offset = hp.Choice('filter_size_offset', [0, 2])
    num_filters = hp.Choice('num_filters', [8, 16, 32])
    num_filters_factor = hp.Choice('num_filters_factor', [1, 2])

    num_units = hp.Choice('dense_units', [50, 100, 150])
    dense_alpha = hp.Float('dense_alpha', 1e-6, 1e4, sampling='log')

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=DataManager.MAP_SHAPE),

        tf.keras.layers.Conv2D(
            num_filters,
            (filter_size, filter_size + filter_size_offset),
            kernel_regularizer=tf.keras.regularizers.l2(conv_alpha),
            **relu_kwargs),
        tf.keras.layers.Conv2D(
            num_filters * num_filters_factor,
            (filter_size, filter_size + filter_size_offset),
            kernel_regularizer=tf.keras.regularizers.l2(conv_alpha),
            **relu_kwargs),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            num_units,
            kernel_regularizer=tf.keras.regularizers.l2(dense_alpha),
            **relu_kwargs),
        tf.keras.layers.Dense(
            num_units,
            kernel_regularizer=tf.keras.regularizers.l2(dense_alpha),
            **relu_kwargs),

        tf.keras.layers.Dense(1, activation='linear'),
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.mean_absolute_error
    )

    return model


def get_cnn_model():
    hp = kt.HyperParameters()
    hp.values['conv_alpha'] = 3.17e-6
    hp.values['filter_size'] = 5
    hp.values['filter_size_offset'] = 2
    hp.values['num_filters'] = 8
    hp.values['num_filters_factor'] = 2
    hp.values['dense_units'] = 150
    hp.values['dense_alpha'] = 0.058

    return get_cnn_model_from_parameters(hp)


@clip
def get_cnn():
    model_wrapper = _get_wrapper(get_cnn_model, 20)

    return Pipeline([
        ('m2t', MatrixToTensor(*DataManager.MAP_SHAPE)),
        ('scaler', TensorStandardScaler(axis=(0, 1, 2))),
        ('regressor', model_wrapper)
    ])


models_by_name = {
    'ridge': get_ridge,
    'lasso': get_lasso,
    'svr': get_svr,
    'custom_mlp': get_custom_mlp,
    'mlp': get_mlp,
    'dropout': get_dropout,
    'cnn': get_cnn,
}
