import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_manager import DataManager
from estimators.transformers.matrix_to_tensor import MatrixToTensor
from estimators.transformers.tensor_standard_scaler import TensorStandardScaler
from models import get_ridge, get_lasso, get_svr, get_mlp, get_cnn_model_from_parameters, \
    get_dropout_model_from_parameters


def _run_grid_search(estimator_name, estimator, param_grid):
    x_train, y_train = DataManager.get_train()
    x_val, y_val = DataManager.get_validation()

    x = pd.concat((x_train, x_val))
    y = np.concatenate((y_train, y_val))
    combined_fold = np.concatenate((-np.ones(len(x_train)), np.zeros(len(x_val))))
    predefined_split = PredefinedSplit(test_fold=combined_fold)

    grid_search = GridSearchCV(
        estimator,
        param_grid,
        cv=predefined_split,
        scoring='neg_mean_absolute_error',
        verbose=4,
        n_jobs=-1
    )
    grid_search.fit(x, y)
    df = pd.DataFrame.from_dict(grid_search.cv_results_)
    df.to_csv(f'hp_search_results/{estimator_name}.csv', index=False)


def _run_bayesian_optimization(
        estimator_name,
        hypermodel,
        is_tabular,
        max_trials=200,
        max_epochs=1000,
        min_delta=0.001,
        patience=30,
):
    x_train, y_train = DataManager.get_train()
    x_val, y_val = DataManager.get_validation()

    if is_tabular:
        scaler = StandardScaler()

        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
    else:
        pipeline = Pipeline([
            ('m2t', MatrixToTensor(*DataManager.MAP_SHAPE)),
            ('scaler', TensorStandardScaler(axis=(0, 1, 2)))
        ])
        x_train = pipeline.fit_transform(x_train)
        x_val = pipeline.transform(x_val)

    tuner = kt.BayesianOptimization(
        hypermodel=hypermodel,
        objective='val_loss',
        max_trials=max_trials,
        directory=f'hp_search_results/{estimator_name}',
        project_name=estimator_name
    )

    tuner.search(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=max_epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=min_delta,
                patience=patience,
            )
        ]
    )


def ridge_gs():
    param_grid = dict(regressor__regressor__alpha=[2 ** k for k in range(-4, 20)])

    _run_grid_search('ridge', get_ridge(), param_grid)


def lasso_gs():
    param_grid = dict(regressor__regressor__alpha=[2 ** k for k in range(-22, 2)])

    _run_grid_search('lasso', get_lasso(), param_grid)


def svr_gs():
    param_grid = dict(
        regressor__regressor__C=[2 ** k for k in range(-9, 1)],
        regressor__regressor__epsilon=[2 ** k for k in range(-15, -5)],
        regressor__regressor__gamma=[2 ** k for k in range(-16, -6)],
    )

    _run_grid_search('svr', get_svr(), param_grid)


def mlp_gs():
    architectures = []
    for num_hidden_layers in (1, 2, 3):
        for units_per_layer in (50, 100, 150):
            architectures.append([units_per_layer] * num_hidden_layers)

    param_grid = dict(
        regressor__regressor__hidden_layers=architectures,
        regressor__regressor__alpha=[2 ** k for k in range(-16, 8)],
    )

    _run_grid_search('mlp', get_mlp(), param_grid)


def custom_mlp_gs():
    param_grid = dict(
        regressor__regressor__alpha=[2 ** k for k in range(-16, 8)],
    )

    _run_grid_search('mlp', get_mlp(), param_grid)


def dropout_hs():
    _run_bayesian_optimization('dropout', get_dropout_model_from_parameters, is_tabular=True)


def cnn_hs():
    _run_bayesian_optimization('cnn', get_cnn_model_from_parameters, is_tabular=False)
