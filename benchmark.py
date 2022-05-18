from timeit import default_timer as timer

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error

from data_manager import DataManager
from formatter import export
from models import models_by_name


def generate_pred_vs_actual_plot(y_true, y_pred, model_name):
    plt.xlabel('Actual value')
    plt.ylabel('Predicted value')
    plt.scatter(y_true, y_pred)
    export(f'{model_name}_pva')
    plt.close()
    plt.cla()
    plt.clf()


if __name__ == '__main__':
    X_train, y_train = DataManager.get_train()
    X_val, y_val = DataManager.get_validation()
    X_test, y_test = DataManager.get_test()

    for model_name, get_model in models_by_name.items():
        print(f'\n[*] Going with {model_name}...')

        print('\tFitting training...')
        model = get_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        train_error = mean_absolute_error(y_train, y_pred)
        print(f'\t\tTrain error: {train_error:.5f}')
        y_pred = model.predict(X_val)
        val_error = mean_absolute_error(y_val, y_pred)
        print(f'\t\tEvaluation error: {val_error:.5f}')

        print('\tFitting training + validation')
        model = get_model()

        start = timer()
        model.fit(pd.concat((X_train, X_val)), np.concatenate((y_train, y_val)))
        end = timer()
        training_time = end - start

        start = timer()
        y_pred = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        end = timer()
        prediction_time = end - start
        generate_pred_vs_actual_plot(y_test, y_pred, model_name)

        print(f'\t\tTest error: {test_mae:.5f}')
        print(f'\t\tTraining time: {training_time:.2f} s')
        print(f'\t\tPrediction time: {prediction_time:.2f} s')
