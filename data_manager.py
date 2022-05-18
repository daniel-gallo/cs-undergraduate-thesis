import numpy as np
import pandas as pd


class DataManager:
    TRAIN_PATH = 'data/data_target_stv_2016.csv'
    VALIDATION_PATH = 'data/data_target_stv_2017.csv'
    TEST_PATH = 'data/data_target_stv_2018.csv'

    MAP_HEIGHT = 15
    MAP_WIDTH = 29
    MAP_CHANNELS = 8
    MAP_SHAPE = (MAP_HEIGHT, MAP_WIDTH, MAP_CHANNELS)

    _cache = {}

    @classmethod
    def _get_data_and_target(cls, path):
        if path not in cls._cache:
            df = pd.read_csv(path)

            x = df.drop(columns=['prediction date', 'targ'])
            y = df['targ'].to_numpy()

            cls._cache[path] = (x, y)

        return cls._cache[path]

    @classmethod
    def get_train(cls):
        return cls._get_data_and_target(cls.TRAIN_PATH)

    @classmethod
    def get_validation(cls):
        return cls._get_data_and_target(cls.VALIDATION_PATH)

    @classmethod
    def get_train_validation(cls):
        X_train, y_train = cls.get_train()
        X_val, y_val = cls.get_validation()

        X = pd.concat((X_train, X_val))
        y = np.concatenate((y_train, y_val))

        return X, y

    @classmethod
    def get_test(cls):
        return cls._get_data_and_target(cls.TEST_PATH)
