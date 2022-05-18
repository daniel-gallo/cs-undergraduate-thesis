import numpy as np
import pandas as pd


def to_numpy(x):
    if type(x) == pd.DataFrame:
        return x.to_numpy()
    elif type(x) == np.ndarray:
        return x

    raise Exception(f'X must be a DataFrame or a NumPy array, but it is {type(x)}')
