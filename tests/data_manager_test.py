import unittest

import numpy as np
import pandas as pd

from data_manager import DataManager


class TestDataManager(unittest.TestCase):
    def test_types(self):
        x, y = DataManager.get_train()

        self.assertEqual(type(x), pd.DataFrame)
        self.assertEqual(type(y), np.ndarray)

    def test_dimensions(self):
        x, y = DataManager.get_train()
        n, d = x.shape

        self.assertEqual(n, len(y))
        self.assertEqual(d, np.prod(DataManager.MAP_SHAPE))

    def test_cache(self):
        DataManager.get_train()

        self.assertGreater(len(DataManager._cache), 0)


if __name__ == '__main__':
    unittest.main()
