import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from estimators.transformers.tensor_standard_scaler import TensorStandardScaler


class TensorStandardScalerTest(unittest.TestCase):
    x = np.array([
        [1, 2],
        [3, 4]
    ])

    def test_transformBeforeFit_throwsException(self):
        scaler = TensorStandardScaler(axis=0)

        self.assertRaises(NotFittedError, lambda: scaler.transform(TensorStandardScalerTest.x))

    def test_transformWithWrongType_throwsException(self):
        scaler = TensorStandardScaler(axis=0)

        self.assertRaises(Exception, lambda: scaler.transform('str'))

    def test_afterFit_MeansAndStdsAreComputed(self):
        scaler = TensorStandardScaler(axis=0)
        scaler.fit(TensorStandardScalerTest.x)

        self.assertTrue(np.array_equal(scaler.means, np.array((2, 3))))
        self.assertTrue(np.array_equal(scaler.stds, np.array((1, 1))))

    def test_afterTransform_outputHasZeroMeanAndUnitVariance(self):
        n, h, w, d = 100, 64, 64, 3
        axis = (0, 1, 2)
        x = np.random.rand(n, h, w, d)

        scaler = TensorStandardScaler(axis=axis)
        output = scaler.fit_transform(x)

        self.assertTrue(np.allclose(0, output.mean(axis=axis)))
        self.assertTrue(np.allclose(1, output.std(axis=axis)))


if __name__ == '__main__':
    unittest.main()
