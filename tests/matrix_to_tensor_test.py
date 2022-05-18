import unittest

import numpy as np

from estimators.transformers.matrix_to_tensor import MatrixToTensor


class MatrixToTensorTest(unittest.TestCase):
    def test(self):
        x = np.array([
            [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
        ])

        n, h, w, c = 2, 2, 3, 2
        m2t = MatrixToTensor(h, w, c)
        images = m2t.fit_transform(x)

        self.assertEqual(images.shape, (n, h, w, c))
        self.assertTrue(np.array_equal(
            images[0, :, :, 0],
            np.array([
                [1, 1, 1],
                [2, 2, 2]
            ])
        ))
        self.assertTrue(np.array_equal(
            images[0, :, :, 1],
            np.array([
                [3, 3, 3],
                [4, 4, 4]
            ])
        ))


if __name__ == '__main__':
    unittest.main()
