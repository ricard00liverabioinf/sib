import unittest
import numpy as np
from si.data.dataset import Dataset
from si.feature_selection.select_percentile import SelectPercentile


class TestSelectPercentile(unittest.TestCase):

    def test_select_percentile(self):
        # Dataset fictício com 20 amostras e 10 features
        np.random.seed(42)
        X = np.random.rand(20, 10)
        y = np.random.randint(0, 2, size=20)
        dataset = Dataset(X, y)

        # Selecionar 40% das features (40% de 10 = 4 features)
        selector = SelectPercentile(percentile=40)
        selector.fit(dataset)
        new_ds = selector.transform(dataset)

        self.assertEqual(new_ds.X.shape[1], 4)
        self.assertEqual(new_ds.X.shape[0], 20)