import logging
import sys
import unittest

import numpy as np
import pandas as pd

from utils import DataLoader


class PreProcessTestCase(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.df = pd.read_csv('resources/raw-data.csv')
        self.df.set_index('Date', inplace=True)

    def test_no_nans(self):
        num_stocks = self.df.shape[1]
        loader = DataLoader(self.df)
        X: np.ndarray = loader.get_data()

        assert isinstance(X, np.ndarray)
        assert np.isnan(X).any()
        assert X.shape[0] == num_stocks


if __name__ == '__main__':
    unittest.main()
