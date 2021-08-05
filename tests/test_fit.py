import logging
import sys
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from app.algorithms import AlgorithmType
from solution import PortfolioTrainer
from utils import prepare_for_training


class FitTestCase(unittest.TestCase):
    """Test fitting methods on real data."""
    def setUp(self) -> None:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.X = np.load('resources/preprocessed-data.npy')
        self.train, self.test = prepare_for_training(self.X)

    def test_fit_stm(self):
        pt = PortfolioTrainer(algorithm=AlgorithmType.STM)
        pt.fit(self.train, type='min-var', lamb=1e-3, train_size=self.train.shape[1])
        portfolio = pt.get_portfolio()
        assert isinstance(portfolio, np.ndarray)
        assert len(portfolio) > 1
        assert_almost_equal(np.sum(portfolio), 1)

    def test_fit_utm(self):
        pt = PortfolioTrainer(algorithm=AlgorithmType.UTM)
        pt.fit(self.train, type='min-var', lamb=63, train_size=self.train.shape[1])

    def test_fit_urm(self):
        pt = PortfolioTrainer(algorithm=AlgorithmType.URM)
        pt.fit(self.train, type='min-var', K=15, train_size=self.train.shape[1])

    def test_fit_em(self):
        pt = PortfolioTrainer(algorithm=AlgorithmType.EM)
        pt.fit(self.train, type='min-var', K=15, train_size=self.train.shape[1])



if __name__ == '__main__':
    unittest.main()
