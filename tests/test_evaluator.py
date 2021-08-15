import logging
import pickle
import sys
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from solution import PortfolioEvaluator
from utils import DataLoader


class EvaluatorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.X = np.load('resources/preprocessed-data.npy')
        self.data_loader = DataLoader.from_ndarray(self.X)
        self.train, self.test = self.data_loader.split_data()

        with open('resources/evaluator.pickle', 'rb') as handle:
            self.evaluator: PortfolioEvaluator = pickle.load(handle)

    def test_compute_portfolio(self):
        evaluator = PortfolioEvaluator()
        fit_kwargs = {
            'method': 'min-var',
            'target_return': 0.8,
            'lamb': 1e-3,
        }
        pt = evaluator.compute_portfolio(self.train, **fit_kwargs)
        portfolio = pt.get_portfolio()

        assert isinstance(portfolio, np.ndarray)
        assert len(portfolio) > 1
        assert_almost_equal(np.sum(portfolio), 1, decimal=3)

    def test_compute_multiple_portfolios(self):
        evaluator = PortfolioEvaluator()
        auto_fit_kwargs = {
            'hyper-params': {
                'target_return': (-1.5, 1.5),
                'lamb': (1e-3 - (1e-10), 1e-3 + (1e-10))
            },
            'trials': 100
        }

        evaluator.compute_multiple_portfolios(self.data_loader, **auto_fit_kwargs)
        for portfolio_trainer_instance in evaluator.get_results():
            portfolio = portfolio_trainer_instance.get_portfolio()
            assert isinstance(portfolio, np.ndarray)
            assert len(portfolio) > 1
            assert_almost_equal(np.sum(portfolio), 1, decimal=3)

        # for testing plot function
        # with open('resources/evaluator.pickle', 'wb') as handle:
          #  pickle.dump(evaluator, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def test_plot(self):
        self.evaluator.plot()


if __name__ == '__main__':
    unittest.main()
