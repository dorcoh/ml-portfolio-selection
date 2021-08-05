from copy import copy

import numpy as np
import pandas as pd

from app.algorithms import compute_urm, compute_utm, compute_em, compute_stm, AlgorithmType
from utils import compute_sigma_sample, DataLoader


def get_portfolio(train: pd.DataFrame):
    loader = DataLoader(train)
    train, test = loader.split_data()

    # TODO: wrap with a class for auto-tuning of the best portfolio
    pt = PortfolioTrainer(algorithm=AlgorithmType.STM)
    pt.fit(train, type='min-var', lamb=70, K=35, train_size=train.shape[1])
    portfolio = pt.get_portfolio()

    return np.ones(train.shape[1]) / train.shape[1]


class PortfolioTrainer:
    """This class goal is generate a portfolio according to desired factor-model algorithm."""
    def __init__(self, algorithm: AlgorithmType = AlgorithmType.STM):
        self.algorithm = algorithm
        self.portfolio = None
        self.returns = None
        self.Sigma = None

    def fit(self, train: np.ndarray, **kwargs):
        """Produce a portfolio, should provide required hyper-parameters."""
        Sigma_SAM = compute_sigma_sample(dataset=train)
        self.Sigma = self._estimate_sigma(Sigma_SAM, **kwargs)
        self.returns = self._estimate_returns(train)
        self._fit_portfolio(self.Sigma, self.returns, **kwargs)

    def _estimate_sigma(self, Sigma_SAM, **kwargs) -> np.array:
        """Call the relevant algorithm according to `self.algorithm` and return estimated Sigma."""
        Sigma = None
        if self.algorithm == AlgorithmType.STM:
            Sigma = compute_stm(Sigma_SAM, kwargs['lamb'], kwargs['train_size'])
        elif self.algorithm == AlgorithmType.UTM:
            Sigma = compute_utm(Sigma_SAM, kwargs['lamb'], kwargs['train_size'])
        elif self.algorithm == AlgorithmType.URM:
            Sigma = compute_urm(Sigma_SAM, kwargs['K'])
        elif self.algorithm == AlgorithmType.EM:
            Sigma = compute_em(Sigma_SAM, kwargs['K'])

        return Sigma

    def _estimate_returns(self, train : np.ndarray, **kwargs):
        """Estimates the expected returns of the stocks from the training data"""
        method = kwargs['method'] if 'method' in kwargs.keys() else 'Naive'
        returns = []
        if method == 'Naive':
            for stock in train:
                estimates = list(stock / np.roll(stock, 1))[1:]
                returns.append(np.average(estimates))
        elif method == "other":
            pass
        return np.array(returns)

    def _fit_portfolio(self, Sigma, returns, **kwargs) -> np.array:
        """Call the relevant fitting method (e.g., min-var portfolio, tangent min-var, etc.) and return portfolio."""
        # TODO: Check if shorting is allowed or not, then update accordingly

        method = kwargs['type']
        if method == 'min-var':
            gamma = None
            if 'target_return' in kwargs.keys():
                gamma = kwargs['target_return']
            if gamma:
                inv_sigma = np.linalg.inv(Sigma)
                e = np.ones(Sigma.shape[0])
                self.portfolio = (1 - gamma) * (returns.T @ inv_sigma @ returns) / (e.T @ inv_sigma @ returns)
                self.portfolio += gamma * (returns.T @ inv_sigma @ e) / (e.T @ inv_sigma @ e)
            else:
                e = np.ones(Sigma.shape[0])
                inv_sigma = np.linalg.inv(Sigma)
                self.portfolio = (inv_sigma @ e)/(e.T @ inv_sigma @ e)

    def get_portfolio(self):
        return self.portfolio

    def get_sigma(self):
        return self.Sigma

    def get_estimated_returns(self):
        return self.returns


class PortfolioEvaluator:
    """Provides the logic for training and evaluating portfolios."""

    def __init__(self):
        self.results = []

    def compute_returns_std(self, portfolio: np.ndarray, sigma: np.ndarray, estimated_returns: np.ndarray):
        """
        Compute (std, return) for a given portfolio
        :param sigma:
        :param portfolio:
        :param estimated_returns:
        :return:
        """
        ret = estimated_returns @ portfolio
        std = np.sqrt(portfolio.T @ sigma @ portfolio)

        return ret, std

    def compute_portfolio(self, train: np.ndarray, **kwargs):
        """
        Compute the portfolio for given hyper-params.
        :param train: a train set.
        :param gamma: chosen gamma.
        :param epsilon:
        :return: a PortfolioTrainer instance.
        """
        train_size = train.shape[1]
        pt = PortfolioTrainer(algorithm=AlgorithmType.STM)

        kwargs.update({'train_size': train_size})
        pt.fit(train, type='min-var', **kwargs)

        return pt

    def get_portfolio_returns_std(self, pt: PortfolioTrainer):
        """
        Given a PortfolioTrainer instance, return its (return,std) tuple.
        :param pt: a portfolio trainer instance
        :return: a tuple (return,std)
        """
        portfolio = pt.get_portfolio()
        sigma = pt.get_sigma()
        returns = pt.get_estimated_returns()
        return self.compute_returns_std(portfolio=portfolio, sigma=sigma, estimated_returns=returns)

    def compute_multiple_portfolios(self, data_loader: DataLoader, **kwargs):
        """ Provide ranges for each hyper-parameter (gamma), sample uniformly and call `compute_portfolio` for each set of
        hyper-parameters. Store the results in self.results
        :param data_loader: an instance of DataLoader class.
        :param kwargs: required key-word arguments.
        :return: None, this function *APPENDS* portfolio_trainer instances to `self.results`
        """
        train, _ = data_loader.split_data()
        train_size = train.shape[1]

        for trial in range(kwargs.get('trials', 1)):
            hyper_params_dict = self._gen_hyperparams(**kwargs['hyper-params'])
            pt = self.compute_portfolio(train, **hyper_params_dict)
            self.results.append(pt)

        pass

    def _gen_hyperparams(self, **kwargs):
        """ Generate a single configuration of hyper-parameters.
        :param kwargs: a dictionary where key indicate on hyper-param, and values are float ranges we can sample from
        :return: a dictionary with a single config of hyper-params.
        """
        ret = copy(kwargs)
        for key, value in kwargs.items():
            assert len(value) == 2 and isinstance(value[0], float) and isinstance(value[1], float), \
                "Value must be a range of floats."
            ret[key] = np.random.uniform(value[0], value[1])

        return ret

    def plot(self):
        """ Plot the evaluation graph, extract actual values from `self.results`.
        :return: None, plots a graph.
        """
        import matplotlib.pyplot as plt

        assert len(self.results), "Empty results."

        x, y = [], []
        for pt in self.results:
            ret, std = self.get_portfolio_returns_std(pt)
            x.append(std)
            y.append(ret)

        plt.plot(x, y, 'o', color='green')
        plt.title('Expected return vs. STD')
        plt.xlabel('Standard deviation')
        plt.ylabel('Expected return')
        plt.show()
        plt.close()

    def get_results(self):
        return self.results