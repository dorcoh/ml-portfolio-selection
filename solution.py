
import numpy as np
import pandas as pd

from app.algorithms import compute_urm, compute_utm, compute_em, compute_stm, AlgorithmType
from utils import compute_sigma_sample, DataLoader


def get_portfolio(train: pd.DataFrame):
    loader = DataLoader(train)
    X = loader.get_data()
    train, test = loader.split_data()

    # TODO: wrap with a class for auto-tuning of the best portfolio
    pt = PortfolioTrainer(algorithm=AlgorithmType.STM)
    pt.fit(train)
    portfolio = pt.get_portfolio()

    return np.ones(train.shape[1]) / train.shape[1]


class PortfolioTrainer:
    """This class goal is generate a portfolio according to desired factor-model algorithm."""
    def __init__(self, algorithm: AlgorithmType = AlgorithmType.STM):
        self.algorithm = algorithm
        self.portfolio = None

    def fit(self, train: pd.DataFrame, **kwargs):
        """Produce a portfolio, should provide required hyper-parameters."""
        Sigma_SAM = compute_sigma_sample(dataset=train)
        Sigma = self._estimate_sigma(Sigma_SAM, **kwargs)
        self.portfolio = self._fit_portfolio(Sigma, **kwargs)

    def _estimate_sigma(self, Sigma_SAM, **kwargs) -> np.array:
        """Call the relevant algorithm according to `self.algorithm` and return estimated Sigma."""
        pass

    def _fit_portfolio(self, Sigma, **kwargs) -> np.array:
        """Call the relevant fitting method (e.g., min-var portfolio, tangent min-var, etc.) and return portfolio."""
        pass

    def get_portfolio(self):
        return self.portfolio


class PortfolioEvaluator:
    """Provides the logic for evaluating a trained portfolio."""
    pass


class PortfolioSelector:
    """Provides the logic for training, evaluating, and selecting the best portfolio."""
    pass


