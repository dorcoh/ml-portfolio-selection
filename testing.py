from solution import *
from main import get_args
from utils import get_train_data
import numpy as np


def portfolio_transform(portfolio):
    return np.array([val for val in portfolio.values()])


def compute_returns(test_data, portfolio):
    daily_returns = []
    x = portfolio_transform(portfolio)
    for investment_day in test_data:
        daily_returns.append(x @ investment_day)
    return daily_returns


def compute_variance(sigma, portfolio):
    x = portfolio_transform(portfolio)
    return x.T @ sigma @ x


def param_testing(param_list, param_name):
    args = get_args()
    data_df = get_train_data(date=args.date_to_evaluate, train_years=args.train_years)
    tickers = list(data_df.columns)
    imp = imputation(data=data_df)
    imp.impute_all()
    imp_train = imp.get_imputed()
    loader = DataLoader(imp_train, **DATALOADER_CONFIG)
    final_train, test = loader.split_data(desired_trainsize=1100, test_start=1101, test_size=160)

    portfolio_stats = {}
    for param in param_list:
        param_dict = {param_name: param}
        r, v = test_portfolio(imp_train, final_train, test, tickers, **param_dict)
        portfolio_stats[param] = {"Return": r, "STDDEV": np.sqrt(v)}

    return portfolio_stats


def test_portfolio(raw_train, processed_train, test, tickers, **kwargs):
    gamma = kwargs['target_return'] if kwargs['target_return'] else 1
    k = kwargs['factors'] if kwargs['factors'] is not None else 35
    lamb = kwargs['lambda'] if kwargs['lambda'] is not None else 70
    pt = PortfolioTrainer(raw_train.to_numpy().T, processed_train, algorithm=AlgorithmType.STM)
    pt.fit(type='min-var', target_return=gamma, lamb=lamb, K=k, train_size=raw_train.shape[1])
    portfolio_array = pt.get_portfolio()

    portfolio = {}
    for name, val in zip(tickers, portfolio_array):
        portfolio[name] = val

    return compute_returns(test, portfolio), compute_variance(pt.get_sigma(), portfolio)

