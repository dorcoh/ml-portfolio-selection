from solution import *
from main import get_args
from utils import get_train_data, daily_returns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

RETURNS = "Return"
STDDEV = "STDDEV"


def portfolio_transform(portfolio: dict):
    return np.array([val for val in portfolio.values()])


def compute_returns(test_data: np.array, portfolio):
    portfolio_returns = []
    x = portfolio_transform(portfolio)
    for investment_day in test_data.T:
        portfolio_returns.append(x @ investment_day)
    return np.mean(portfolio_returns)


def compute_variance(sigma, portfolio):
    x = portfolio_transform(portfolio)
    return x.T @ sigma @ x


def compute_sigma_likelihood(sigma, sigma_sam, test):
    M = sigma.shape[0]
    N = test.shape[1]
    return (-N/2)*(M*np.log(2*np.pi) + np.log(np.linalg.det(sigma)) + np.trace(np.linalg.inv(sigma) @ sigma_sam))


def gamma_testing(param_list, param_name, data_df=None, **kwargs):
    if data_df is None:
        args = get_args()
        data_df = get_train_data(date=args.date_to_evaluate, train_years=args.train_years)
    tickers = list(data_df.columns)
    imp = imputation(data=data_df)
    imp.impute_all()
    imp_train = imp.get_imputed()
    process_test = kwargs['process_test'] if 'process_test' in kwargs else False
    loader = DataLoader(imp_train, process_test=process_test, **DATALOADER_CONFIG)
    final_train, test = loader.split_data(desired_trainsize=1100, test_start=1101, test_size=160)

    portfolio_stats = {}
    for param in param_list:
        param_dict = {param_name: param}
        r, v = test_portfolio(imp_train, final_train, test, tickers, **param_dict)
        print(param, r, v)
        portfolio_stats[param] = {"Return": r, "STDDEV": np.sqrt(v)}

    return portfolio_stats


def test_portfolio(raw_train, processed_train, test, tickers, **kwargs):
    gamma = kwargs['target_return'] if 'target_return' in kwargs else 0.0
    k = kwargs['factors'] if 'factors' in kwargs else 35
    lamb = kwargs['lambda'] if 'lambda' in kwargs else 70
    pt = PortfolioTrainer(raw_train.to_numpy().T, processed_train, algorithm=AlgorithmType.STM)
    pt.fit(type='min-var', target_return=gamma, lamb=lamb, K=k, train_size=raw_train.shape[1])
    portfolio_array = pt.get_portfolio()

    portfolio = {}
    for name, val in zip(tickers, portfolio_array):
        portfolio[name] = val

    return compute_returns(test, portfolio), compute_variance(pt.get_sigma(), portfolio)


def lambda_testing(param_list, param_name, data_df=None, **kwargs):
    if data_df is None:
        args = get_args()
        data_df = get_train_data(date=args.date_to_evaluate, train_years=args.train_years)

    imp = imputation(data=data_df)
    imp.impute_all()
    imp_train = imp.get_imputed()
    process_test = kwargs['process_test'] if 'process_test' in kwargs else False
    loader = DataLoader(imp_train, process_test=process_test, **DATALOADER_CONFIG)
    final_train, test = loader.split_data(desired_trainsize=1100, test_start=1101, test_size=160)

    stm_stats = {}
    for param in param_list:
        param_dict = {param_name: param, "train_size": final_train.shape[1]}
        stm_ll = test_stm(imp_train, final_train, test, **param_dict)
        print(param, stm_ll)
        stm_stats[param] = stm_ll

    return stm_stats


def test_stm(raw_train, processed_train, test, **kwargs):
    pt = PortfolioTrainer(raw_train.to_numpy().T, processed_train, algorithm=AlgorithmType.STM)
    sigma_sam = compute_sigma_sample(dataset=processed_train)
    sigma_stm = pt._estimate_sigma(sigma_sam, **kwargs)

    return compute_sigma_likelihood(sigma_stm, sigma_sam, test)


def plot_returns(param_res, param_name):
    returns = [(param, param_res[param][RETURNS]) for param in param_res.keys()]
    params = [r[0] for r in returns]
    avg_ret = [r[1] for r in returns]
    sn.barplot(x=params, y=avg_ret, color='blue').set(title="Average of daily returns by: " + param_name)
    plt.show()


def plot_stddev(param_res, param_name):
    std = [(param, param_res[param][STDDEV]) for param in param_res.keys()]
    params = [s[0] for s in std]
    std_list = [s[1] for s in std]
    sn.barplot(x=params, y=std_list, color='blue').set(title="Standard deviation by: " + param_name)
    plt.show()


def plot_log_likelihood(param_res, param_name):
    ll = [(param, param_res[param]) for param in param_res.keys()]
    params = [l[0] for l in ll]
    ll_list = [l[1] for l in ll]
    sn.barplot(x=params, y=ll_list, color='blue').set(title="Log likelihood by: " + param_name)
    plt.show()


if __name__ == '__main__':
    args = get_args()
    data = get_train_data(date='2021-08-26', train_years=5)
    gamma_list = [-2 + i*0.25 for i in range(16)]
    lamb_list = [10 + 5*i for i in range(8)]

    lamb_res = lambda_testing(lamb_list, 'lamb', data, process_test=True)
    plot_log_likelihood(lamb_res, 'lamb')
