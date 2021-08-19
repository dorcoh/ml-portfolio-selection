import datetime
import logging
from typing import Tuple, List

import pandas as pd
import yfinance as yf
import numpy as np


def get_tickers() -> List:
    wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    wiki_df = wiki_table[0]

    return wiki_df['Symbol'].to_list()


def get_start_end_date(date: str, train_years: int) -> Tuple[str, str]:
    date_formatted = datetime.datetime.strptime(date, '%Y-%m-%d')
    start_date = date_formatted - datetime.timedelta(days=train_years * 365)
    start_date_str = start_date.strftime('%Y-%m-%d')

    return start_date_str, date


def get_train_data(date: str, train_years: int) -> pd.DataFrame:
    start_date, end_date = get_start_end_date(date, train_years)
    tickers = get_tickers()

    return yf.download(tickers, start_date, end_date)['Adj Close']

# processing data utils


def process_data(df: pd.DataFrame, **kwargs):
    """Wrapper to handle all of the required transformations"""
    limit_stocks: int = kwargs['limit_stocks'] if 'limit_stocks' in kwargs else None
    ldf = pre_process(df, limit_stocks=limit_stocks)

    cdf = clip_log_returns(ldf)

    norm_window_size: int = kwargs['norm_window_size'] if 'norm_window_size' in kwargs else 50
    ndf = normalize_with_vol(cdf, norm_window_size)

    limit_dataset: bool = kwargs['limit_dataset'] if 'limit_dataset' in kwargs else None
    flip_dataset: bool = kwargs['flip_dataset'] if 'flip_dataset' in kwargs else None
    X = gen_dataset(ndf, limit=limit_dataset, flip=flip_dataset)

    return X


def prepare_for_training(X, **kwargs):
    """Prepares our data array X for training."""
    desired_train_size: int = kwargs['desired_trainsize'] if 'desired_trainsize' in kwargs else 100
    test_starting_point: int = kwargs['test_start'] if 'test_start' in kwargs else 110
    test_size: int = kwargs['test_size'] if 'test_size' in kwargs else 10
    train, test = split_train_test(X=X, N=desired_train_size, t=test_starting_point, test_size=test_size)

    return train, test


def pre_process(df: pd.DataFrame, limit_stocks=70):
    logging.info(f"pre_process")

    _df = handle_nans(df)

    if limit_stocks is not None:
        _df = _df.iloc[:, :limit_stocks].reset_index(drop=True)

    logging.info(f"post pre_process. df shape: {_df.shape}")
    return _df

def handle_nans(df: pd.DataFrame):
    nan_cols = df.columns[df.isna().any()].tolist()
    for col in nan_cols:
        if df[col].isnull().all():
            # all nans, fill with 0
            df[col] = 0
        else:
            # some nans
            df[col].fillna(df[col].mean())

    return df

def log_returns(df: pd.DataFrame):
    logging.info(f"log_returns")
    _df = pd.DataFrame()
    cols = [x for x in df.columns]
    for col in cols:
        _df[col] = np.log(df[col] / df[col].shift(1))

    logging.info(f"post log_returns. df shape: {_df.shape}")
    return _df.iloc[1:]


def clip_log_returns(df):
    logging.info(f"clip_log_returns")
    ub = np.quantile(df.values, .95)
    lb = np.quantile(df.values, .05)
    _df = df.apply(lambda x: np.clip(x, lb, ub))

    logging.info(f"post clip_log_returns. df shape: {_df.shape}")
    return _df


def normalize_with_vol(df, window_size=50):
    logging.info(f"normalize_with_vol")
    _sqr = lambda x: np.sqrt(np.sum([_x ** 2 for _x in x]) / window_size)
    vol = df.rolling(window_size).apply(lambda x: _sqr(x))

    _df = df / vol
    __df = _df.iloc[window_size:].reset_index(drop=True)
    logging.info(f"post normalize_with_vol. df shape: {__df.shape}")
    return __df


def gen_dataset(df, limit=None, flip=True):
    """Generate a dataset out of a processed pd.DataFrame"""
    logging.info("gen_dataset")
    X = df.values
    if limit is not None:
        X = X[:limit, :]
    _X = X.T
    if flip:
        _X = np.flip(_X, axis=1)

    return _X


def split_train_test(X, N, t, test_size=10):
    """ Splits into dataset and test, according to selected point in time t
    :param X: dataset
    :param N: desired dataset set size
    :param t: test starting point
    :param test_size: desired test set size
    :return: a tuple (dataset, test)
    """
    logging.info(f"split_train_test")
    logging.info(f"X shape: {X.shape}, N: {N}, t: {t}, test_size: {test_size}")

    test = X[:, t:t + test_size]
    train = X[:, t - N:t]

    return train, test


class DataLoader:
    """this class is responsible for producing our datasets.
    It should process the data once, but able to generate multiple splitting configurations"""
    def __init__(self, df: pd.DataFrame, **kwargs):
        self.X = process_data(df, **kwargs)

    def get_data(self):
        return self.X

    def split_data(self, **kwargs):
        train, test = prepare_for_training(self.X, **kwargs)
        return train, test

    @classmethod
    def from_ndarray(cls, X: np.ndarray):
        """Create an instance of this class from a processed self.X"""
        self = cls.__new__(cls)
        self.X = X
        return self


def log_likelihood(Sigma, Sigma_SAM, N) -> float:
    """ Compute log-likelihood out of estimated Sigma and and sampled Sigma.
    :param Sigma: Sigma estimated by algorithm (e.g., Sigma_UTM)
    :param Sigma_SAM: Sigma as estimated from our samples (MLE)
    :param N: number of samples.
    :return: log-likeligood
    """
    logging.info("log_likelihood")
    if (np.isnan(Sigma)).any():
        return np.nan
    M = Sigma.shape[0]
    a = M * np.log(2*np.pi)
    b = np.log(np.linalg.det(Sigma))
    c = np.trace(np.linalg.lstsq(Sigma, Sigma_SAM, rcond=None)[0])
    ret_value = -(N/2) * (a+b+c)
    return ret_value


def compute_sigma_sample(dataset: np.array):
    """ Compute sample covariance matrix
    :param dataset: train/test set
    :return: cov matrix
    """
    logging.info("compute_sigma_sample")
    M = dataset.shape[0]
    N = dataset.shape[1]
    Sigma_SAM = np.zeros((M, M))
    for n in range(N):
        Sigma_SAM += np.outer(dataset[:, n], dataset[:, n])
    Sigma_SAM /= N

    return Sigma_SAM
