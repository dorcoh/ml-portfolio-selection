import argparse

from solution import *
from utils import get_train_data


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--date_to_evaluate', type=str, help='format: yyyy-mm-dd', default='2021-12-12')
    parser.add_argument('--train_years', type=int, help='format: yyyy-mm-dd', default=5)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    train_df = get_train_data(date=args.date_to_evaluate, train_years=args.train_years)
    portfolio = get_portfolio(train=train_df)
