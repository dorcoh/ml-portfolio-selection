import numpy as np

from utils import get_train_data, DataLoader


def gen_data():
    # to avoid downloading it each time
    train_df = get_train_data(date='2021-07-01', train_years=1)

    loader = DataLoader(train_df)
    X: np.ndarray = loader.get_data()

    np.save('resources/preprocessed-data.npy', X)


if __name__ == '__main__':
    gen_data()