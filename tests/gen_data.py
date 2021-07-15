from utils import get_train_data


def gen_data():
    # to avoid downloading it each time
    train_df = get_train_data(date='2021-07-01', train_years=1)
    train_df.to_csv('resources/data.csv', index=False)


if __name__ == '__main__':
    gen_data()