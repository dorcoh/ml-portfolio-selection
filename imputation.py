import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from copy import deepcopy

class imputation:
    def __init__(self, data: pd.DataFrame = pd.DataFrame(), data_path: str = None):
        if not data.empty:
            self.df = data
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise Exception("No input was passed, pass either DataFrame or path to data")

        self.imputed_df = deepcopy(self.df)
        num_rows = self.df.shape[0]
        self.locf_limit = round(0.1*num_rows)
        self.reg_limit = round(0.5*num_rows)

    def impute_all(self):
        self.impute(self.locf, self.locf_limit)
        self.impute(self.regression_imputation, self.reg_limit)
        self.imputed_df = self.imputed_df.dropna(axis=1)

    def impute(self, method, method_lim):
        nan_by_col = imputation.get_num_missing_values_by_col(self.imputed_df)
        impute_cols = []
        for col in self.imputed_df.columns:
            if col.lower() != 'date' and nan_by_col[col] > 0:
                if nan_by_col[col] < method_lim:
                    impute_cols.append(col)
        method(impute_cols)

    def mean_imputation(self, cols):
        fillna_dict = {}
        for col in cols:
            if self.imputed_df[col].isnull().all():
                # all nans, fill with 0
                fillna_dict[col] = 0
            else:
                # some nans
                fillna_dict[col] = self.imputed_df[col].mean()
        self.imputed_df = self.imputed_df.fillna(value=fillna_dict)

    def locf(self, cols):
        self.imputed_df.loc[:, cols] = self.imputed_df.loc[:, cols].ffill()
        print("Done last observation carried forward imputation for", cols)

    def regression_imputation(self, cols):
        full_cols = imputation.get_full_cols(self.imputed_df).columns
        for col in cols:
            reg_data = self.imputed_df[~self.imputed_df[col].isna()]
            X = reg_data[full_cols]
            y = reg_data[col]
            reg = LinearRegression().fit(X, y)
            for i, row in self.imputed_df.iterrows():
                if row[col] is None:
                    x = row[full_cols]
                    self.imputed_df[[i]][col] = reg.predict(x)
            print("Done regression imputation for", col)

    @staticmethod
    def get_num_full_cols(df):
        _df = df.dropna(axis=1)
        return len(_df.columns)

    @staticmethod
    def get_full_cols(df):
        _df = df.dropna(axis=1)
        return _df

    @staticmethod
    def get_num_full_rows(df):
        _df = df.dropna(axis=0)
        return _df.shape[0]

    @staticmethod
    def get_full_rows(df):
        _df = df.dropna(axis=0)
        return _df

    @staticmethod
    def get_num_missing_values_by_col(df):
        n_nan_by_col = {}
        for col in df.columns:
            if col.lower() == 'date' or col is None:
                continue
            n_nan_by_col[col] = df[col].isna().sum()
        return n_nan_by_col

    def save_imputed_df(self, save_path):
        self.imputed_df.to_csv(save_path)

    def get_imputed(self):
        return self.imputed_df

    def get_original(self):
        return self.df
