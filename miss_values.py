import numpy as np
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor


class Miss_values:

    @staticmethod
    def get_nan(y, col: str):
        return y[y[col].isna()]

    @staticmethod
    def drop_nan(y, col: str):
        y_copy = y.copy()
        return y_copy.dropna()

    @staticmethod
    def fill_nan_with_median(y, col: str):
        y_copy = y.copy()
        not_nan = y_copy.dropna()
        median = np.median(not_nan[col])

        for i in range(0, len(y_copy)):
            if np.isnan(y_copy.at[i, col]):
                y_copy.at[i, col] = median
        return y_copy

    @staticmethod
    def fill_nan_with_last(y, col: str):
        y_copy = y.copy()
        not_nan = y_copy.dropna()
        last = np.nan
        for i in range(0, len(y_copy)):
            if not np.isnan(y_copy.at[i, col]):
                last = y_copy.at[i, col]

        if np.isnan(last):
            return y_copy

        for i in range(0, len(y_copy)):
            if np.isnan(y_copy.at[i, col]):
                y_copy.at[i, col] = last
            else:
                last = y_copy.at[i, col]
        return y_copy

    @staticmethod
    def fill_nan_with_nearest_mean(y, col: str):
        y_copy = y.copy()
        not_nan = y_copy.dropna()

        for i in range(0, len(y_copy)):
            if np.isnan(y_copy.at[i, col]):
                j = i
                left, right = None, None
                while j > 0 and np.isnan(y_copy.at[j, col]):
                    j -= 1
                if not np.isnan(y_copy.at[j, col]):
                    left = y_copy.at[j, col]
                j = i
                while j < len(y_copy) and np.isnan(y_copy.at[j, col]):
                    j += 1
                if not np.isnan(y_copy.at[j, col]):
                    right = y_copy.at[j, col]

                if left and right:
                    y_copy.at[i, col] = np.mean([left, right])
                elif left:
                    y_copy.at[i, col] = left
                elif right:
                    y_copy.at[i, col] = right
                else:
                    y_copy.at[i, col] = 0

        return y_copy

    @staticmethod
    def fill_nan_with_model_forecast(y: DataFrame):
        y_copy = y.copy()
        not_nan = y_copy.dropna()

        nan_values = np.isnan(y_copy[list(y_copy)[1]])
        nan_values = y_copy[nan_values]

        reg = RandomForestRegressor(n_estimators=1000)
        reg.fit(not_nan[list(not_nan)[0]].values.reshape(-1, 1), not_nan[list(not_nan)[1]].values)
        preds = reg.predict(nan_values[list(nan_values)[0]].values.reshape(-1, 1))

        col = list(y_copy)[1]
        for pred in preds:
            for i in range(0, len(y_copy[col])):
                if np.isnan(y_copy[col][i]):
                    y_copy[col][i] = pred
        return y_copy
