import numpy as np
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor

class Outlier_definition:

    @staticmethod
    def iqr(data, col: str):
        data_copy = data.copy()
        P = np.percentile(data_copy[col], [5, 95])
        outliers = data_copy[(data_copy[col] < P[0]) | (data_copy[col] > P[1])]
        return outliers


    @staticmethod
    def sliding_window_definition(data, col: str, window_size: int, step: int):
        outliers = []
        data_copy = data.copy()
        for index in range(0, len(data_copy[col]) - step, step):
            sub_array = data_copy[col].loc[index: index + window_size - 1]
            std = sub_array.std()
            mean = sub_array.mean()
            sub_index = 0
            for value in sub_array:
                if value < mean - std or value > mean + std:
                    if index + sub_index not in outliers:
                        outliers.append(index + sub_index)
                sub_index += 1
        return data_copy.iloc[outliers]

