from sklearn.ensemble import RandomForestRegressor
import numpy as np


class Outliers_correction:

    @staticmethod
    def drop_outliers(data, outliers):
        data_copy = data.copy()
        return data_copy.drop(outliers.index)

    @staticmethod
    def median_correction(data, col: str, outliers):
        data_copy = data.copy()
        median = np.median(data[col])

        col_data = list(data_copy)[0]
        for outlier in outliers[col_data]:
            index = data_copy[col_data][data_copy[col_data] == outlier].index
            data_copy[col][index] = median
        return data_copy

    @staticmethod
    def correction_with_model(data, outliers, col, y):
        reg = RandomForestRegressor(n_estimators=1000, bootstrap=False)
        data_copy = data.copy()
        clean_data = Outliers_correction.drop_outliers(data, outliers)

        reg.fit(X=clean_data[y].values.reshape(-1, 1), y=clean_data[col])
        pred = reg.predict(outliers[y].values.reshape(-1, 1))

        data_copy[col][outliers.index] = pred
        return data_copy
