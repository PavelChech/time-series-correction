import pandas as pd
from pandas import DataFrame
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from graphics import Graphics
from miss_values import Miss_values
from outlier_definition import Outlier_definition
from outliers_correction import Outliers_correction
from statistics import mean


class Analyze:

    def __init__(self):
        self.data = DataFrame()
        self.median = 0
        self.min = 0
        self.max = 0
        self.auto_corr_coef = 0

    def setData(self, data: DataFrame):
        if len(data.columns) > 2:
            raise ValueError('Error! Data has more than 2 columns')
        elif data.empty:
            raise ValueError('Error! Data is empty')
        # Проверить тип дата-времени
        self.data = data

    @staticmethod
    def errors(y_pred, y_test):
        return [mean_absolute_percentage_error(y_test, y_pred), mean_absolute_error(y_test, y_pred),
                max_error(y_test, y_pred)]

    def compute_stats(self):
        value_col = list(self.data)[1]
        self.median = self.data[value_col].median()
        self.max = self.data[value_col].max()
        self.min = self.data[value_col].min()
        self.auto_corr_coef = Analyze.auto_corr(self.data)

    @staticmethod
    def compute_statistics(data):
        value_col = list(data)[1]
        stats = {"median": data[value_col].median(), "max": data[value_col].max(), "min": data[value_col].min(),
                 "auto_corr": Analyze.auto_corr(data)}
        return stats

    @staticmethod
    def auto_corr(data, lag=1):
        value_col = list(data)[1]
        return data[value_col].autocorr(lag=1)

    def data_with_model_forecast(self):
        data = self.data.copy()
        nan = data[data['Value'].isna()]
        data = data.dropna()
        outliers = Outlier_definition.iqr(data, 'Value')
        data = Outliers_correction.drop_outliers(data, outliers)
        X = data['DATE']
        y = data['Value']
        reg = RandomForestRegressor(n_estimators=1000, bootstrap=False).fit(X.values.reshape(-1, 1), y)
        pred_outliers = reg.predict(outliers['DATE'].values.reshape(-1, 1))
        pred_nans = reg.predict(nan['DATE'].values.reshape(-1, 1))
        outliers['Value'] = pred_outliers
        nan['Value'] = pred_nans
        data = data.append(nan).append(outliers)
        return data.sort_index()

    @staticmethod
    def compare_datasets(data, another_data):
        data_stats = Analyze.compute_statistics(data)
        another_data_stats = Analyze.compute_statistics(another_data)
        print(f"median: {data_stats['median']}\t{another_data_stats['median']}\n"
              f"min max: [{data_stats['min']}, {data_stats['max']}]\t[{another_data_stats['min']}, "
              f"{another_data_stats['max']}]\n"
              f"auto correlation coefficient: {data_stats['auto_corr']}\t{another_data_stats['auto_corr']}\n")
        if data_stats["auto_corr"] < another_data_stats["auto_corr"]:
            return 1
        elif data_stats["auto_corr"] == data_stats["auto_corr"]:
            return 0
        else:
            return -1

    def multiple_data_correction(self):
        date_col = list(self.data)[0]
        val_col = list(self.data)[1]

        data = self.data.copy()
        original_data = self.data_with_model_forecast()

        data_copy = self.data.copy()
        datasets_without_nan = [
            {'nan': 'fill_nan_with_median', 'data': Miss_values.fill_nan_with_median(data_copy, val_col)},
            {'nan': 'fill_nan_with_last', 'data': Miss_values.fill_nan_with_last(data_copy, val_col)},
            {'nan': 'fill_nan_with_model_forecast', 'data': Miss_values.fill_nan_with_model_forecast(data_copy), },
            {'nan': 'fill_nan_with_nearest_mean', 'data': Miss_values.fill_nan_with_nearest_mean(data_copy, val_col)}]

        datasets_without_nan_outliers = []
        for dataset_type in datasets_without_nan:
            dataset = dataset_type['data'].copy()
            outliers = Outlier_definition.iqr(dataset, val_col)
            datasets_without_nan_outliers.append(
                {'nan': dataset_type['nan'], 'outliers': 'iqr', 'outliers_clean': 'median_smooth',
                 'data': Outliers_correction.median_correction(dataset, val_col, outliers)})
            datasets_without_nan_outliers.append(
                {'nan': dataset_type['nan'], 'outliers': 'iqr', 'outliers_clean': 'smooth_with_model',
                 'data': Outliers_correction.correction_with_model(dataset, outliers, val_col, date_col)})

            outliers = Outlier_definition.sliding_window_definition(dataset, val_col, round(len(dataset) * 0.4),
                                                                    round(len(dataset) * 0.1))
            datasets_without_nan_outliers.append(
                {'nan': dataset_type['nan'], 'outliers': 'sliding_window_definition 0.4 0.1',
                 'outliers_clean': 'median_smooth',
                 'data': Outliers_correction.median_correction(dataset, val_col, outliers)})
            datasets_without_nan_outliers.append(
                {'nan': dataset_type['nan'], 'outliers': 'sliding_window_definition 0.4 0.1',
                 'outliers_clean': 'smooth_with_model',
                 'data': Outliers_correction.correction_with_model(dataset, outliers, val_col, date_col)})

            outliers = Outlier_definition.sliding_window_definition(dataset, val_col, round(len(dataset) * 0.05),
                                                                    round(len(dataset) * 0.05))
            datasets_without_nan_outliers.append(
                {'nan': dataset_type['nan'], 'outliers': 'sliding_window_definition 0.05 0.05',
                 'outliers_clean': 'median_smooth',
                 'data': Outliers_correction.median_correction(dataset, val_col, outliers)})
            datasets_without_nan_outliers.append(
                {'nan': dataset_type['nan'], 'outliers': 'sliding_window_definition 0.05 0.05',
                 'outliers_clean': 'smooth_with_model',
                 'data': Outliers_correction.correction_with_model(dataset, outliers, val_col, date_col)})

        # for dataset in datasets_without_nan_outliers:
        #     y_pred = dataset['data'].loc[original_data[val_col].index]['Value']
        #     print(f"nan:{dataset['nan']}\n out:{dataset['outliers']}\n out_clean:{dataset['outliers_clean']}")
        #     print(Analyze.errors(y_pred, original_data[val_col]))
        #     print('==============================')

        min_dataset = {}
        for dataset in datasets_without_nan_outliers:
            y_pred = dataset['data'].loc[original_data[val_col].index]['Value']
            title = f"nan:{dataset['nan']}, out:{dataset['outliers']}, out_clean:{dataset['outliers_clean']}"
            if len(min_dataset) == 0:
                min_dataset = dataset
            else:
                y_pred_min = min_dataset['data'].loc[original_data[val_col].index]['Value']
                if Analyze.errors(y_pred, original_data[val_col])[0] < Analyze.errors(y_pred_min,
                                                                                      original_data[val_col])[0]:
                    min_dataset = dataset

        # for dataset in datasets_without_nan_outliers:
        #     if len(min_dataset) == 0:
        #         min_dataset = dataset
        #     else:
        #         if Analyze.compare_datasets(dataset['data'], min_dataset['data']) == 0:
        #             min_dataset = dataset

            # if (Analyze.errors(y_pred, original_data[val_col])[0]) < 0.2:
            #     print(Analyze.errors(y_pred, original_data[val_col]))
            #     Graphics.linear_graph(title, dataset['data'])
        Graphics.linear_graph("Original data", data)
        Graphics.linear_graph("Correct data", min_dataset['data'])

        return min_dataset['data']
