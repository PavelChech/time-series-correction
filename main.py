import numpy as np
import pandas as pd

import outlier_definition
from tests import Testing
from analyze import Analyze
from graphics import Graphics
from miss_values import Miss_values
from outlier_definition import Outlier_definition
from outliers_correction import Outliers_correction

def data_preprocessing(data):
    data[list(data)[0]] = pd.to_datetime(data[list(data)[0]], format='%d-%m-%Y')
    return data


def main():
    data = pd.read_csv("C:\\Users\\Fork\\Desktop\\Electric_Production.csv")
    data = data[['DATE', 'Value']]
    data = data_preprocessing(data)
    Graphics.linear_graph("True data", data)
    data_copy = data.copy()
    data_copy = Testing.break_data(data_copy)

    data_to_analyze = Analyze()
    data_to_analyze.setData(data_copy)
    data_to_analyze.compute_stats()

    correct_df = data_to_analyze.multiple_data_correction()


if __name__ == "__main__":
    main()
