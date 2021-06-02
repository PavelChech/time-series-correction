import random
import numpy as np
import pandas as pd


class Testing:

    @staticmethod
    def break_data(data):
        data_copy = data.copy()
        data_copy.dropna()
        for i in range(1, len(data_copy)-1):
            if random.random() > 0.8:
                data_copy[list(data_copy)[1]][i] = np.nan

        data_copy[list(data_copy)[1]][25] = 100000
        data_copy[list(data_copy)[1]][35] = -100000
        data_copy[list(data_copy)[1]][45] = 50000
        data_copy[list(data_copy)[1]][55] = -50000
        return data_copy
