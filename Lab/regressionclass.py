import numpy as np
import scipy.stats as stats
import pandas as pd

df = pd.read_csv("Lab/Small-diameter-flow.csv", index_col=0)

class LinearRegression:
    def __init__(self, matrix):
        if isinstance(matrix, pd.DataFrame):
            (n, d) = matrix.shape
            self._n = n
            self._d = d
        else:
            raise TypeError("Needs to be a Dataframe")

    @property
    def n(self):
        return self._n
    
    @property
    def d(self):
        return self._d
    

test = LinearRegression(df)

print(test.n,test.d)
    
        