import numpy as np
import scipy.stats as stats
import pandas as pd

df = pd.read_csv("Lab/Small-diameter-flow.csv", index_col=0)

class LinearRegression:
    def __init__(self, matrix, yvariable):
        if isinstance(matrix, pd.DataFrame):
            (n, d) = matrix.shape
            self._n = n
            self._d = d
            self._Y = matrix[yvariable]
            self._X = matrix.drop(yvariable, axis=1)
            self._X.insert(0, "bias", np.ones(self._X.shape[0]))
            self._B = np.linalg.pinv(self._X.T @ self._X) @ self._X.T @ self._Y
        else:
            raise TypeError("Needs to be a Dataframe")

    @property
    def n(self):
        return self._n
    
    @property
    def d(self):
        return self._d
    
    @property
    def Y(self):
        return self._Y.to_numpy()
    
    @property
    def X(self):
        return self._X.to_numpy()
    
    @property
    def B(self):
        return self._B.to_numpy()
    
    @property
    def SSE(self):
        return np.sum(np.square(self.Y - self.X @ self.B))
    
    @property
    def var(self):
        return self.SSE()/(self.n-self.d-1)
    
    @property
    def std(self):
        return np.sqrt(self.var())
    
    @property
    def SSR(self):
        return (self.n*np.sum(self.B*(self.X.T @ self.Y)) - (np.square(np.sum(self.Y))))/self.n
    
    @property
    def SST(self):
        return (self.n*np.sum(np.square(self.Y)) - np.square(np.sum(self.Y)))/self.n
    
    @property
    def R2(self):
        return self.SSR / self.SST
    
    

test = LinearRegression(df,"Flow")

print(test.R2)
    
        