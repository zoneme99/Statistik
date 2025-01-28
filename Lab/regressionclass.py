import numpy as np
import scipy.stats as stats
import pandas as pd

class LinearRegression:
    def __init__(self, matrix, yvariable, confedence_level = 0.95):
        if isinstance(matrix, pd.DataFrame):
            (n, d) = matrix.shape
            self._n = n
            self._d = d
            self._conf_level = confedence_level
            self._matrix = matrix
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
    def conf_level(self):
        return self._conf_level
    
    @property
    def matrix(self):
        return self._matrix
    
    @property
    def Y(self):
        return self._Y
    
    @property
    def X(self):
        return self._X
    
    @property
    def B(self):
        return self._B
    
    @property
    def SSE(self):
        return np.sum(np.square(self.Y.to_numpy() - self.X.to_numpy() @ self.B.to_numpy()))
    
    @property
    def var(self):
        return self.SSE/(self.n-self.d-1)
    
    @property
    def std(self):
        return np.sqrt(self.var)
    
    @property
    def SSR(self):
        return (self.n*np.sum(self.B.to_numpy()*(self.X.to_numpy().T @ self.Y.to_numpy())) - (np.square(np.sum(self.Y.to_numpy()))))/self.n
    
    @property
    def SST(self):
        return (self.n*np.sum(np.square(self.Y.to_numpy())) - np.square(np.sum(self.Y.to_numpy())))/self.n
    
    @property
    def R2(self):
        return self.SSR / self.SST
    
    @property
    def F_test(self):
        f = stats.f(self.d, self.n-self.d-1)
        f_stat = (self.SSR/self.d)/self.var
        return f.sf(f_stat)
    

    @property
    def covar_matrix(self):
        return (np.linalg.pinv(self.X.to_numpy()[:,1:].T @ self.X.to_numpy()[:,1:]))*self.var
    
    #Double sided T-test
    def T_test(self, feature):
        cindex = self.X.columns.get_loc(feature)
        Bhat = self.B.to_numpy()[cindex]
        C = self.covar_matrix[cindex-1,cindex-1] #cindex not reset to 0
        T_stat = Bhat/(self.std*np.sqrt(C))
        t_object = stats.t(self.n-self.d-1)
        p_value = 2*min(t_object.cdf(T_stat), t_object.sf(T_stat))
        return p_value
    
    #Compare all features with eachother
    @property
    def Pearson_pairs(self):
        features = self._X.drop("bias", axis=1).columns
        for x in range(len(features)):
            for y in range(x,len(features)):
                if features[x] == features[y]:
                    continue
                print(f"{features[x]}/{features[y]} : {stats.pearsonr(self._X[features[x]], self._X[features[y]])}")      


if  __name__ == "__main__":
    df = pd.read_csv("Lab/Small-diameter-flow.csv")
    test = LinearRegression(df,"Flow")
    #print(test.T_test("Observer"))
    print(test.conf_level)