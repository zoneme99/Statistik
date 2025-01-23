import pandas as pd
import numpy as np

df = pd.read_csv("Lab/Small-diameter-flow.csv", index_col=0)
#print(type(df))
X = df.drop('Flow', axis=1)
Y = df['Flow']


X.insert(0, "1", np.ones(X.shape[0]))
b = np.linalg.pinv(X.T @ X) @ X.T @ Y
print(b)