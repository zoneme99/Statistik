import pandas as pd

df = pd.read_csv("Lab/Small-diameter-flow.csv", index_col=0)
#print(type(df))
print(df.shape)