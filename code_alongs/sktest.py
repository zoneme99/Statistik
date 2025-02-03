from sklearn.linear_model import LinearRegression
import pandas as pd
df = pd.read_csv("/home/jonatanc/Documents/git_repos/Statistik/code_alongs/Advertising.csv", index_col=0)
X = df.drop('sales', axis=1)
Y = df['sales']

reg = LinearRegression().fit(X,Y)
print(reg.score(X,Y), reg.coef_, reg.intercept_)
