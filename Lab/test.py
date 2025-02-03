from regressionclass import LinearRegression as LR
import pandas as pd

df = pd.read_csv("/home/jonatanc/Documents/git_repos/Statistik/code_alongs/Advertising.csv", index_col=0)
reg = LR(df, "sales", 0.05)

print(reg.B)

for feature in reg.matrix.columns:
    if feature == "sales":
        continue
    print(f"{feature} : {reg.T_test(feature)}")

print(reg.Pearson_pairs)

