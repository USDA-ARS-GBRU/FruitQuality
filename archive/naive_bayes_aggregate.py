import pandas as pd 

df = pd.read_csv("naive_bayes_baseline.csv")

stats_df = df.groupby('Name').describe()

print(stats_df)

stats_df.to_csv("naive_bayes_aggregate.csv")