import pandas as pd

df = pd.read_csv('Datasets/tutorial.csv')
print(df)
print(df.describe())

print(df.loc[2:4,'col3'])

