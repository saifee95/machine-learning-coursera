import pandas as pd
import numpy as np


# TODO:
# Load up the dataset, setting correct header labels
# Use basic pandas commands to look through the dataset...
# get a feel for it before proceeding!
# Find out what value the dataset creators used to
# represent "nan" and ensure it's properly encoded as np.nan
#
x = ['education', 'age', 'capital-gain', 'race', 'capital-loss', 'hours-per-week', 'sex', 'classification']
df = pd.read_csv('Datasets/census.data',header = None,names = x,index_col=0)
df.fillna(np.nan)
#print(df[0:90])


# TODO:
# Figure out which features should be continuous + numeric
# Conert these to the appropriate data type as needed,
# that is, float64 or int64
df['age'] = pd.to_numeric(df['age'])
df['capital-gain'] = pd.to_numeric(df['capital-gain'])
df['capital-loss'] = pd.to_numeric(df['capital-loss'])
df['hours-per-week'] = pd.to_numeric(df['hours-per-week'])

#print(df.dtypes)

# TODO:
# Look through your data and identify any potential categorical
# features. Ensure you properly encode any ordinal types using
# the method discussed in the chapter.
#
#print(df[''].unique())
#order = ['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th','HS-grad','Some-college','Bachelors','Masters','Doctorate']
cl = ['<=50K','>50K']
#df['education'] = df['education'].astype("category",ordered=True,categories=order).cat.codes
df['classification'] = df['classification'].astype("category",ordered=True,categories=cl).cat.codes
#print(df)
# TODO:
# Look through your data and identify any potential categorical
# features. Ensure you properly encode any nominal types by
# exploding them out to new, separate, boolean fatures.
#
df = pd.get_dummies(df,columns=['race'])
df = pd.get_dummies(df,columns=['sex'])
df = pd.get_dummies(df,columns=['education'])
print(df)

# TODO:
# Print out your dataframe
print(df.columns)