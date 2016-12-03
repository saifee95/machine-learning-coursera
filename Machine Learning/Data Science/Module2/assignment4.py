import pandas as pd


# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#
df = pd.read_csv('dataset.csv',index_col=0)
#print(df.columns)
df.columns = df.loc[1]
#print(df.columns)
# TODO: Get rid of any row that has at least 4 NANs in it
#
df = df.dropna(axis=0,thresh=4)
df = df.drop(labels=['RK'],axis=1)
df = df.drop_duplicates(subset = ['PLAYER'])
df = df[1:]
df = df.reset_index(drop=True)
#print(df.dtypes)

df['GP'] = pd.to_numeric(df['GP'],errors = 'coerce')
df['PCT'] = pd.to_numeric(df['PCT'],errors = 'coerce')

print(len(df['PCT'].unique()))

x = df.loc[15,'GP'] + df.loc[16,'GP']
print(x)
# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
#
# .. your code here ..


# TODO: Get rid of the 'RK' column
#
# .. your code here ..


# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
#
# .. your code here ..



# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric



# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.

