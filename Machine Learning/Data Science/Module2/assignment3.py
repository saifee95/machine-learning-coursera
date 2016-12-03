import pandas as pd

# TODO: Load up the dataset
# Ensuring you set the appropriate header column names
#
na = ['motor','screw','pgain','vgain','class']
df = pd.read_csv('Datasets/servo.data',header=None,names=na)
#print(df[0:5])



# TODO: Create a slice that contains all entries
# having a vgain equal to 5. Then print the 
# length of (# of samples in) that slice:
#
x = df[df['vgain'] == 5]
#print(len(x))


# TODO: Create a slice that contains all entries
# having a motor equal to E and screw equal
# to E. Then print the length of (# of
# samples in) that slice:
#
y = df[(df['motor'] == 'E') & (df['screw'] == 'E')]
#print(y)
#print(len(y))



# TODO: Create a slice that contains all entries
# having a pgain equal to 4. Use one of the
# various methods of finding the mean vgain
# value for the samples in that slice. Once
# you've found it, print it:
#
z = df[(df['pgain'] == 4)]
print(z.describe())



# TODO: (Bonus) See what happens when you run
# the .dtypes method on your dataframe!



