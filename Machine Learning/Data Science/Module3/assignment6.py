import pandas as pd
import matplotlib.pyplot as plt

#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
df = pd.read_csv('Datasets/wheat.data')



#
# TODO: Drop the 'id', 'area', and 'perimeter' feature
# 
df = df.drop(labels=['id'],axis=1)
x = df.corr()
print(x)

#
# TODO: Graph the correlation matrix using imshow or matshow
# 
plt.imshow(x,interpolation='nearest')
plt.colorbar()

plt.show()


