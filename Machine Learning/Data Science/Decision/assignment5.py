import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import tree
from subprocess import call
#https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names


# 
# TODO: Load up the mushroom dataset into dataframe 'X'
# Verify you did it properly.
# Indices shouldn't be doubled.
# Header information is on the dataset's website at the UCI ML Repo
# Check NA Encoding
#
X = pd.read_csv('Datasets/agaricus-lepiota.data')
X['e.1'] = X['e.1'].replace('?',np.NaN)
#print(X['e.1'])
# INFO: An easy way to show which rows have nans in them
#print X[pd.isnull(X).any(axis=1)]


# 
# TODO: Go ahead and drop any row with a nan
#
X = X.dropna(axis = 0)
#print (X.shape)


#
# TODO: Copy the labels out of the dset into variable 'y' then Remove
# them from X. Encode the labels, using the .map() trick we showed
# you in Module 5 -- canadian:0, kama:1, and rosa:2
#
y = X['p']
y = y.map({'e':0,'p':1})
X = X.drop(labels = ['p'],axis=1)
#print(y)
#
# TODO: Encode the entire dataset using dummies
#
#print(X.columns[0])
X = pd.get_dummies(X,columns = X.columns)
#print(X.corr())

# 
# TODO: Split your data into test / train sets
# Your test size can be 30% with random_state 7
# Use variable names: X_train, X_test, y_train, y_test
#
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=7)



#
# TODO: Create an DT classifier. No need to specify any parameters
#
model = tree.DecisionTreeClassifier()

 
#
# TODO: train the classifier on the training data / labels:
# TODO: score the classifier on the testing data / labels:
#
model.fit(X_train,y_train)
score = model.score(X_test,y_test)
print ("High-Dimensionality Score: ", round((score*100), 3))
ans = model.feature_importances_
print(ans)
#
# TODO: Use the code on the courses SciKit-Learn page to output a .DOT file
# Then render the .DOT to .PNGs. Ensure you have graphviz installed.
# If not, `brew install graphviz`.
#
#tree.export_graphviz(model.tree_,out_file='tree.dot',feature_names=X.columns)
#call(['dot','-T','png','tree.dot','-o','tree.png'])

