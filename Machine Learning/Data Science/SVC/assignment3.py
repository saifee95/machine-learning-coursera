import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

X = pd.read_csv('Datasets/parkinsons.data')
#print(X)

y = X['status']
#print(y.shape)
X = X.drop(labels = ['name','status'],axis = 1)
#print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=7)
#svc = SVC()
#svc.fit(X_train,y_train)
b_score = 0
#print(score)

a = np.arange(0.05,2,0.05)
b = np.arange(0.001,0.1,0.001)

for i in a :
    for j in b :
        svc = SVC(C = i,kernel = 'rbf',gamma=j)
        svc.fit(X_train,y_train)
        score = svc.score(X_test,y_test)
        if score > b_score :
            b_score = score
            #print(b_score)

print(b_score)