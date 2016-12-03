import pandas as pd
#import matplotlib.pyplot as plt
from sklearn import svm
#import numpy as np
from sklearn.decomposition import PCA


train = pd.read_csv("train.csv")
X_test = pd.read_csv("test.csv")
y_train = train['label']
#print(y_train)
X_train = train.drop(labels = ['label'],axis = 1)
#print(train)

#train_data = np.array(X_train)
#train_labels = np.array(y_train)

pca = PCA(n_components = 50,whiten=True)
pca.fit(X_train)
X_train = pca.transform(X_train)


print ("Training SVC Classifier...")
model = svm.SVC()
model.fit(X_train,y_train)

#test_data = np.array()
X_test = pca.transform(X_test)

print ("Scoring SVC Classifier...")
score = model.predict(X_test);
with open('predict.csv', 'w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in score:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')
print ("Done:\n")
