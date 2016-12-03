import pandas as pd
import matplotlib.pyplot as plt


test = pd.read_csv('test.csv',index_col=0)
train = pd.read_csv('train.csv',index_col=0)

test = test.drop(labels = ['Name','Ticket','Fare','Cabin'],axis = 1)
train = train.drop(labels = ['Name','Ticket','Fare','Cabin'],axis = 1)
