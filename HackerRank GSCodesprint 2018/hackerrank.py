# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 00:46:27 2018

@author: Aditya Sinha
"""
from sklearn import tree

import pandas as pd

cars = pandas.read_csv("train.csv")
#plt.hist(cars["popularity"])
#plt.show()
cars.drop('buying_price', 1)
test.drop(test.columns[1], 1)

test= pandas.read_csv("test.csv",index_col=False,header=None)
pop=cars["popularity"]
train = cars.drop('popularity', 1)

clf = tree.DecisionTreeClassifier()
clf.fit(train, pop)
pred=clf.predict(test).astype(int)
#print(pred)
d={}
prediction= pd.DataFrame(data=d)
prediction['']=pred
prediction.to_csv('prediction.csv',index=False,header=False)
#plt.hist(prediction[""])
#plt.show()
