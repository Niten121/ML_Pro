# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:23:24 2022

@author: sethy
"""

import  pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

df = pd.read_csv("E:\ml pro\iris_data.csv")

print("head",df.head())
print("shape",df.shape) 
print(df.info())
print(df.columns)

le  =LabelEncoder()
label = le.fit_transform(df.Species)
df['outcome'] = label
print(df['outcome'].head())

x= df.drop(['Species','Id','outcome'],axis=1)
y = df.outcome
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)
print(x_train.shape)
model = SVC()
print(model)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(y_pred)
model.score(x_test, y_test)
model_g=SVC(gamma=10)
model_g.fit(x_train,y_train)
model_g.score(x_test, y_test)
print(model_g.score(x_test, y_test))

model_linear_kernel=SVC(kernel='linear')
model_linear_kernel.fit(x_train,y_train)
print(model_linear_kernel)

model_linear_kernel.score(x_test,y_test)
print(model.predict([[6.3,3.3,6.0,2.5]]))

print(model.predict([[64.8,3.0,1.5,0.3]]))


