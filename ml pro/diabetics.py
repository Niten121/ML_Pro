# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:50:56 2022

@author: sethy
"""

import  pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt
from sklearn import svm
from sklearn.metrics import accuracy_score

df = pd.read_csv("E:\ml pro\diabetes.csv")
print("dataset ::")
print(df.head())
print("shape of the data set::",df.shape)
print(df.isnull().sum())
x = df.drop(columns="Outcome",axis=1)
print("x outcome::",x.shape)
y = df["Outcome"]
print("y outcome ::",y)
scaler = StandardScaler()
standard_data= scaler.fit_transform(x)
print("standard data")
print(standard_data)
x=standard_data
y=df['Outcome']

print("standard data:")
print(x)
print("diebetic data")
print(y)
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

classifier=svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

x_train_prediction=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction, y_train)

input_data=(1,85,66,29,0,26.6,0.351,31)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
std_data=scaler.transform(input_data_reshaped)
print(std_data)
prediction=classifier.predict(std_data)
print("prediction",prediction)
if(prediction[0]==0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')