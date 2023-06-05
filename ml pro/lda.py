# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:21:28 2022

@author: sethy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


df=pd.read_csv("iris_data.csv")
print("data set")
print(df)
print(df.head())
print('shape of data')
print(df.shape)
print('columns of data set')
print(df.columns)
l=LabelEncoder()
label=l.fit_transform(df.Species)
print(df['outcomes'] == label)


x = df.drop(["Id","Species","outcomes"], axis= 1)
y = df.outcome

print(x)
print(y)

x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=.2)  
print('x_train.shape,y_train.shape, x_test.shape, y_test.shape')
print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)

LDA = LinearDiscriminantAnalysis()
lda = LDA(n_components= 1)
x_train_lda= lda.fit_transform(x_train,y_train)
x_test_lda = lda.transdoem(x_test)
x_train_lda.shape
x_test_lda.shape
print(lda.explained_varience_ratio)

lda1 = LDA()
lda1.fit(x_train, y_train)