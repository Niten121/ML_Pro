# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:06:41 2022

@author: sethy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as  sns 
import warnings 
warnings.filterwarnings('ignore')

dataset= pd.read_csv()
dataset.head()
#to check unique vlues
dataset.nunique()
#shape o data set 
dataset.shape

col = ['suburb,rooms,tyoes ,method,sellerG, region name,propertycount,dist,']
dataset.isna().sum()
col_to_fill_zero = [propertycount,dist.bedroom2,bathroom,car]
dataset[col_to_fill_zero]= dataset[col_to_fill_zero].
dataset['Landsize']= dataset['landsize'].fillna(dataset.Landsize.mean())
dataset["council"]=dataset[]
dataset.dropna(inplace=True)
dataset.shape
dataset= pd .get_dummies(dataset.drop_first= True)
dataset.head()
X = dataset.drop("price", axis=1)
Y = dataset[price]
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split
X_ltrain,X_test,y_train,y_test= train_test_split(X,y,test)

reg.score(X_train,y_train)
reg.score(Xtest,y_test)

from sklearn import linear_model
lasso = liner_model.Lasso(alpha=50,max_iter=100,tot=0.1)
lasso.fit(X_train,y_train)

lasso.score(X_train,y_train)
lasso.score(Xtest,y_test)

from sklear.linear_model  import Ridge
redge = Ridge(alpha=50,max_iter=100,tot=0.1)
redge.fit(X_train,y_train)
redge.score(X_train,y_train)
redge.score(Xtest,y_test)

