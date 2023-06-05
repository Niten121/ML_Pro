# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:25:42 2022

@author: sethy
"""

import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
'''
dataset = pd.read_csv("E:/ml pro/Placement_Dataset.csv")
print(dataset.shape)
print(dataset.isnull().sum())
dataset["salary"].fillna(dataset["salary"].mean(),inplace=True)
'''
df = pd.read_csv("E:\ml pro\Placement_Dataset.csv")
'''print(df.shape)
df = df.head()
df = df.dropna(how='any')'''
cols = ["workex","status","gender"]
le= LabelEncoder()
df[cols]= df[cols].apply(le.fit_transform)
print(df.head())