# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:51:07 2022

@author: sethy
"""

#converting the lables into numeric value
#handline imbalance dataset

import pandas as pd
import numpy as np

df = pd.read_csv()

df.head()
df.shape
df['Class'].value.counts()
legit = df[df.Class == 0]
fraud = df[df.Class == 1]
print(legit.shape)
print(fraud.shape)

legit_sample = legit.sample(n=500)
print(legit_sample.shape())

#concatenate legit sample and fraud
new_dataset = pd.concate([legit_sample],axis=0)
print(new_dataset.shape)
print(new_dataset['Class'].value_counts())