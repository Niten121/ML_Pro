

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv("customer.csv")
print(data.head())

data["Income"] = data[["Annual Income (k$)"]]
data["Spending"] = data[["Spending Score (1-100)"]]
data = data[["Income", "Spending"]]
print(data.head())

from sklearn.preprocessing import normalize
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
print(data_scaled.head())

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(5, 5))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))

from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering()
model.fit(data)
pred = model.fit_predict(data)
score = accuracy_score(model, pred)  
print(score)  
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(5, 5))
plt.scatter(data["Income"], data["Spending"], c=pred, cmap='rainbow', alpha=1)
plt.show()

