# Linear Regression
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df=pd.read_csv('salary_data.csv')

print("linear Regression")
print(df.head())

print(df.shape)
print(df.isnull().sum())

print(df.columns)


x=df[['YearsExperience']]
y=df['Salary']

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,random_state = 2)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


from sklearn.linear_model import LinearRegression
reg=LinearRegression().fit(x_train, y_train)

test_data_prediction = reg.predict(x_test)
print(test_data_prediction)

reg.score(x_train, y_train)

reg.score(x_test, y_test)

print("logistic Regression")

# # **Logistic Regression(classification)**
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(max_iter=2000).fit(x_train, y_train)

y_pred = logreg.predict(x_test)
print(y_pred)

logreg.score(x_train, y_train)
logreg.score(x_test, y_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))


print(classification_report(y_test,y_pred))


# # **SVM(classification)**

from sklearn.svm import SVC
model= SVC()
print("SVM")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)

print(model.score(x_train,y_train))

print(model.score(x_test,y_test))


# # **Decision tree(classification)**

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
print('DECISION TREE')

clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# # ## 2) Comparing 4 clustering algorithms:

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



df=pd.read_csv('iris_data.csv')


print(df)


print(df.shape)

print(df.info)



print(df.isnull().sum())


print(df.columns)


l=LabelEncoder()
label=l.fit_transform(df.Species)
df['Outcome']=label



print(df.columns)


x=df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y=df['Outcome']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
StandardScaler(copy=True, with_mean=True, with_std=True)
x = scaler.transform(x)
print(x)


# # **K-means**

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
wcss=[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(x_train,y_train)
    wcss.append(kmeans.inertia_)


plt.plot(range(1,11), wcss, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(x_train,y_train)

y_pre=kmeans.fit_predict(x_test)
print(y_pre)



labels = kmeans.labels_


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pre, labels)
print("Accuracy:", accuracy)


# # **Hierarchial clustering**

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
hierarchical_cluster.fit(x_train,y_train)



labels = hierarchical_cluster.labels_

y_test_pred = hierarchical_cluster.fit_predict(x_test)

linkage_data = linkage(x_test, method='ward', metric='euclidean')

dendrogram(linkage_data)
plt.show()

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_test_pred)
print("Accuracy:", accuracy)


# # **PCA**
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import numpy as np

pca = PCA(n_components=0.95)
x_pca=pca.fit_transform(x_train)

y_pr = pca.transform(x_test)



# # Import the classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(x_pca, y_train)


y_pred = clf.predict(y_pr)


# Compute the accuracy 
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# # **DBSCAN**

from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
import numpy as np


dbscan = DBSCAN()

dbscan.fit(x_train)

labels = dbscan.labels_


y_pred = dbscan.fit_predict(x_test)



accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:", accuracy)


# # **K-means is good for given dataset.**
