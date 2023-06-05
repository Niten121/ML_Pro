            #iris data set using KNN

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

df=pd.read_csv("heart.csv")
print("data set")
print(df)
print(df.head())
print('shape of data')
print(df.shape)
print('columns of data set')
print(df.columns)

l=LabelEncoder()
label=l.fit_transform(df.target)
df["target"]=label
df.head()
print(df.head())


X = df.drop(['target'], axis='columns')
y = df.target
print(X)
print(y)

classifier = SVC(kernel='linear', random_state=0)  
x_train, x_test, y_train,y_test=train_test_split(X,y,test_size=.2, random_state=2)  
print(classifier.fit(x_train, y_train))
print('x_train.shape,y_train.shape, x_test.shape, y_test.shape')
print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)

y1_pred= classifier.predict(x_test)  
print("y predict")
print(y1_pred)
print('x_test shape')
print(x_test.shape)
print('y_test shape')
print(y_test.shape)

 
cm= confusion_matrix(y_test, y1_pred)
print("cm=", cm)

model= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
print(model)
print(model.fit(x_train, y_train))

y_pred= model.predict(x_test)  
print(y_pred)

print(y_test)

cm= confusion_matrix(y_test, y_pred)  
print('confusion matrix::')
print(cm)
print(y_pred)
 
cm1= confusion_matrix(y_test, y_pred)
print('cm1')
print(cm1)

print(metrics.classification_report(y_test, y_pred))

metrics.plot_confusion_matrix(model, x_test, y_test, display_labels=['No heart', 'heart'])
confusion = metrics.confusion_matrix(y_test, y_pred)
print('confusion ')
print(confusion.ravel())
accuracy = metrics.accuracy_score(y_test, y1_pred)
print('accuracy')
print(accuracy) 



