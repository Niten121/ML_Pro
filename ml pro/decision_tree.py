# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:22:35 2022

@author: sethy
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('diabetes.csv')
print(df.head())

X = df.drop('Outcome',axis = 1)
y = df.Outcome
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 

# Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train
clf = clf.fit(X_train,y_train)

#Prediction
y_pred = clf.predict(X_test)

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


clf = DecisionTreeClassifier(criterion="entropy", max_depth=2)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

plot_tree(clf) 

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier  
model= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
model.fit(X_train, y_train) 
from sklearn import metrics
metrics.plot_confusion_matrix(model, X_test, y_test, display_labels=['non - diabetic', 'diabetic'])