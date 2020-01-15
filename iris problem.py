# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:37:51 2020

@author: utkar
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:,1:5].values
y = dataset.iloc[:,5:6].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.67,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)
y_pred_random_forest = classifier.predict(X_test)
cm_random_forest = confusion_matrix(y_test,y_pred_random_forest)

from sklearn.svm import SVC
classifier = SVC(random_state = 0)
classifier.fit(X_train,y_train)
y_pred_svm = classifier.predict(X_test)
cm_svm = confusion_matrix(y_test,y_pred_svm)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)
y_pred_logistic = classifier.predict(X_test)
cm_logistic = confusion_matrix(y_test,y_pred_logistic)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train,y_train)
y_pred_knn = classifier.predict(X_test)
cm_knn = confusion_matrix(y_test,y_pred_knn)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_pred_naive_bayes = classifier.predict(X_test)
cm_naive_bayes = confusion_matrix(y_test,y_pred_naive_bayes)
