# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:26:33 2020

@author: tharunivavilala
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 07:34:55 2020

@author: Lucky
"""


# ' Dual Degree Bachelor of Technology - Master of Technology (Mechanical Engineering)'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\Users\\sri\\Desktop\\data.csv')
df.head(5)
df = df.loc[:, ['Grade', 'CA_100', 'MTT_50', 'ETT_100',
       'ETP_100', 'Course_Att', 'MHRDName']]
print(df.shape)

d = df.loc[df['MHRDName'] == 'Dual Degree Bachelor of Technology - Master of Technology (Mechanical Engineering)']
print(d.shape[0])
n = df.shape[0]//4
df = df.head(n)
print(df.shape)
x = df.iloc[:, 1:6]
y = df.iloc[:, 0]
print(df.columns)

from sklearn.impute import SimpleImputer
x = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0).fit_transform(x)


from sklearn.preprocessing import LabelEncoder, StandardScaler
y = LabelEncoder().fit_transform(y)
x = StandardScaler().fit_transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)




df['Grade'].value_counts().plot(kind='barh', title = 'Grade')
plt.show()
plt.show()
df['MTT_50'].value_counts().plot(kind='hist', title = 'MTT_50')
plt.show()
df['ETT_100'].value_counts().plot(kind='hist', title = 'ETT_100')
plt.show()
df['ETP_100'].value_counts().plot(kind='hist', title = 'ETP_100')
plt.show()
df['CA_100'].value_counts().plot(kind='hist', title = 'CA_100')
plt.show()


from sklearn.svm import SVC
svm = SVC(C=10,kernel='linear')
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))

plt.scatter(y_test, svm_pred, color = 'red')
plt.title('SVC')
plt.xlabel('Target')
plt.ylabel('Predicted')
plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

plt.scatter(y_test, y_pred, color = 'red')
plt.title('KNeighborsClassifier')
plt.xlabel('Target')
plt.ylabel('Predicted')
plt.show()

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='gini')
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Accuracy Score :")
print(accuracy_score(y_pred,y_test))

plt.scatter(y_test, y_pred, color = 'red')
plt.title('DecisionTreeClassifier')
plt.xlabel('Target')
plt.ylabel('Predicted')
plt.show()


