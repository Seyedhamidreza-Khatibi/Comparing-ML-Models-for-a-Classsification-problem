# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
##############################
#Logistic Regression
##############################
# Feature Scaling(It is not necessary in logistic regression but it will improve the performance)
from sklearn.preprocessing import StandardScaler
#scaling features(y is already 0 and 1 so no need to be scaled)
sc = StandardScaler()
X_train1= sc.fit_transform(X_train)
X_test1= sc.transform(X_test)
# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train1, y_train)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred1= classifier.predict(X_test1)
cm1=confusion_matrix(y_test, y_pred1)
# print(cm1)
print('Logistic Regression=',accuracy_score(y_test, y_pred1))
##############################
#K-Nearest Neighbors (K-NN)
##############################
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2= sc.fit_transform(X_train)
X_test2= sc.transform(X_test)
# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train2, y_train)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred2= classifier.predict(X_test2)
cm2= confusion_matrix(y_test, y_pred2)
# print(cm2)
print("K-Nearest Neighbors (K-NN)=",accuracy_score(y_test, y_pred2))

##############################
# Decision Tree Classification
##############################
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train3= sc.fit_transform(X_train)
X_test3= sc.transform(X_test)
# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)#criterion{“gini”, “entropy”}, default=”gini”
classifier.fit(X_train3, y_train)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred3= classifier.predict(X_test3)
cm3= confusion_matrix(y_test, y_pred3)
# print(cm3)
print("Decision Tree Classification=",accuracy_score(y_test, y_pred3))


##############################
# Random Forest Classification
##############################

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train4= sc.fit_transform(X_train)
X_test4= sc.transform(X_test)
# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train4, y_train)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred4= classifier.predict(X_test4)
cm4= confusion_matrix(y_test, y_pred4)
# print(cm4)
print("Random Forest Classification=",accuracy_score(y_test, y_pred4))

##############################
# Naive Bayes
##############################

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train5= sc.fit_transform(X_train)
X_test5= sc.transform(X_test)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train5, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred5= classifier.predict(X_test5)
cm5= confusion_matrix(y_test, y_pred5)
# print(cm4)
print("Naive Bayes=",accuracy_score(y_test, y_pred5))

#########################################
# Support Vector Machine (SVM)-Linear SVM
#########################################

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train6= sc.fit_transform(X_train)
X_test6= sc.transform(X_test)

# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train6, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred6 = classifier.predict(X_test6)
cm6= confusion_matrix(y_test, y_pred6)
# print(cm6)
print("Support Vector Machine (SVM)-Linear SVM=",accuracy_score(y_test, y_pred6))

############################
# Kernel SVM-Nonlinear SVM
############################

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train7= sc.fit_transform(X_train)
X_test7= sc.transform(X_test)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)#kernel{‘linear’,‘rbf’, ‘poly’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
classifier.fit(X_train7, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred7 = classifier.predict(X_test7)
cm7 = confusion_matrix(y_test, y_pred7)
# print(cm7)
print("Kernel SVM-Nonlinear SVM=",accuracy_score(y_test, y_pred7))

##############################
##XGBoost-->Scaling is not needed
##############################
# Training XGBoost on the Training set
#for classification
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred8 = classifier.predict(X_test)
cm8 = confusion_matrix(y_test, y_pred8)
# print(cm8)
print("XGBoost=",accuracy_score(y_test, y_pred8))