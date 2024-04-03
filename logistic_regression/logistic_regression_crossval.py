import numpy as np
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

datadir ="256_2k/"
features = np.load(datadir + "256_2k_features.npy", allow_pickle = True)
labels = np.load(datadir + "256_2k_labels.npy", allow_pickle = True)
cases = np.load(datadir + "256_2k_cases.npy", allow_pickle = True)

X_train = features
y_train = labels

#Standardize the data
scaler = StandardScaler()
X = np.array(X_train)
num_samples, width, height = X.shape
X_train = X.reshape(num_samples, width * height)
X_train = scaler.fit_transform(X_train)

y_train_labels = [1 if value == "high" else 0 for value in y_train]

#Set up grid search
logistic = LogisticRegression()

param_grid = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(estimator=logistic, param_grid=param_grid, cv=5, scoring='accuracy', verbose = 2)
grid_search.fit(X_train, y_train_labels)

#Get the best parameters from grid search
c = grid_search.best_params_['C']
penalty = grid_search.best_params_['penalty']
best_score = grid_search.best_score_

print("Best Parameters:", c, penalty)
print("Best Score:", best_score)

#Set up logistic regression model 
logreg_model = LogisticRegression( solver = "liblinear", max_iter=1000, random_state=88, verbose = 1, C = c, penalty = penalty)

#Perform cross validation on the entire training set and measure accuracy
cv_scores = cross_val_score(logreg_model, X_train, y_train_labels, cv=5, scoring='accuracy')

# Calculate accuracy scores 
mean_cv_score = cv_scores.mean()
std_cv_score = cv_scores.std()
print("Accuracy cross-validation scores")
print("Mean CV Score:", mean_cv_score)
print("Standard Deviation of CV Scores:", std_cv_score)
print("Scores:", cv_scores)

#Perform cross validation on the entire training set and measure f1
cv_scores = cross_val_score(logreg_model, X_train, y_train_labels, cv=5, scoring='f1')

# Calculate f1 scores for cross validation
mean_cv_score = cv_scores.mean()
std_cv_score = cv_scores.std()
print("Accuracy cross-validation scores")
print("Mean CV Score:", mean_cv_score)
print("Standard Deviation of CV Scores:", std_cv_score)
print("Scores:", cv_scores)
