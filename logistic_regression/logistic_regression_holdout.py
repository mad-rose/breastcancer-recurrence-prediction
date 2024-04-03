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

#256 20000
datadir ="256_2k/"
features = np.load(datadir + "256_2k2_features.npy", allow_pickle = True)
labels = np.load(datadir + "256_2k2_labels.npy", allow_pickle = True)
cases = np.load(datadir + "256_2k2_cases.npy", allow_pickle = True)

X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.2, random_state=88)

#Used to determine the cases used in the split
#X_train, X_test, y_train, y_test = train_test_split(features,cases, test_size=0.2, random_state=88)

#Standardize the data
scaler = StandardScaler()
X = np.array(X_train)
num_samples, width, height = X.shape
X_train = X.reshape(num_samples, width * height)
X_train = scaler.fit_transform(X_train)

X = np.array(X_test)
num_samples, width, height = X.shape
X_test = X.reshape(num_samples, width * height)
X_test = scaler.transform(X_test)

y_train_labels = [1 if value == "high" else 0 for value in y_train]
y_test_labels = [1 if value == "high" else 0 for value in y_test]


param_grid = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

#Set up grid search to determine the best parameters
logistic = LogisticRegression()
grid_search = GridSearchCV(estimator=logistic, param_grid=param_grid, cv=5, scoring='accuracy', verbose = 2)
grid_search.fit(X_train, y_train_labels)

#Report the top parameters
c = grid_search.best_params_['C']
penalty = grid_search.best_params_['penalty']
best_score = grid_search.best_score_

print("Best Parameters:", c, penalty)
print("Best Score:", best_score)

#Fit logistic regression on the training data
logreg_model = LogisticRegression( solver = "liblinear", max_iter=1000, random_state=88, verbose = 1, C = 10.0, penalty = 'l1')
logreg_model.fit(X_train, y_train_labels)

#Perform testing and report accuracy
y_pred = logreg_model.predict(X_test)
accuracy = accuracy_score(y_test_labels, y_pred)
print("Accuracy:", accuracy)

# Perform 5-fold cross-validation on the training data
cv_scores = cross_val_score(logreg_model, X_train, y_train_labels, cv=5, scoring='accuracy')

# Report cross-validation accuracy
mean_cv_score = cv_scores.mean()
std_cv_score = cv_scores.std()
print("Mean CV Score:", mean_cv_score)
print("Standard Deviation of CV Scores:", std_cv_score)
print("Scores:", cv_scores)

#Calculate values for true positive, false positive, false negative, and true negative
tn, fp, fn, tp = confusion_matrix(y_test_labels, y_pred).ravel()

#Calculate values for f1 score
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
f1score = f1_score(y_test_labels, y_pred)

#Report f1 results
print("True positive:", tp)
print("False positive:", fp)
print("True negative:", tn)
print("False negative:", fn)
print("Sensitivity (True Positive Rate, Recall):", sensitivity)
print("Specificity (True Negative Rate):", specificity)
print("F1 score:", f1score)

# Calculate AUC-ROC score
y_prob = logreg_model.predict_proba(X_test)[:, 1]
auc_roc = roc_auc_score(y_test_labels, y_prob)
print("AUC-ROC Score:", auc_roc)


'''
Confusion matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()'''