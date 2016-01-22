# coding: utf-8

# http://pandas.pydata.org/
# http://scikit-learn.org/
# data: https://archive.ics.uci.edu/ml/datasets/Covertype

import numpy as np
import pandas as pd


### Fetch the data and load it in pandas
data = pd.read_csv('train.csv')
print "Size of the data: ", data.shape

# See data (five rows) using pandas tools
#print data.head()


### Prepare input to scikit and train and test cut

binary_data = data[np.logical_or(data['Cover_Type'] == 1,data['Cover_Type'] == 2)] # two-class classification set
X = binary_data.drop('Cover_Type', axis=1).values
y = binary_data['Cover_Type'].values
#print np.unique(y)
y = 2 * y - 3 # converting labels from [1,2] to [-1,1]

# Import cross validation tools from scikit
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=None)


### Train a single decision tree

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=8)

# Train the classifier and print training time
clf.fit(X_train, y_train)

# Do classification on the test dataset and print classification results
from sklearn.metrics import classification_report
target_names = data['Cover_Type'].unique().astype(str).sort()
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))

# Compute accuracy of the classifier (correctly classified instances)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)




#===================================================================
### Train AdaBoost

D = 2 # tree depth
T = 1000 # number of trees
n = X_train.shape[0]
w = np.ones(n) / n
training_scores = np.zeros(X_train.shape[0])
test_scores = np.zeros(X_test.shape[0])
weak = []
alphas = []

training_errors = []
test_errors = []

for t in range(T):
    clf = DecisionTreeClassifier(max_depth=D)
    weak.append(clf)

    clf.fit(X_train, y_train, sample_weight=w)
    y_pred = clf.predict(X_train)
    rate = w.dot(np.not_equal(y_pred, y_train)) / sum(w)
    alpha = np.log(1 / rate - 1)
    alphas.append(alpha)

    for i in range(n):
        aux = 0
        if y_pred[i] != y_train[i]:
            aux = 1
        w[i] = w[i] * np.exp(alpha * aux)

for i in range(T):
    training_scores += alphas[i] * weak[i].predict(X_train)
    test_scores += alphas[i] * weak[i].predict(X_test)
    training_errors.append(np.sum(np.sign(training_scores) != y_train) * 1. / X_train.shape[0])
    test_errors.append(np.sum(np.sign(test_scores) != y_test) * 1. / X_test.shape[0])

#  Plot training and test error
import matplotlib.pyplot as plt
import pylab
plt.plot(training_errors, label="training error")
plt.plot(test_errors, label="test error")
plt.legend()
pylab.show()

#===================================================================
### Optimize AdaBoost

# Your final exercise is to optimize the tree depth in AdaBoost.
# Copy-paste your AdaBoost code into a function, and call it with different tree depths
# and, for simplicity, with T = 100 iterations (number of trees). Plot the final
# test error vs the tree depth. Discuss the plot.

#===============================

# Your code should go here


#===============================
