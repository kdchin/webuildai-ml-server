import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def run_decision_tree(data, nsplits, test_size, train_size):
    X_new = []
    y_new = []

    for i in range(len(data)):
        x1 = np.array(data[i][0])
        x2 = np.array(data[i][1])
        label = np.array(data[i][2])

        x_diff = x1 - x2
        X_new.append(x_diff)

        if (label == 1):
            y_new.append(0)
        elif (label == 2):
            y_new.append(1)

    X_new = np.array(X_new)
    y_new = np.array(y_new)

    kf = ShuffleSplit(n_splits=nsplits, test_size=test_size, train_size=train_size)

    accuracy = []
    for train_index, test_index in kf.split(X_new):
        X_train, X_test = X_new[train_index], X_new[test_index]
        y_train, y_test = y_new[train_index], y_new[test_index]

        print(pd.value_counts(y_train))
        clf = DecisionTreeClassifier(max_depth=2)
        clf.fit(X_train, y_train)

        clf_predictions = clf.predict(X_test)
        clf_score = clf.predict_proba(X_test)
        accuracy.append(float(np.sum(y_test == clf_predictions)) / len(y_test))

    return accuracy, clf