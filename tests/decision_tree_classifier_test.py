from decision_tree_classifier import *
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

def test_decision_tree():
    iris = datasets.load_iris()

    data_train, data_test, targets_train, target_test = train_test_split(
        iris.data,
        iris.target,
        train_size=.70,
        test_size=.30,
        random_state=0
    )

    # classifier = DecisionTreeClassifier()
    # classifier.fit(data_train, targets_train)
