from DataStructures.decision_tree import DecisionTree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

TEST_INPUTS = [
    ['good', 'high', 'good'],
    ['good', 'high', 'poor'],
    ['good', 'low', 'good'],
    ['good', 'low', 'poor'],
    ['average', 'high', 'good'],
    ['average', 'low', 'poor'],
    ['average', 'high', 'poor'],
    ['average', 'low', 'good'],
    ['low', 'high', 'good'],
    ['low', 'high', 'poor'],
    ['low', 'low', 'good'],
    ['low', 'low', 'poor'],
]

CLASSES = [
    'y',
    'y',
    'y',
    'n',
    'y',
    'n',
    'y',
    'n',
    'y',
    'n',
    'n',
    'n',
]

def test_find_class_percentages():
    tree = DecisionTree.build(pd.DataFrame(TEST_INPUTS), pd.DataFrame(CLASSES))
    percentages = tree.class_percentages(pd.DataFrame(['y', 'y', 'y', 'n']))

    assert(percentages == [.75, .25])

    percentages = tree.class_percentages(pd.DataFrame(['y', 'y', 'y']))

    assert(percentages == [1, 0])

def test_find_lowest_entropy_attribute():
    tree = DecisionTree.build(pd.DataFrame(TEST_INPUTS), pd.DataFrame(CLASSES))

    assert(1 == tree.find_attribute_with_lowest_entropy())

def test_find_attribute_with_lowest_entropy():
    tree = DecisionTree.build(pd.DataFrame(TEST_INPUTS), pd.DataFrame(CLASSES))

    tree = DecisionTreeClassifier()

    # iris = load_iris()
    # clf = tree.fit(iris.data, iris.target)
    # predicitions = clf.predict([[5.1, 3.5, 1.4, 0.2]])
    # print(predicitions)

    # assert(1 == tree.next_attribute())

