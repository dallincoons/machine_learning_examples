from knn_classifier import *
from sklearn import datasets

training_data = [
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [5, 5, 5, 5],
]

training_targets = [
    'uno',
    'dos',
    'tres',
]

def test_classifier_predicts_using_knn_algorithm():
    classifier = KNNClassifier()
    classifier.fit(training_data, training_targets)
    result = classifier.predict([[3, 3, 3, 3], [4, 4, 4, 4]])
    assert(['dos', 'tres'] == result)

