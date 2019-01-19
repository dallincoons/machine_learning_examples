from knn_classifier import *
from sklearn import datasets
from DistanceStrategies.euclidien_distance import *

training_data = [
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [5, 5, 5, 5],
    [8, 8, 8, 8],
    [9, 9, 9, 9],
    [10, 10, 10, 10],
]

training_targets = [
    'A',
    'B',
    'C',
    'C',
    'A',
    'C',
]

def test_classifier_predicts_using_knn_algorithm_using_euclidean_distance():
    classifier = KNNClassifier(EuclidienDistance(), k=1)
    classifier.fit(training_data, training_targets)
    result = classifier.predict([[3, 3, 3, 3], [4, 4, 4, 4]])
    assert(['B', 'C'] == result.tolist())

def test_get_most_common_n():
    classifier = KNNClassifier(EuclidienDistance())
    classifier.fit(training_data, training_targets)
    result = classifier.predict([[9, 9, 9, 9]])
