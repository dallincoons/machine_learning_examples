from knn_classifier import *
from sklearn import datasets
from DistanceStrategies.euclidien_distance import *

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

def test_classifier_predicts_using_knn_algorithm_using_euclidean_distance():
    classifier = KNNClassifier(EuclidienDistance())
    classifier.fit(training_data, training_targets)
    result = classifier.predict([[3, 3, 3, 3], [4, 4, 4, 4]])
    assert(['dos', 'tres'] == result.tolist())

