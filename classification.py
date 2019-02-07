from sklearn.naive_bayes import GaussianNB
from hard_coded_classifier import *
from knn_classifier import *
from DistanceStrategies.euclidien_distance import *
from decision_tree_classifier import DecisionTreeClassifier

GAUSSIAN_CLASSIFIER = '1'
HARD_CODED_CLASSIFIER = '2'
KNN_CLASSIFIER = '3'
DECISION_TREE_CLASSIFIER = '4'

class Classification:
    @staticmethod
    def get(key):
        switcher = {
            GAUSSIAN_CLASSIFIER: GaussianNB(),
            HARD_CODED_CLASSIFIER: HardCodedClassifier(),
            KNN_CLASSIFIER: KNNClassifier(EuclidienDistance(), 3),
            DECISION_TREE_CLASSIFIER: DecisionTreeClassifier(),
        }
        return switcher.get(key)
