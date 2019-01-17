from sklearn.naive_bayes import GaussianNB
from hard_coded_classifier import *
from knn_classifier import *
from DistanceStrategies.euclidien_distance import *

GAUSSIAN_CLASSIFIER = '1'
HARD_CODED_CLASSIFIER = '2'
KNN_CLASSIFIER = '3'

class Classification:
    @staticmethod
    def get(key):
        switcher = {
            GAUSSIAN_CLASSIFIER: GaussianNB(),
            HARD_CODED_CLASSIFIER: HardCodedClassifier(),
            KNN_CLASSIFIER: KNNClassifier(EuclidienDistance()),
        }
        return switcher.get(key)
