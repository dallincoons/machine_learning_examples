from sklearn.naive_bayes import GaussianNB
from hard_coded_classifier import *

GAUSSIAN_CLASSIFIER = '1'
HARD_CODED_CLASSIFIER = '2'

class Classification:
    @staticmethod
    def get(key):
        switcher = {
            GAUSSIAN_CLASSIFIER: GaussianNB(),
            HARD_CODED_CLASSIFIER: HardCodedClassifier()
        }
        return switcher.get(key)
