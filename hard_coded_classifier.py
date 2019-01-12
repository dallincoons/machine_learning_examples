from iris_targets import *
import numpy as np

class HardCodedClassifier():
    def fit(self, data, targets):
        pass

    def predict(self, data):
        return np.array([SETOSA for _ in data])
