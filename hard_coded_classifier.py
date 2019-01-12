from iris_targets import *

class HardCodedClassifier():
    def fit(self, data, targets):
        pass

    def predict(self, data):
        return [SETOSA for _ in data]
