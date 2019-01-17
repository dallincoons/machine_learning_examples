from math import sqrt
import numpy as np

class KNNClassifier():
    def __init__(self, distance):
        self.distance = distance

    def fit(self, training_data, training_targets):
        self.training_data = training_data
        self.training_targets = training_targets
        self.answers = list(zip(self.training_data, self.training_targets))

    def predict(self, test_data):
        return np.array(list(map(lambda data: self.calculatePrediction(data), test_data)))

    def calculatePrediction(self, data):
        distances = self.calculateDistances(data, self.answers)
        return self.mostCommonN(distances)

    def calculateDistances(self, item, distances):
        return list(map(lambda answer: (self.distance.calculateDistance(answer[0], item), answer[1]), distances))

    def mostCommonN(self, distances):
        distances.sort(key=lambda tup: tup[0])
        return distances[0][1]
