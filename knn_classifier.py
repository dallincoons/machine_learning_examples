from math import sqrt
import numpy as np

class KNNClassifier():
    def __init__(self, distance, k = 1):
        self.distance = distance
        self.k = k

    def fit(self, training_data, training_targets):
        self.training_data = training_data
        self.training_targets = training_targets
        self.answers = np.array(list(zip(self.training_data, self.training_targets)))

    def predict(self, test_data):
        return np.array(list(map(lambda data: self.calculatePrediction(data), test_data)))

    def calculatePrediction(self, data):
        distances = self.calculateDistances(data, self.answers)
        return self.mostCommonN(distances)

    def calculateDistances(self, item, answers):
        return list(map(lambda answer: (self.distance.calculateDistance(answer[0], item), answer[1]), answers))

    def mostCommonN(self, distances):
        distances.sort(key=lambda up: up[0])
        candidates = [x[1] for x in distances[0:self.k]]
        return self.mostCommon(candidates)

    def mostCommon(self, candidates):
        return max(set(candidates), key=candidates.count)
