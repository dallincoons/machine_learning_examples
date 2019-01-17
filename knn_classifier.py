from math import sqrt
import numpy as np

class KNNClassifier():
    def fit(self, training_data, training_targets):
        self.training_data = training_data
        self.training_targets = training_targets
        self.answers = list(zip(self.training_data, self.training_targets))

    def predict(self, test_data):
        return np.array(list(map(lambda data: self.calculatePrediction(data), test_data)))

    def calculatePrediction(self, data):
        distances = self.calculateDistances(data)
        return self.mostCommonN(distances)

    def calculateDistances(self, item):
        return list(map(lambda answer: self.calculateDistance(answer, item), self.answers))

    def calculateDistance(self, answer, item):
        answerData = answer[0]
        distance = sqrt((answerData[0] - item[0])**2 + (answerData[1] - item[1])**2 + (answerData[2] - item[2])**2 + (answerData[3] - item[3])**2)
        return (distance, answer[1])

    def mostCommonN(self, distances):
        distances.sort(key=lambda tup: tup[0])
        return distances[0][1]
