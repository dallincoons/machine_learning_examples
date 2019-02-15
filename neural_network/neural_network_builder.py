import numpy as np

class NeuralNetworkBuilder:
    def __init__(self, learning_rate, threshold = 0):
        self.learning_rate = learning_rate
        self.threshold = threshold

    def create(self, input, classes):
        x = np.array(input)
        y = np.array(classes).T

        weights = np.random.random((x.shape[1], 1))

        z = np.dot(x, weights)

        return list(map(self.is_greater_than_threshold, z))

    def is_greater_than_threshold(self, value):
        if value >= self.threshold:
            return 1
        return 0
