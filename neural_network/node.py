import random
import numpy as np

class Node:
    def __init__(self, input, bias = -1):
        self.inputs = input
        self.weights = list(map(lambda x: random.uniform(-1,1), range(0, len(input))))
        self.bias = bias
        self.bias_weight = random.uniform(-1, 1)

    def addInput(self, input, weight):
        self.inputs.append(input)
        self.weights.append(weight)

    def calculateOutput(self):
        inputs = self.inputs
        weights = self.weights
        inputs.append(self.bias)
        weights.append(self.bias_weight)

        self.value = np.dot(inputs, weights)

        return self.one_or_zero(self.value)

    def one_or_zero(self, value):
        if (value > random.uniform(-1, 1)):
            return 1
        return 0
