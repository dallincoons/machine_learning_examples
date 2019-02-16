import random
import numpy as np

class Node:
    def __init__(self, correct_answer, bias = -1, threshold = 0):
        self.inputs = []
        self.weights = []
        self.correct_answer = correct_answer
        self.bias = bias
        self.bias_weight = random.uniform(-1, 1)
        self.threshold = threshold

    def addInput(self, input, weight):
        self.inputs.append(input)
        self.weights.append(weight)

    def calculateOutput(self):
        inputs = self.inputs
        weights = self.weights
        inputs.append(self.bias)
        weights.append(self.bias_weight)

        dot_product = np.dot(inputs, weights)

        return self.one_or_zero(dot_product)

    def one_or_zero(self, value):
        if (value > self.threshold):
            return 1
        return 0
