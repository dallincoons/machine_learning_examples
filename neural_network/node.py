import random
import numpy as np
from neural_network.activation_function import ActivationFunction

class Node:
    def __init__(self, input, bias = -1):
        self.inputs = input
        self.weights = list(map(lambda x: random.uniform(-1,1), range(0, len(input))))
        self.bias = bias
        self.bias_weight = random.uniform(-1, 1)
        self.weights.append(self.bias_weight)
        self.inputs.append(bias)

    def setInput(self, input):
        self.inputs = input

    def addInput(self, input, weight):
        self.inputs.append(input)
        self.weights.append(weight)

    def calculateOutput(self):
        inputs = self.inputs[:]
        weights = self.weights[:]

        self.value = np.dot(inputs, weights)
        self.activation = ActivationFunction.sigmoid(self.value)

        return self.activation

    def calculateError(self, target):
        self.error = self.activation * (1 - self.activation) * (self.activation - target)
        return self.error

    def calculateHiddenError(self, layer, targets):
        nodeErrors = 0
        for key, node in enumerate(layer.nodes):
            nodeErrors += node.weights[key + 1] * node.calculateError(targets[key])

        self.error = self.activation * (1 - self.activation) * nodeErrors
        return self.error

    def updateWeights(self, learning_rate):
        self.bias_weight = self.weights[-1]
        self.weights[-1] = self.bias_weight - (learning_rate * self.error * self.inputs[-1])

        for key, weight in enumerate(self.weights[0:-1]):
            self.weights[key] = weight - (learning_rate * self.error * self.inputs[key])

