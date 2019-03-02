import numpy as np
from neural_network.node import Node
import random
import inspect

class Layer:
    def __init__(self, num_nodes = 4):
        self.num_nodes = num_nodes
        self.bias = -1
        self.prevLayer = None
        self.nodes = []

    def setInput(self, input):
        self.input = input
        [node.setInput(input) for node in self.nodes]

    def setPreviousLayer(self, layer):
        self.prevLayer = layer
        self.nodes = self.initialize_nodes()

    def calculateOutput(self):
        return list(map(lambda x: x.calculateOutput(), self.nodes))

    def calculateErrors(self, targets):
        for key, node in enumerate(self.nodes):
            node.calculateError(targets[key])

    def updateWeights(self, learning_rate):
        for node in self.nodes[0:-1]:
            node.updateWeights(learning_rate)

    def setBias(self, bias):
        self.bias = bias

    def getValues(self):
        return list(map(lambda x: x.value,self.nodes))

    def initialize_nodes(self):
        return [Node(self.prevLayer.num_nodes, self.bias) for _ in range(self.num_nodes)]
