import numpy as np
from neural_network.node import Node
import random
import inspect

class Layer:
    def __init__(self, input, num_nodes = 4):
        self.num_nodes = num_nodes
        self.input = input
        self.bias = -1
        self.nodes = self.initialize_nodes(input)
        self.prevLayer = None

    @staticmethod
    def initialize(input, bias = -1, num_nodes = 4):
        layer = Layer([])
        layer.nodes = [Node(input[:], bias) for _ in range(num_nodes)]

    def setInput(self, input):
        self.input = input
        [node.setInput(input) for node in self.nodes]

    def calculateOutput(self):
        return list(map(lambda x: x.calculateOutput(), self.nodes))

    def setBias(self, bias):
        self.bias = bias

    def getValues(self):
        return list(map(lambda x: x.value,self.nodes))

    def initialize_nodes(self, input):
        return [Node(input[:], self.bias) for _ in range(self.num_nodes)]
