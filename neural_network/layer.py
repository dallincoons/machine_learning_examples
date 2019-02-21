import numpy as np
from neural_network.node import Node
import random

class Layer:
    def __init__(self, num_nodes = 4, threshold = 0):
        self.threshold = threshold
        self.num_nodes = num_nodes
        self.bias = -1
        self.nodes = []

    def setBias(self, bias):
        self.bias = bias

    def create(self, inputs, classes):
        self.nodes = self.initialize_nodes(inputs)
        return self

    def initialize_nodes(self, input):
        return [Node(input[:], self.bias) for _ in range(self.num_nodes)]
