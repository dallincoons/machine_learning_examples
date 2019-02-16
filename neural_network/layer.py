import numpy as np
from neural_network.node import Node
import random

class Layer:
    def __init__(self, learning_rate, num_nodes = 4, threshold = 0):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.num_nodes = num_nodes
        self.bias = -1
        self.nodes = []

    def setBias(self, bias):
        self.bias = bias

    def create(self, inputs, classes):
        inputs = list(zip(inputs, classes))
        for input in inputs:
            node = Node(input[1])
            for attribute in input[0]:
                node.addInput(attribute, random.uniform(-1, 1))
            self.nodes.append(node)

    def initialize_nodes(self, node_count):
        return [Node(self.bias) for _ in range(node_count)]
