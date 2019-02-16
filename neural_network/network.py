from neural_network.layer import Layer

class Network():
    def __init__(self, learning_rate, num_layers = 1):
        self.learning_rate = learning_rate
        self.num_layers = num_layers

    def create(self, dataset, classes):
        layer = Layer(self.learning_rate)
        layer.create(dataset, classes)

        nodes = layer.nodes

        correct = [node for node in nodes if node.calculateOutput() == node.correct_answer]

        print(len(correct) / len(dataset))
