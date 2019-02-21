from neural_network.layer import Layer

class Network():
    def __init__(self, layer_layouts, learning_rate):
        self.learning_rate = learning_rate
        self.layers_layouts = layer_layouts
        self.layers = []

    def create(self, dataset, classes):
        # for index, input in enumerate(dataset):
        dataset = dataset[0]
        for layer_size in self.layers_layouts:
            layer = Layer(dataset, layer_size)
            dataset = layer.calculateOutput()
            self.layers.append(layer)
