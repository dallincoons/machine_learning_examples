from neural_network.layer import Layer

class Network():
    def __init__(self, layer_layouts, learning_rate):
        self.learning_rate = learning_rate
        self.layers_layouts = layer_layouts
        self.layers = []

    def create(self, dataset):
        if len(self.layers) == 0:
            for layer_size in self.layers_layouts:
                layer = Layer(dataset, layer_size)
                dataset = layer.calculateOutput()
                self.layers.append(layer)
        else:
            for layer in self.layers:
                layer.setInput(dataset)
                dataset = layer.calculateOutput()

        return self.layers[-1].calculateOutput()
