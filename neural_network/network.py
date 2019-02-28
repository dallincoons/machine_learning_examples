from neural_network.layer import Layer

class Network():
    def __init__(self, layer_layouts, learning_rate, dataset):
        self.learning_rate = learning_rate
        self.layers_layouts = layer_layouts
        self.input = dataset
        self.layers = []

        prevLayer = None

        for layer_size in self.layers_layouts:
            layer = Layer(dataset, layer_size)
            dataset = layer.calculateOutput()
            if (prevLayer != None):
                layer.prevLayer = prevLayer
            prevLayer = layer
            self.layers.append(layer)

    def calculateOutput(self):
        dataset = self.input
        for layer in self.layers:
            layer.setInput(dataset)
            dataset = layer.calculateOutput()

        return self.layers[-1].calculateOutput()
