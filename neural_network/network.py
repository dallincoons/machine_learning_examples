from neural_network.layer import Layer

class Network():
    def __init__(self, layer_layouts, learning_rate):
        self.learning_rate = learning_rate
        self.layers_layouts = layer_layouts
        self.layers = []

    def calculateOutput(self, dataset):
        prevLayer = None
        if len(self.layers) == 0:
            for layer_size in self.layers_layouts:
                layer = Layer(dataset, layer_size)
                dataset = layer.calculateOutput()
                if (prevLayer != None):
                    layer.prevLayer = prevLayer
                prevLayer = layer
                self.layers.append(layer)
        else:
            for layer in self.layers:
                layer.setInput(dataset)
                dataset = layer.calculateOutput()

        return self.layers[-1].calculateOutput()
