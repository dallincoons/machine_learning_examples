from neural_network.layer import Layer

class Network():
    def __init__(self, layer_layouts, initial_num, output_num, learning_rate = .1):
        self.learning_rate = learning_rate
        self.layers_layouts = layer_layouts
        self.layers = []

        prevLayer = Layer(initial_num)

        for layer_size in self.layers_layouts:
            layer = Layer(layer_size)
            if (prevLayer != None):
                layer.setPreviousLayer(prevLayer)
            prevLayer = layer
            self.layers.append(layer)

    def calculateOutput(self, dataset):
        self.train(dataset)
        return self.layers[-1].calculateOutput()

    def train(self, dataset):
        input_layer = self.layers[0]
        input_layer.setInput(dataset)

        for layer in self.layers[1:]:
            layer.setInput(input_layer.calculateOutput())
            input_layer = layer
        input_layer.calculateOutput()

    def updateWeights(self):
        for layer in self.layers[1:]:
            layer.updateWeights(self.learning_rate)
