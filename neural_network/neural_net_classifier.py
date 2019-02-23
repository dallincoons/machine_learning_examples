import numpy as np
from neural_network.network import Network

class NeuralNetClassifier():
    def fit(self, training_data, training_targets):
        print('----')
        self.network = Network([4, len(np.unique(training_targets))], .1)
        for training in training_data.tolist():
            self.network.calculateOutput(training)

    def predict(self, test_data):
        outputs = list(map(lambda data: self.network.calculateOutput(data), test_data.tolist()))

        return np.array(list(map(self.highest_index, outputs)))

    def highest_index(self, output):
        return np.argmax(output)
