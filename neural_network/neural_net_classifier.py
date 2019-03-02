import numpy as np
from neural_network.network import Network

class NeuralNetClassifier():
    def fit(self, training_data, training_targets):
        print('----')

        self.network = Network([4, len(np.unique(training_targets))], len(training_data.tolist()[0]), len(set(training_targets)), .5)

        for i in range(0, 300):
            for key, training in enumerate(training_data.tolist()):
                self.network.train(training)
                self.network.layers[-1].calculateErrors(self.target_pad(len(training_targets), training_targets[key]))
                self.network.updateWeights()

    def predict(self, test_data):
        outputs = list(map(lambda data: self.network.calculateOutput(data), test_data.tolist()))

        return np.array(list(map(self.highest_index, outputs)))

    def highest_index(self, output):
        return np.argmax(output)

    def target_pad(self, target_length, target):
        empty = [0] * target_length
        empty[target] = 1
        return empty
