import numpy as np
from neural_network.network import Network

class NeuralNetClassifier():
    def fit(self, training_data, training_targets):
        print('----')

        print(training_data)

        # self.network.calculateOutput(training_data.tolist()[0])
        # for key, training in enumerate(training_data.tolist()):
        self.network = Network([4, len(np.unique(training_targets))], .1, training_data.tolist()[0])
        self.network.calculateOutput()
            # targets = (self.target_pad(training_data.shape[1] - 1, training_targets[key]))

    def predict(self, test_data):
        outputs = list(map(lambda data: self.network.calculateOutput(), test_data.tolist()))

        return np.array(list(map(self.highest_index, outputs)))

    def highest_index(self, output):
        return np.argmax(output)

    def target_pad(self, target_length, target):
        empty = [0] * target_length
        empty[target] = 1
        return empty
