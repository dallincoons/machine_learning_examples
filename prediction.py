import numpy as np

class Prediction:
    def __init__(self, data_train, targets_train, data_test, target_test):
       self.data_train = data_train
       self.targets_train = targets_train
       self.data_test = data_test
       self.target_test = target_test


    def runWith(self, classifier):
        """
        run test data through the classifer and return a decimal number
        representing the accuracy percentage
        """
        classifier.fit(self.data_train, self.targets_train)

        targets_predicted = classifier.predict(self.data_test)

        innacurateResults = self.getInnacurateResults(targets_predicted, self.target_test)

        return '%.2f'%((len(targets_predicted) - len(innacurateResults)) / len(targets_predicted))

    def getInnacurateResults(self, targets_predicted, target_test):
        return [targets_predicted[i] for i in range(len(targets_predicted.tolist())) if targets_predicted[i] != target_test[i]]
