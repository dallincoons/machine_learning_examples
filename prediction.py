class Prediction:
    def __init__(self, data_train, targets_train, data_test):
       self.data_train = data_train
       self.targets_train = targets_train
       self.data_test = data_test

    def runWith(self, classifier):
        classifier.fit(self.data_train, self.targets_train)

        targets_predicted = classifier.predict(self.data_test)

        print(targets_predicted)

