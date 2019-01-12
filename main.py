from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from hard_coded_classifier import *


def main():
    iris = datasets.load_iris()

    data_train, data_test, targets_train, target_test = train_test_split(
        iris.data,
        iris.target,
        train_size=.70,
        test_size=.30,
        stratify=iris.target,
        shuffle=True
    )

    runGaussianNBPrediction(data_train, targets_train, data_test)
    runHardCodedPrediction(data_train, targets_train, data_test)


def runGaussianNBPrediction(data_train, targets_train, data_test):
    classifier = GaussianNB()
    classifier.fit(data_train, targets_train)

    targets_predicted = classifier.predict(data_test)

    print('GaussianNB prediction: ')
    print(targets_predicted)

def runHardCodedPrediction(data_train, targets_train, data_test):
    classifier = HardCodedClassifier()
    classifier.fit(data_train, targets_train)
    targets_predicted = classifier.predict(data_test)

    print(targets_predicted)

if __name__== "__main__":
  main()

