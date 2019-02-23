from sklearn import datasets
from sklearn.model_selection import train_test_split
from prediction import *
from classification import *
import pandas as pd

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

    prediction = Prediction(data_train, targets_train, data_test, target_test)

    classificationKey = input('Select a classification algorithm: \n'
         + GAUSSIAN_CLASSIFIER + ') GaussianNB \n'
         + HARD_CODED_CLASSIFIER + ') Hard coded \n'
         + KNN_CLASSIFIER + ') KNN Classifier  \n'
         + NEURAL_NET_CLASSIFIER + ') Neural Net Classifier  \n'
    )

    accuracy = prediction.runWith(Classification.get(classificationKey))

    print('The accuracy was: ' + str(accuracy))


if __name__== "__main__":
  main()

