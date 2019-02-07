from sklearn import datasets
from sklearn.model_selection import train_test_split
from prediction import *
from classification import *
import pandas as pd

def main():
    classificationKey = input('Select a classification algorithm: \n'
      + GAUSSIAN_CLASSIFIER + ') GaussianNB \n'
      + HARD_CODED_CLASSIFIER + ') Hard coded \n'
      + KNN_CLASSIFIER + ') KNN Classifier  \n'
      + DECISION_TREE_CLASSIFIER + ') Decision Tree Classifier \n',
    )

    #ugly hack for now to load a different (discrete) dataset for decision trees
    if classificationKey == DECISION_TREE_CLASSIFIER:
        dataset = pd.read_csv('./datasets/lens.csv', delim_whitespace=True)
    else:
        dataset = datasets.load_iris()

    data_train, data_test, targets_train, target_test = train_test_split(
        dataset.data,
        dataset.target,
        train_size=.70,
        test_size=.30,
        stratify=dataset.target,
        shuffle=True
    )

    prediction = Prediction(data_train, targets_train, data_test, target_test)

    accuracy = prediction.runWith(Classification.get(classificationKey))

    print('The accuracy was: ' + str(accuracy))


if __name__== "__main__":
  main()

