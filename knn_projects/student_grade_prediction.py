import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

def main():

    data = pd.read_csv('student-mat.csv', sep=';')

    #convert values to integers
    data["sex"] = data.sex.map({"M": 1, "F": 0})
    data["address"] = data.address.map({"U": 1, "R": 0})
    data["famsize"] = data.famsize.map({"GT3": 1, "LE3": 0})
    data["Pstatus"] = data.Pstatus.map({"T": 1, "A": 0})
    data["schoolsup"] = data.schoolsup.map({"yes": 1, "no": 0})
    data["famsup"] = data.famsup.map({"yes": 1, "no": 0})
    data["paid"] = data.paid.map({"yes": 1, "no": 0})
    data["activities"] = data.activities.map({"yes": 1, "no": 0})
    data["nursery"] = data.nursery.map({"yes": 1, "no": 0})
    data["higher"] = data.higher.map({"yes": 1, "no": 0})
    data["internet"] = data.internet.map({"yes": 1, "no": 0})
    data["romantic"] = data.romantic.map({"yes": 1, "no": 0})

    #one-hot encode the rest
    data = pd.get_dummies(data, columns=["Mjob"])
    data = pd.get_dummies(data, columns=["Fjob"])
    data = pd.get_dummies(data, columns=["school"])
    data = pd.get_dummies(data, columns=["guardian"])

    target = np.array(data['G3'])

    #we won't need reason or the grades for trainging
    data = data.drop(columns=['G3', 'G2', 'G1', 'reason'])

    data_train, data_test, targets_train, target_test = train_test_split(
        data.values,
        target,
        train_size=.70,
        test_size=.30,
        shuffle=True
    )

    regr = KNeighborsRegressor(n_neighbors=3)
    regr.fit(data_train, targets_train)
    predictions = regr.predict(data_test)

    innacurateResults = [
        predictions[i] for i in range(len(predictions.tolist()))
        if int(predictions[i]) > (int(target_test[i]) + 1) | int(predictions[i] < (int(target_test[i]) - 1) )
    ]
    print('accuracy is: ' + str(1 - len(innacurateResults) / len(data_train)))

if __name__ == '__main__':
    main()
