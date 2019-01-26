import pandas as pd
from sklearn.model_selection import train_test_split
from prediction import *
from classification import *
from sklearn.neighbors import KNeighborsRegressor

def main():
    names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']

    data = pd.read_csv('auto-mpg.txt', names=names, delim_whitespace=True, na_values=["?"])

    #fill missing data with the column mean
    data.horsepower = data.horsepower.fillna(int(data.horsepower.mean()))

    target = np.array(data['mpg'])

    data = data.drop(columns=['car name', 'mpg'])

    prepped_data = data

    # I think the discrete values such as this will work as-is
    prepped_data['model year'].unique()
    prepped_data['cylinders'].unique()

    data_train, data_test, targets_train, target_test = train_test_split(
        prepped_data.values,
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
