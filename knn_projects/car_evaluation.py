import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from prediction import *
from classification import *

def main():
    """
    CAR                      car acceptability
   . PRICE                  overall price
   . . buying               buying price
   . . maint                price of the maintenance
   . TECH                   technical characteristics
   . . COMFORT              comfort
   . . . doors              number of doors
   . . . persons            capacity in terms of persons to carry
   . . . lug_boot           the size of luggage boot
   . . safety               estimated safety of the car
    """
    names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    #create an array for data prepared to enter training

    #looks like not missing data
    data = pandas.read_csv('car.csv', names=names)
    target = np.array(data['class'])
    prepped_data = pandas.DataFrame()

    prepped_data['buying_cat'] = data.buying.astype('category').cat.codes
    prepped_data['maint_cat'] = data.maint.astype('category').cat.codes
    prepped_data['safety_cat'] = data.safety.astype('category').cat.codes
    prepped_data['lug_boot_cat'] = data.lug_boot.astype('category').cat.codes

    data_train, data_test, targets_train, target_test = train_test_split(
        prepped_data.values,
        target,
        train_size=.70,
        test_size=.30,
        stratify=target,
        shuffle=True
    )

    prediction = Prediction(data_train, targets_train, data_test, target_test)

    accuracy = prediction.runWith(Classification.get(KNN_CLASSIFIER))

    print('The accuracy was: ' + str(accuracy))

if __name__ == '__main__':
    main()
