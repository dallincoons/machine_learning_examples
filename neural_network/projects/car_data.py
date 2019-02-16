import pandas as pd

class CarData():
    def __init__(self, dataset):
        self.dataset = dataset

    def transform(self):
        print(self.dataset['class'])
