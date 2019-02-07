import math
import pandas as pd
from DistanceStrategies.entropy import Entropy
from decision_tree.entropy_calculator import EntropyCalculator as Entropy

class AttributeEntropy:
    def __init__(self, attributes, classes):
        self.attributes = pd.DataFrame(attributes)
        self.classes = pd.DataFrame(classes)
        self.possibleClasses = self.classes[0].unique()

    def lowest_attributes(self):
        columns = self.attributes
        classes = self.classes
        entropies = {}

        for col_index in range(0, columns.shape[1]):
            columnsToSplit = columns[col_index].unique()
            weights = []
            for columnToSplit in columnsToSplit:
                micro_entropies = []
                matchingColumnindexes = columns[columns[col_index] == columnToSplit].index.tolist()
                classifications = classes[classes.index.isin(matchingColumnindexes)]
                class_percentages = self.class_percentages(classifications)
                micro_entropies.append(Entropy.calculate(class_percentages))
                weights.append([Entropy.calculate(class_percentages), len(classifications) / len(columns)])
            entropies[col_index] = Entropy.weighted_average(weights)
        return min(entropies, key=entropies.get)

    def class_percentages(self, classes):
        counts = classes[0].value_counts()
        collection = []

        for possible in self.possibleClasses:
            classCount = 0
            if possible in counts:
                classCount = int(counts[possible])
            collection.append(classCount / len(classes))
        return collection

    def is_leaf(self, classes):
        return len(set(classes)) == 1
