from DistanceStrategies.entropy import Entropy

class DecisionTree:
    def __init__(self, columns, classes):
        self.columns = columns
        self.classes = classes
        self.possibleClasses = classes[0].unique()

    @staticmethod
    def build(columns, classes):
        tree = DecisionTree(columns, classes)
        # attribute_with_lowest_entropy = tree.find_attribute_with_lowest_entropy()
        return tree

    def find_attribute_with_lowest_entropy(self):
        columns = self.columns
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

