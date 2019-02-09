from distance_strategies.entropy import Entropy
from DataStructures.decision_node import DecisionNode
import pandas as pd

class DecisionTree:
    tree = {}

    def __init__(self, columns, classes):
        self.columns = pd.DataFrame(columns)
        self.classes = pd.DataFrame(classes)
        self.possibleClasses = self.classes[0].unique()

    @staticmethod
    def build(columns, classes, featureNames):
        tree = DecisionTree(columns, classes)
        tree.make(columns, classes, featureNames)
        return tree

    def make(self, columns, classes, featureNames):
        if (len(columns) == 0 or len(featureNames) == 0):
            return 'y'
        elif classes.values.tolist().count('n') == len(columns):
            return 'n'

        newColumns = []
        newClasses = []
        newFeatureNames = []
        gain = []

        classes = classes.values.tolist()

        bestFeature = self.find_attribute_with_lowest_entropy()
        bestFeatureName = featureNames[bestFeature]

        self.tree = {bestFeatureName:{}}

        values = columns[bestFeature].unique()
        for value in values:
            index = 0
            for datapoint in columns.values:
                datapoint = datapoint.tolist()
                if datapoint[bestFeature] == value:
                    if bestFeature == 0:
                        datapoint = datapoint[1:]
                        newFeatureNames = featureNames[1:]
                    elif bestFeature == len(featureNames):
                        datapoint = datapoint[:-1]
                        newFeatureNames = featureNames[:-1]
                    else:
                        datapoint = datapoint[:bestFeature]
                        datapoint.extend(datapoint[bestFeature+1:])
                        print(featureNames[bestFeature:])
                        # newFeatureNames = featureNames[:bestFeature]
                        # newFeatureNames.extend(featureNames[bestFeature+1:])
                    newColumns.append(datapoint)
                    newClasses.append(classes[index])
                index = index + 1

            subtree = self.make(pd.DataFrame(newColumns), pd.DataFrame(newClasses), newFeatureNames)

            self.tree[bestFeatureName][value] = subtree
        return self.tree
        # for datapoint in columns.values:
        #     print(datapoint[bestFeature], value)
            # if datapoint[bestFeature] == value:

    def decide(self, columns):
        self.root_node.decide()

    def addRootNode(self):
        attribute_with_lowest_entropy = self.find_attribute_with_lowest_entropy()
        self.root_node = DecisionNode(1)

    def make_tree(data,classes,featureNames):
        nData = len(data)
        nFeature = len(featureNames)
        newData = []
        newClasses = []
        newNames = []
        index = 0

        default = classes[np.argmax(frequency)]
        if nData==0 or nFeatures == 0:
            # Have reached an empty branch
            return default
        elif classes.count(classes[0]) == nData:
            # Only 1 class remains
            return classes[0]
        else:
            # Choose which feature is best
            gain = np.zeros(nFeatures)
            for feature in range(nFeatures):
                g = calc_info_gain(data,classes,feature)
                gain[feature] = totalEntropy - g
            bestFeature = np.argmax(gain)
            tree = {featureNames[bestFeature]:{}} # Find the possible feature values
            for value in values:
                # Find the datapoints with each feature value
                for datapoint in data:
                    if datapoint[bestFeature]==value:
                        if bestFeature==0:
                            datapoint = datapoint[1:]
                            newNames = featureNames[1:]
                        elif bestFeature==nFeatures:
                            datapoint = datapoint[:-1]
                            newNames = featureNames[:-1]
                        else:
                            datapoint = datapoint[:bestFeature]
                            datapoint.extend(datapoint[bestFeature+1:])
                            newNames = featureNames[:bestFeature]
                            newNames.extend(featureNames[bestFeature+1:])
                        newData.append(datapoint)
                        newClasses.append(classes[index])
                    index += 1
                # Now recurse to the next level
                subtree = make_tree(newData,newClasses,newNames) # And on returning, add the subtree on to the tree
                tree[featureNames[bestFeature]][value] = subtree
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

