from decision_tree.tree import Tree
from decision_tree.attribute_entropy import AttributeEntropy
import numpy as np

class DecisionTreeBuilder:
    def create(self, attributes, classes, featureNames):
        best_attribute = AttributeEntropy(attributes, classes).lowest_attributes()
        best_attribute_unique_values = self.unique_values_in_col(attributes, best_attribute)

        tree = {featureNames[best_attribute]: {}}

        for attribute_category in best_attribute_unique_values:
            matching_rows, matching_classes = self.rows_matching_attribute_value(list(zip(attributes, classes)), best_attribute, attribute_category)
            if self.attribute_is_pure(matching_classes):
                tree[featureNames[best_attribute]][attribute_category] = list(set(matching_classes))[0]
            else:
                newAttributes = self.remove_col(matching_rows, best_attribute)
                newFeatureNames = featureNames[:]
                del newFeatureNames[best_attribute]
                tree[featureNames[best_attribute]][attribute_category] = self.create(newAttributes, matching_classes, newFeatureNames)
        return tree

    def rows_matching_attribute_value(self, attributes, best_attribute, attribute_category):
        return [attribute[0] for attribute in attributes if attribute[0][best_attribute] == attribute_category], [attribute[1] for attribute in attributes if attribute[0][best_attribute] == attribute_category]

    def unique_values_in_col(self, arr, col):
        return list(set(self.get_col(arr, col)))

    def get_col(self, arr, col):
        return list(map(lambda x : x[col], arr))

    def remove_col(self, arr, col):
        for row in arr:
            del row[col]
        return arr

    def attribute_is_pure(self, classes):
        return len(set(classes)) == 1

    # def generateDecisions(self, attributes, column, categories):
    #     for category in categories:
    #         self.generateDecision([attribute for attribute in attributes if attribute[column] == category])
    #
    # def generateDecision(self, attributes, category):
    #     print(attributes)
