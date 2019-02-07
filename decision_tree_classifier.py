from DataStructures.decision_tree import DecisionTree
import numpy as np

class DecisionTreeClassifier:
    def fit(self, data, targets):
        self.tree = DecisionTree.build(data, targets, ['credit_score', 'income', 'collatoral'])

    def predict(self, data):
        decisions = []
        for point in data:
            decisions.append(self.tree.decide(point))
        return np.array(decisions)
