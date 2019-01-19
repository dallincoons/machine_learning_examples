from math import sqrt
import numpy as np

class EuclidienDistance():
    def calculateDistance(self, answer, item):
        """
        As a micro optimization, we're avoiding using square root, which
        is what you normally do with the Euclidien Distance formula, but has
        no effect on the algorithm results
        """
        return (sum((np.array(answer) - np.array(item)) ** 2))
