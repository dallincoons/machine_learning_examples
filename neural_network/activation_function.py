import math
import numpy as np

class ActivationFunction():
    @staticmethod
    def run(x):
        return ActivationFunction.sigmoid(x)

    @staticmethod
    def sigmoid(x):
        return math.exp(-np.logaddexp(0, -x))

