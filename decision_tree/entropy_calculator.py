import math

class EntropyCalculator:
    @staticmethod
    def calculate(values):
        return sum(map(lambda value: 0 if value == 0 else -value * math.log(value, 2), values))

    @staticmethod
    def weighted_average(weights):
        return sum(map(lambda weight: weight[0] * weight[1], weights))
