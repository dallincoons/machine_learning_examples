from math import sqrt

class EuclidienDistance():
    def calculateDistance(self, answer, item):
        return sqrt((answer[0] - item[0])**2 + (answer[1] - item[1])**2 + (answer[2] - item[2])**2 + (answer[3] - item[3])**2)
