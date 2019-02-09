from distance_strategies.euclidien_distance import *

def test_calculate_distance_between_multidimensional_points():
    distance = EuclidienDistance()
    result = distance.calculateDistance([1,2,3,4], [2,3,4,5])
    assert(4.0 == result)
    result = distance.calculateDistance([1,2,3,4], [3,6,7,12])
    assert(100.0 == result)
