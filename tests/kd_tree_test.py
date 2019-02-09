from DataStructures.kd_tree import KdTree
from distance_strategies.euclidien_distance import EuclidienDistance

dataset = [
    [5,6,7,8],
    [3,4,5,6],
    [4,5,6,7],
    [2,3,4,5],
    [6,7,8,9],
    [1,2,3,4],
    [1,2,3,7],
]

def test_kd_tree():
    kdtree = KdTree(dataset, EuclidienDistance())

    assert([1,2,3,4] == kdtree.closest_point(kdtree.structure, [1,2,3,5]))

