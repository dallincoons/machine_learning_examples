class KdTree():
    k = 3
    i = 0
    def __init__(self, dataset, distance):
        self.distance = distance
        self.dataset = dataset
        self.structure = self.build(dataset, 0)

    def build(self, dataset, depth = 0):
        n = len(dataset)

        if n <= 0:
            return None

        # depth will increment as we build the tree
        axis = depth % self.k

        sorted_points = sorted(dataset, key=lambda point: point[axis])

        return {
            'point': sorted_points[n // 2],
            'left': KdTree.build(self, sorted_points[: n // 2], depth + 1),
            'right': KdTree.build(self, sorted_points[n//2 + 1:], depth + 1)
        }

    def closer_distance(self, pivot, p1, p2):
        if p1 is None:
            return p2
        if p2 is None:
            return p1

        d1 = self.distance.calculateDistance(pivot, p1)
        d2 = self.distance.calculateDistance(pivot, p2)

        if d1 < d2:
            return p1
        else:
            return p2

    def closest_point(self, root, point, depth = 0, best=None):
        if root is None:
            return None
        axis = depth % self.k

        if point[axis] < root['point'][axis]:
            next_branch = root['left']
            opposite_branch = root['right']
        else:
            next_branch = root['right']
            opposite_branch = root['left']

        best = self.closer_distance(point, self.closest_point(next_branch, point, depth+1), root['point'])

        if self.distance.calculateDistance(point, best) > abs(point[axis] - root['point'][axis]):
            best = self.closer_distance(point, self.closest_point(opposite_branch, point, depth + 1), best)
        return best




