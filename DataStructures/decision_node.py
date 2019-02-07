import random

class DecisionNode:
    def decide(self):
        return bool(random.getrandbits(1))
