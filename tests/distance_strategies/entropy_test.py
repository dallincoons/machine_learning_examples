from DistanceStrategies.entropy import Entropy

def test_calculates_entropy():
    result = Entropy.calculate([.75, .25])
    assert(round(result, 3) == .811)

    result = Entropy.calculate([.5, .5])
    assert(round(result, 3) == 1.0)

def test_calculates_entropy_handles_zero():
    result = Entropy.calculate([0, .25])
    assert(round(result, 3) == .5)

def test_get_weighted_average():
    result = Entropy.weighted_average([
        [2, .5],
        [4, .25]
    ])

    assert(2 == result)

    result = Entropy.weighted_average([
        [12, .5],
        [8, .25]
    ])

    assert(8 == result)
