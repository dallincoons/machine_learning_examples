from neural_network.network import Network
from neural_network.layer import Layer
import inspect

SMALL_INPUT = [
    [1, 0, 1],
    [0, -1, 0],
    [0, -.8, 1],
    [1, -.2, 0],
]

SMALL_INPUT_CLASS = [
    1,
    0,
    1,
    0
]

def test_build_network():
    network = Network([2,4,7], learning_rate=.1)
    network.create(SMALL_INPUT, SMALL_INPUT_CLASS)
    layers = network.layers

    assert(len(layers) == 3)
    assert(len(layers[0].nodes) == 2)
    assert(len(layers[1].nodes) == 4)
    assert(len(layers[2].nodes) == 7)
