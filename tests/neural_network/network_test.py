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
    network = Network([2,4,7], learning_rate=.1, dataset=SMALL_INPUT[0])
    network.calculateOutput()
    layers = network.layers

    assert(len(layers) == 3)
    assert(len(layers[0].nodes) == 2)
    assert(len(layers[1].nodes) == 4)
    assert(len(layers[2].nodes) == 7)

def test_network_creation_returns_correct_number_of_output():
    network = Network([3, 2], learning_rate=.1, dataset=SMALL_INPUT[1])

    output = network.calculateOutput()

    assert(2 == len(output))

def test_layer_has_reference_to_previous_layer():
    network = Network([3, 2], learning_rate=.1, dataset=[])

    network.calculateOutput()

    assert(3 == len(network.layers[1].prevLayer.nodes))

def test_correct_output():
    network = Network([2,2], learning_rate=.1, dataset=[1.2, -.2])

    output = network.calculateOutput()

    print(output)
