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
    network = Network([2,4,7], 3, 2, learning_rate=.1)
    layers = network.layers

    assert(len(layers) == 3)

def test_network_creation_returns_correct_number_of_output():
    network = Network([3, 2], 3, 2, learning_rate=.1)

    output = network.calculateOutput([1,2,3])

    assert(2 == len(output))

def test_layer_has_reference_to_previous_layer():
    network = Network([3, 2], 3, 2, learning_rate=.1)

    network.calculateOutput([1,2,3])

    assert(3 == len(network.layers[1].prevLayer.nodes))

def test_is_initialized_with_correct_number_of_layers():
    network = Network([2, 2], 3, 2, learning_rate=.1)

    assert(2 == len(network.layers))
    # assert([1.2, -0.2] == network.layers[0].inputs)

def test_is_initialized_with_correct_number_of_nodes():
    network = Network([2,2], initial_num=2, output_num=2, learning_rate=.1)

    assert(2 == len(network.layers[0].nodes))

def test_nodes_have_correct_number_of_weights():
    network = Network([2,2], initial_num=2, output_num=2, learning_rate=.1)

    assert(3 == len(network.layers[1].nodes[0].weights))

def test_correct_output():
    network = build_network()

    output = network.layers[1].nodes[0].calculateOutput()
    output2 = network.layers[1].nodes[1].calculateOutput()

    assert(0.43905454967528385 == output)
    assert(0.502499979166875 == output2)

    output3 = network.layers[0].nodes[0].calculateOutput()

    assert(0.5199893401555817 == output3)

def test_correct_error():
    network = build_network()

    network.layers[1].calculateOutput()
    network.layers[0].calculateOutput()

    network.layers[1].calculateErrors([1, 0])
    error = network.layers[1].nodes[0].error
    error2 = network.layers[0].nodes[0].calculateHiddenError(network.layers[1], [1, 0])

    assert(-0.1381528160171783 == error)
    assert(-0.013167654026580511 == error2)

def build_network():
    network = Network([2,2], initial_num=2, output_num=2, learning_rate=.1)

    network.layers[1].nodes[0].weights = [.2, -0.1, .3]
    network.layers[1].nodes[0].biasless_weights = [.2, -0.1]
    network.layers[1].nodes[0].inputs = [.52, .49]

    network.layers[1].nodes[1].weights = [.4, -0.2, .1]
    network.layers[1].nodes[1].biasless_weights = [.4, -0.2]
    network.layers[1].nodes[1].inputs = [.52, .49]

    network.layers[0].nodes[0].weights = [.3, .4, .2]
    network.layers[0].nodes[0].biasless_weights = [.3, .4]
    network.layers[0].nodes[0].inputs = [1.2, -0.2]

    network.layers[0].nodes[1].weights = [.2, -0.1, .1]
    network.layers[0].nodes[1].biasless_weights = [.2, -0.1]
    network.layers[0].nodes[1].inputs = [1.2, -0.2]

    return network
