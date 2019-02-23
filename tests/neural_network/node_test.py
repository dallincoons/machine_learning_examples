from neural_network.node import Node

def test_determines_whether_neuron_fires():
    node = Node([0, 0, 0], 0)

    assert(node.calculateOutput() == .5)
