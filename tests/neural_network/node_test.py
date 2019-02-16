from neural_network.node import Node

def test_determines_wether_neuron_fires():
    neuron = Node(1, -1, 0)

    neuron.addInput(1, -.3)
    neuron.addInput(2, .2)
    neuron.addInput(3, -.4)
    neuron.addInput(4, .4)

    neuron.bias_weight = .5

    assert(neuron.calculateOutput() == 0)
