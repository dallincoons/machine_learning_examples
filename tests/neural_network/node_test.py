from neural_network.node import Node

def test_correct_node_output():
    node = Node(2)
    node.inputs = [.52, .49]
    node.weights = [.2, -0.1, .3]

    node.calculateOutput()

    assert(-0.1381528160171783 == node.calculateError(1))

def test_updates_weights():
    node = Node(2)
    node.inputs = [1, -.2, -1]
    node.weights = [.3, .4, .2]
    node.error = -.013

    node.updateWeights(.1)

    assert([0.3013, 0.39974000000000004, 0.19870000000000002] == node.weights)
