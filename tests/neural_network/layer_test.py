from neural_network.layer import Layer
from neural_network.node import Node

SMALL_INPUT = [
    [1, 0, 1],
    [0, -1, 0],
    [0, -.8, 1],
    [1, -.2, 0],
]

SMALL_INPUT_CLASS = [[
    1,
    0,
    1,
    0
]]

SMALL_TEST = [
    0, 1, 1
]

SMALL_TEST_CLASS = [
    1
]

def test_initialize_nodes():
    layer = Layer(4)
    layer.setPreviousLayer(Layer(4))
    layer.initialize_nodes()
    result = layer.nodes
    assert(len(result) == 4)
    assert(type(result[0]) == type(Node(0)))

def test_calculate_output():
    layer = Layer(5)
    layer.setPreviousLayer(Layer(3))
    layer.setInput([1,2,3])

    assert(len(layer.calculateOutput()) == 5)
