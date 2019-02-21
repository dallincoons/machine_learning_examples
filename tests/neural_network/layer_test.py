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
    result = Layer(SMALL_TEST).initialize_nodes(SMALL_TEST)
    assert(len(result) == 4)
    assert(type(result[0]) == type(Node([])))

def test_calculate_output():
    layer = Layer([2 ,4, 7], 5)

    assert(len(layer.calculateOutput()) == 5)
