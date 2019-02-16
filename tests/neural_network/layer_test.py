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

def setup_class():
    print('test')

def test_heyo():
    result = Layer(learning_rate=1).create(SMALL_INPUT, SMALL_INPUT_CLASS)
    print(result)

def test_initialize_nodes():
    result = Layer(learning_rate=1, num_nodes=4).initialize_nodes(4)
    assert(len(result) == 4)
    assert(type(result[0]) == type(Node(1)))
