from neural_network.neural_network_builder import NeuralNetworkBuilder

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

def test_heyo():
    result = NeuralNetworkBuilder(learning_rate=1).create(SMALL_INPUT, SMALL_INPUT_CLASS)
    print(result)
