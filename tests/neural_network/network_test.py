from neural_network.network import Network

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
    network = Network(learning_rate=.1)
    network.create(SMALL_INPUT, SMALL_INPUT_CLASS)
