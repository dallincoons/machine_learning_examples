from neural_network.activation_function import ActivationFunction

def test_sigmoid_activation_function():
    assert(.5 == ActivationFunction.run(0))
    assert(0.7310585786300049 == ActivationFunction.run(1))
    assert(0.2689414213699951== ActivationFunction.run(-1))
