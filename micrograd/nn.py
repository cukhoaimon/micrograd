import random

from micrograd.engine import Value


class Neuron:
    def __init__(self, nin, b=0):
        self.w = [random.uniform(-1, 1) for _ in range(nin)]
        self.b = Value(b)

    def __call__(self, x):
        # total wx + b
        act: Value = sum([xi * wi for xi, wi in zip(self.w, x)]) + self.b
        return act.tanh()


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]


"""
    Multi layer perceptron
"""


class MLP:
    """
        nouts = [outputFirstLayer, outputSecondLayer, .... outputLastLayer]
    """

    def __init__(self, nin, nouts):
        # total layer including first
        layer_def = [nin] + nouts
        self.layers = [Layer(layer_def[i], layer_def[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x[0] if len(x) == 1 else x

