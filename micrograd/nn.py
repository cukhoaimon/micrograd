import random

from micrograd.engine import Value


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # total wx + b
        act: Value = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


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

        return x

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
