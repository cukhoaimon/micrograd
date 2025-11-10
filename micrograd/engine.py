import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, op={self._op})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    """
        f(x) = x^n 
        f'(x) = n.x'.x^(n-1)
    """
    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Value(self.data**power, (self,), f"^{power}")

        def _backward():
            self.grad += power * out.grad * (self.data ** power - 1)

        self._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other ** -1

    def __neg__(self): # -self
        return self * (-1)

    def __sub__(self, other):
        out = self + (-other)
        out._op = '-'
        return out

    """
        L(x)  = e^x
        L'(x) = dL/dx = d(e^x)/dx = e^x

        L(1)  = e^1
        L'(1) = L(1) = e^1

        -> self = x
        -> out  = e^x
    """
    def exp(self):
        out = Value(math.exp(self.data), (self,), _op='exp')

        def _backward():
            self.grad += out.data * out.grad

        self._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
