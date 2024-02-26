from mlc.nnir import (Conv2D, ReLU, BatchNorm2D, MaxPool2D,
        Flatten, Linear, Sequential)
from mlc.hlir import Operation

def test_op():
    o = Operation("matmul", 2, [100, 200])
    print(o)

def test_linear():
    layers = Sequential([
        Linear(256, 2048, bias=False),
        Linear(2048, 768, bias=False)
        ])
    print(layers)

def test_beautiful_mnist():
    layers = Sequential([
      Conv2D(1, 32, 5, bias=False), ReLU(),
      Conv2D(32, 32, 5, bias=False), ReLU(),
      BatchNorm2D(32), MaxPool2D((2,2)),
      Conv2D(32, 64, 3, bias=False), ReLU(),
      Conv2D(64, 64, 3, bias=False), ReLU(),
      BatchNorm2D(64), MaxPool2D((2,2)),
      Flatten(1, -1), Linear(576, 10, bias=False)])
    print(layers)
