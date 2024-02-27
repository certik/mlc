from mlc.nnir import (Conv2D, ReLU, BatchNorm2D, MaxPool2D,
        Flatten, Linear, Sequential,
        # Transpose,
        Softmax)
from mlc.hlir import Operation, Array
from mlc.nn_to_hl import nn_to_hl

def test_op():
    A = Array("A", 2, (100, 5))
    B = Array("B", 2, (5, 200))
    o = Operation("matmul", 2, (100, 200), (A, B))
    print(o)

def test_linear():
    layers = Sequential([
        Linear(256, 2048, bias=False),
        Linear(2048, 768, bias=False)
        ])
    print(layers)
    hl = nn_to_hl(layers, in_shape=(100,256))
    print(hl)
    assert hl.shape == (100, 768)

def test_ggml_mnist_cnn():
    # input: (28, 28)
    layers = Sequential([
      Conv2D(1, 32, 3, bias=True), # (26, 26, 32)
      ReLU(), # (26, 26, 32)
      MaxPool2D((2,2)), # (13 13 32)
      Conv2D(32, 64, 3, bias=True), # (11 11 64)
      ReLU(), # (11 11 64)
      MaxPool2D((2,2)), # (5 5 64)
      #Transpose((1, 2, 0)), # (64, 5, 5)
      Flatten(1, -1), # (1600)
      Linear(1600, 10, bias=True), # (10)
      Softmax(),
      ])
    print(layers)

    hl = nn_to_hl(layers, in_shape=(28,28))
    print(hl)
    assert hl.shape == (10,)

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

    hl = nn_to_hl(layers, in_shape=(28,28))
    print(hl)
    assert hl.shape == (10,)
