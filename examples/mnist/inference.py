print("Importing Python packages...")
import random
import numpy as np
from tensorflow import keras
import tensorflow as tf
from gguf.gguf_reader import GGUFReader
import torch
print("    Done.")


N_iter = 1
N_test = 10000


def load_test_data():
    _, (x, y) = keras.datasets.mnist.load_data()
    x = x.astype("float32") / 255
    # Shapes:
    # x (10000, 28, 28)
    # y (10000,)
    return x, y

def draw_digit(A):
    # (28, 28)
    assert A.shape == (28, 28)
    shades = " ░▒▓█"
    print("╔" + "══"*28 + "╗")
    for row in range(28):
        print("║", end="")
        for col in range(28):
            v = A[row,col]
            if v > 0.99:
                print(shades[4]*2, end="")
            elif v > 0.75:
                print(shades[3]*2, end="")
            elif v > 0.50:
                print(shades[2]*2, end="")
            elif v > 0.25:
                print(shades[1]*2, end="")
            else:
                print(shades[0]*2, end="")
        print("║")
    print("╚" + "══"*28 + "╝")

def gguf_to_array(g, expected_name):
    if g.name != expected_name:
        raise Exception("Expected array name `%s`, got `%s`" % \
                (expected_name, g.name))
    # The GGUF format stores the shape in reversed order
    return np.reshape(g.data, np.flip(g.shape))

def run_model(inp, kernel1, bias1, kernel2, bias2, dense_w, dense_b):
    tf_model = keras.models.load_model("mnist-cnn-model")
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential (
                torch.nn.Conv2d(1, 32, 3, bias=True),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),
                torch.nn.Conv2d(32, 64, 3, bias=True),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),
                torch.nn.Flatten(0, -1),
                torch.nn.Linear(1600, 10, bias=True),
                torch.nn.Softmax(dim=0),
                )

            kernel1_ = np.transpose(kernel1, (3,2,0,1))
            self.model[0].weight = torch.nn.Parameter(torch.from_numpy(
                    kernel1_.copy()))
            self.model[0].bias = torch.nn.Parameter(torch.from_numpy(
                    bias1.copy()))
            kernel2_ = np.transpose(kernel2, (3,2,0,1))
            self.model[3].weight = torch.nn.Parameter(torch.from_numpy(
                    kernel2_.copy()))
            self.model[3].bias = torch.nn.Parameter(torch.from_numpy(
                    bias2.copy()))
            dense_w_ = np.reshape(dense_w, (5, 5, 64, 10))
            dense_w_ = np.transpose(dense_w_, (3, 2, 0, 1))
            dense_w_ = np.reshape(dense_w_, (10, 1600))
            self.model[7].weight = torch.nn.Parameter(torch.from_numpy(
                    dense_w_.copy()))
            self.model[7].bias = torch.nn.Parameter(torch.from_numpy(
                    dense_b.copy()))

        def forward(self, x):
            return self.model(x)

    print("Input shape:", inp.shape)
    assert inp.shape == (28, 28)
    model = Model()
    inp_ = np.expand_dims(inp, 0)
    torch_inp = torch.tensor(inp_)
    torch_out = model(torch_inp)
    out = torch_out.detach().numpy()
    print("Output shape:", out.shape)
    assert out.shape == (10,)
    print("PT:", out)
    print("PT max:", out.argmax())

    out_tf = tf_model(np.expand_dims(inp, 0))
    print("TF:", out_tf)
    print("TF max:", out_tf.numpy().argmax())

    return out

def run_model_np(inp, kernel1, bias1, kernel2, bias2, dense_w, dense_b):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential (
                torch.nn.Conv2d(1, 32, 3, bias=True),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),
                torch.nn.Conv2d(32, 64, 3, bias=True),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),
                torch.nn.Flatten(0, -1),
                torch.nn.Linear(1600, 10, bias=True),
                torch.nn.Softmax(dim=0),
                )

            kernel1_ = np.transpose(kernel1, (3,2,0,1))
            self.model[0].weight = torch.nn.Parameter(torch.from_numpy(
                    kernel1_.copy()))
            self.model[0].bias = torch.nn.Parameter(torch.from_numpy(
                    bias1.copy()))
            kernel2_ = np.transpose(kernel2, (3,2,0,1))
            self.model[3].weight = torch.nn.Parameter(torch.from_numpy(
                    kernel2_.copy()))
            self.model[3].bias = torch.nn.Parameter(torch.from_numpy(
                    bias2.copy()))
            dense_w_ = np.reshape(dense_w, (5, 5, 64, 10))
            dense_w_ = np.transpose(dense_w_, (3, 2, 0, 1))
            dense_w_ = np.reshape(dense_w_, (10, 1600))
            self.model[7].weight = torch.nn.Parameter(torch.from_numpy(
                    dense_w_.copy()))
            self.model[7].bias = torch.nn.Parameter(torch.from_numpy(
                    dense_b.copy()))

        def forward(self, x):
            return self.model(x)

    print("Input shape:", inp.shape)
    assert inp.shape == (28, 28)
    model = Model()
    inp_ = np.expand_dims(inp, 0)
    torch_inp = torch.tensor(inp_)
    torch_out = model(torch_inp)
    out = torch_out.detach().numpy()
    print("Output shape:", out.shape)
    assert out.shape == (10,)
    print("PT:", out)
    print("PT max:", out.argmax())

    return out


def main():
    print("Loading MNIST test images...")
    x_test, y_test = load_test_data()
    print("    Done.")

    print("Loading MNIST model GGUF...")
    g = GGUFReader("mnist-cnn-model.gguf")
    kernel1 = gguf_to_array(g.tensors[0], "kernel1")
    bias1 = gguf_to_array(g.tensors[1], "bias1")
    kernel2 = gguf_to_array(g.tensors[2], "kernel2")
    bias2 = gguf_to_array(g.tensors[3], "bias2")
    dense_w = gguf_to_array(g.tensors[4], "dense_w")
    dense_b = gguf_to_array(g.tensors[5], "dense_b")
    print("    Done.")

    for iter in range(N_iter):
        i = random.randint(0, N_test)
        print("Input digit index:", i)
        inp = x_test[i,:,:]
        draw_digit(inp)
        print("Reference value:", y_test[i])

        x = run_model(inp, kernel1, bias1, kernel2, bias2, dense_w, dense_b)
        infer_val = np.argmax(x)
        print("Inferred value:", infer_val)
        print("Digit probabilities:", x)

        print("---------")
        x = run_model_np(inp, kernel1, bias1, kernel2, bias2, dense_w, dense_b)
        infer_val = np.argmax(x)
        print("NumPy Inferred value:", infer_val)
        print("NumPy Digit probabilities:", x)


if __name__ == '__main__':
    main()
