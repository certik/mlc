print("Importing Python packages...")
import random
import numpy as np
from tensorflow import keras
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
    return np.reshape(g.data, np.flip(g.shape))

def run_model(inp, kernel1, bias1, kernel2, bias2, dense_w, dense_b):
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

            kernel1_ = np.transpose(kernel1, (3,2,1,0))
            kernel1_ = np.transpose(kernel1_, (3, 2, 0, 1)).copy()
            self.model[0].weight = torch.nn.Parameter(torch.from_numpy(
                    kernel1_).float())

            # The bias is duplicated in the file
            bias1_ = np.transpose(bias1, (3,2,1,0))
            bias1_ = bias1_[0,0,:,0].copy()
            self.model[0].bias = torch.nn.Parameter(torch.from_numpy(
                    bias1_).float())

            kernel2_ = np.transpose(kernel2, (3,2,1,0))
            kernel2_ = np.transpose(kernel2_, (3, 2, 0, 1)).copy()
            self.model[3].weight = torch.nn.Parameter(torch.from_numpy(
                    kernel2_).float())

            # The bias is duplicated in the file
            bias2_ = np.transpose(bias2, (3,2,1,0))
            bias2_ = bias2_[0,0,:,0].copy()
            self.model[3].bias = torch.nn.Parameter(torch.from_numpy(
                    bias2_).float())

            dense_w_ = np.transpose(dense_w, (1,0))
            dense_w_ = np.transpose(dense_w_, (1, 0)).copy()
            self.model[7].weight = torch.nn.Parameter(torch.from_numpy(
                    dense_w_).float())

            self.model[7].bias = torch.nn.Parameter(torch.from_numpy(
                    dense_b.copy()).float())

        def forward(self, x):
            return self.model(x)

    print("Input shape:", inp.shape)
    assert inp.shape == (28, 28)
    model = Model()
    inp = np.expand_dims(inp, 0)
    torch_inp = torch.tensor(inp)
    torch_out = model(torch_inp)
    out = torch_out.detach().numpy()
    print("Output shape:", out.shape)
    assert out.shape == (10,)
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


if __name__ == '__main__':
    main()
