# Test NumPy inference from GGUF, no PyTorch or TensorFlow dependency.

print("Importing Python packages...")
import numpy as np
from gguf.gguf_reader import GGUFReader
print("    Done.")

def load_test_data():
    g = GGUFReader("mnist-tests.gguf")
    x = gguf_to_array(g.tensors[0], "x_test")
    y = gguf_to_array(g.tensors[1], "y_test")
    x = x.astype(np.uint8).astype(np.float32) / 255
    assert x.shape == (10000, 28, 28)
    assert y.shape == (10000,)
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


# (10,) -> (10,)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def relu(x):
    y = x.copy()
    y[x < 0] = 0
    return y

# (channel, w, h)
def max_pool_2d(x):
    channel, w, h = x.shape
    w2 = w//2
    h2 = h//2
    r = np.empty((channel, w2, h2), dtype=x.dtype)
    for c in range(channel):
        for i in range(w2):
            for j in range(h2):
                r[c, i, j] = np.max(x[c, 2*i:2*i+2, 2*j:2*j+2])
    return r

# (h, w)
def conv2d_kernel(kernel_size, weight, x):
    assert len(weight.shape) == 2
    assert weight.shape[0] == kernel_size
    assert weight.shape[1] == kernel_size
    assert len(x.shape) == 2
    in_h, in_w = x.shape
    out_w = in_w - (kernel_size-1)
    out_h = in_h - (kernel_size-1)
    out = np.empty((out_h, out_w), dtype=x.dtype)

    for h in range(out_h):
        for w in range(out_w):
            out[h,w] = np.sum(weight*x[h:h+kernel_size,w:w+kernel_size])
    return out

# (batch, channel, h, w)
def conv2d(in_channels, out_channels, kernel_size, weight, bias, x):
    in_channels_x, in_h, in_w = x.shape
    assert in_channels == in_channels_x
    out_w = in_w - (kernel_size-1)
    out_h = in_h - (kernel_size-1)
    out = np.empty((out_channels, out_h, out_w), dtype=x.dtype)
    for c in range(out_channels):
        s = np.zeros((out_h, out_w), dtype=x.dtype)
        for k in range(in_channels):
            s += conv2d_kernel(kernel_size, weight[c,k,:,:], x[k,:,:])
        out[c, :, :] = bias[c] + s
    return out

def run_model_np(N, inp, kernel1, bias1, kernel2, bias2, dense_w, dense_b):
    assert inp.shape == (28, 28, N)
    out2 = np.empty((10,N), dtype=np.float32)
    for b in range(N):
        inp_ = np.expand_dims(inp[:,:,b], 0)
        out = inp_.copy()

        # Conv2D
        # (C_out, C_in, H, W)
        out = conv2d(1, 32, 3, kernel1, bias1, out)

        # ReLU
        out = relu(out)

        # MaxPool2D
        out = max_pool_2d(out)

        # Conv2D
        # (C_out, C_in, H, W)
        out = conv2d(32, 64, 3, kernel2, bias2, out)

        # ReLU
        out = relu(out)

        # MaxPool2D
        out = max_pool_2d(out)

        # Flatten
        out = np.reshape(out, (1600,))

        # Linear
        # (N_out, C_in*H*W)
        out = np.dot(dense_w, out) + dense_b

        # Softmax
        out = softmax(out)

        assert out.shape == (10,)
        out2[:,b] = out
    return out2


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

    N = 2
    i = 4212
    print("Input digit index:", i, N)
    inp = x_test[i:i+N,:,:]
    inp = np.transpose(inp, (1, 2, 0))
    for b in range(N):
        draw_digit(inp[:,:,b])
    ref_val = y_test[i:i+N]
    print("Reference values:", ref_val)
    x = run_model_np(N, inp, kernel1, bias1, kernel2, bias2, dense_w, dense_b)
    infer_val = np.argmax(x, axis=0)
    print("NumPy Inferred values:", infer_val)
    print("NumPy Digit probabilities:\n", x)
    assert all(infer_val == ref_val)


if __name__ == '__main__':
    main()