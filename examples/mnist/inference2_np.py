# Test NumPy inference from GGUF, no PyTorch or TensorFlow dependency.

print("Importing Python packages...")
import numpy as np
from gguf.gguf_reader import GGUFReader
print("    Done.")


N_iter = 1
N_test = 10000


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

def batch_norm_2d(in_channels,
        gamma, beta,
        moving_mean, moving_variance,
        x, eps, momentum):
    assert len(x.shape) == 3
    C, W, H = x.shape
    assert gamma.shape == (C,)
    assert beta.shape == (C,)
    assert moving_mean.shape == (C,)
    assert moving_variance.shape == (C,)
    y = np.empty((in_channels, W, H), dtype=x.dtype)
    for c in range(C):
        y[c,:,:] = ((x[c,:,:] - moving_mean[c]) \
                / np.sqrt(moving_variance[c] + eps)) * gamma[c] + beta[c]
    return y

def run_model_np(inp,
        kernel1, bias1, kernel2, bias2, kernel3, bias3, kernel4, bias4,
        batchnorm1_gamma, batchnorm1_beta,
        batchnorm1_moving_mean, batchnorm1_moving_variance,
        batchnorm2_gamma, batchnorm2_beta,
        batchnorm2_moving_mean, batchnorm2_moving_variance,
        dense_w, dense_b):
    #print("Input shape:", inp.shape)
    assert inp.shape == (28, 28)
    inp_ = np.expand_dims(inp, 0)
    out = inp_.copy()

    # Conv2D
    # (C_out, C_in, H, W)
    out = conv2d(1, 32, 5, kernel1, bias1, out)
    # ReLU
    out = relu(out)
    # (C_out, C_in, H, W)
    out = conv2d(32, 32, 5, kernel2, bias2, out)
    # ReLU
    out = relu(out)
    # BatchNorm2D
    out = batch_norm_2d(32,
            batchnorm1_gamma, batchnorm1_beta,
            batchnorm1_moving_mean, batchnorm1_moving_variance,
            out, eps=0.001, momentum=0.01)
    # MaxPool2D
    out = max_pool_2d(out)

    # Conv2D
    # (C_out, C_in, H, W)
    out = conv2d(32, 64, 3, kernel3, bias3, out)
    # ReLU
    out = relu(out)
    # (C_out, C_in, H, W)
    out = conv2d(64, 64, 3, kernel4, bias4, out)
    # ReLU
    out = relu(out)
    # BatchNorm2D
    out = batch_norm_2d(64,
            batchnorm2_gamma, batchnorm2_beta,
            batchnorm2_moving_mean, batchnorm2_moving_variance,
            out, eps=0.001, momentum=0.01)
    # MaxPool2D
    out = max_pool_2d(out)

    # Flatten
    out = np.reshape(out, (576,))
    # Linear
    # (N_out, C_in*H*W)
    out = np.dot(dense_w, out) + dense_b
    # Softmax
    out = softmax(out)

    #print("Output shape:", out.shape)
    assert out.shape == (10,)
    #print("NumPy:", out)
    #print("NumPy max:", out.argmax())

    return out


def main():
    print("Loading MNIST test images...")
    x_test, y_test = load_test_data()
    print("    Done.")

    print("Loading MNIST model GGUF...")
    g = GGUFReader("mnist-cnn-beautiful-model.gguf")
    kernel1 = gguf_to_array(g.tensors[0], "kernel1")
    bias1 = gguf_to_array(g.tensors[1], "bias1")
    kernel2 = gguf_to_array(g.tensors[2], "kernel2")
    bias2 = gguf_to_array(g.tensors[3], "bias2")
    batchnorm1_gamma = gguf_to_array(g.tensors[4], "batchnorm1_gamma")
    batchnorm1_beta = gguf_to_array(g.tensors[5], "batchnorm1_beta")
    batchnorm1_moving_mean = gguf_to_array(g.tensors[6],
            "batchnorm1_moving_mean")
    batchnorm1_moving_variance = gguf_to_array(g.tensors[7],
            "batchnorm1_moving_variance")
    kernel3 = gguf_to_array(g.tensors[8], "kernel3")
    bias3 = gguf_to_array(g.tensors[9], "bias3")
    kernel4 = gguf_to_array(g.tensors[10], "kernel4")
    bias4 = gguf_to_array(g.tensors[11], "bias4")
    batchnorm2_gamma = gguf_to_array(g.tensors[12], "batchnorm2_gamma")
    batchnorm2_beta = gguf_to_array(g.tensors[13], "batchnorm2_beta")
    batchnorm2_moving_mean = gguf_to_array(g.tensors[14],
            "batchnorm2_moving_mean")
    batchnorm2_moving_variance = gguf_to_array(g.tensors[15],
            "batchnorm2_moving_variance")
    dense_w = gguf_to_array(g.tensors[16], "dense_w")
    dense_b = gguf_to_array(g.tensors[17], "dense_b")
    print("    Done.")

    for iter in range(N_iter):
        i = 4212
        print("Input digit index:", i)
        inp = x_test[i,:,:]
        draw_digit(inp)
        print("Reference value:", y_test[i])
        x = run_model_np(inp,
                kernel1, bias1, kernel2, bias2, kernel3, bias3, kernel4, bias4,
                batchnorm1_gamma, batchnorm1_beta,
                batchnorm1_moving_mean, batchnorm1_moving_variance,
                batchnorm2_gamma, batchnorm2_beta,
                batchnorm2_moving_mean, batchnorm2_moving_variance,
                dense_w, dense_b)
        infer_val = np.argmax(x)
        print("NumPy Inferred value:", infer_val)
        print("NumPy Digit probabilities:", x)


if __name__ == '__main__':
    main()
