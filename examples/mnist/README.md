# MNIST

## Train using TensorFlow

Create and load Conda environment:

    mamba env create -f environment_tf.yml
    conda activate tf

Train the MNIST model using TensorFlow and save the model arrays in the GGUF
format:

    python mnist-tf.py train mnist-cnn-model
    python mnist-tf.py convert mnist-cnn-model

The first command downloads the MNIST dataset and trains the neural network.
The training takes about 1min 20s on Apple M1 Max, it automatically runs in
parallel. The second command creates `mnist-cnn-model.gguf`.

## Model Weights

The GGUF format can be examined using the provided utilities in the GGUF Python
package, already installed in the `tf` conda environment:

    $ gguf-dump mnist-cnn-model.gguf
    * Loading: mnist-cnn-model.gguf
    * File is LITTLE endian, script is running on a LITTLE endian host.

    * Dumping 4 key/value pair(s)
          1: UINT32     |        1 | GGUF.version = 3
          2: UINT64     |        1 | GGUF.tensor_count = 6
          3: UINT64     |        1 | GGUF.kv_count = 1
          4: STRING     |        1 | general.architecture = 'mnist-cnn'

    * Dumping 6 tensor(s)
          1:        288 |     3,     3,     1,    32 | F16     | kernel1
          2:      21632 |    26,    26,    32,     1 | F32     | bias1
          3:      18432 |     3,     3,    32,    64 | F16     | kernel2
          4:       7744 |    11,    11,    64,     1 | F32     | bias2
          5:      16000 |  1600,    10,     1,     1 | F32     | dense_w
          6:         10 |    10,     1,     1,     1 | F32     | dense_b

To load these weights from Python:

    >>> from gguf.gguf_reader import GGUFReader
    >>> g = GGUFReader("mnist-cnn-model.gguf")
    >>> print(g.fields)
    >>> print(g.tensors)
    >>> from numpy import array
    >>> array(g.tensors[2].data)
    array([-0.00785,  0.0158 ,  0.282  , ...,  0.3318 ,  0.09576, -0.11816],
          dtype=float16)

## Dimension Order

In general, PyTorch keeps the input array in the `(N, C, H, W)` order, while
TensorFlow uses `(N, H, W, C)` order. Where `N` is a batch size, C a number of
channels, H is a height of input planes in pixels, W is a width.

PyTorch uses the `(N, C, H, W)` order for convolutions, the weights are ordered
`(C_out, C_in, H, W)`.

TensorFlow by default uses `(N, H, W, C)` (this corresponds to the default
`data_format='channels_last'`), and it can also use `(N, C, H, W)` like PyTorch
(`data_format='channels_first'`), but currently only for a GPU.
The weights are ordered `(H, W, C_in, C_out)`.

When converting the convolution weights from TensorFlow to PyTorch, we use the
following transposition:

    # Conv2D
    # (H, W, C_in, C_out) -> (C_out, C_in, H, W)
    kernel1_ = np.transpose(kernel1, (3,2,0,1))

PyTorch uses `(N_out, C_in*H*W)` for weights of a linear layer, TensorFlow uses
`(H*W*C_in, N_out)`. After flattening, the linear layer matmul weights (matrix)
must be tranposed as follows when going from TensorFlow to PyTorch:

    # (H*W*C_in, N_out) -> (H, W, C_in, N_out)
    dense_w_ = np.reshape(dense_w, (5, 5, 64, 10))
    # (H, W, C_in, N_out) -> (N_out, C_in, H, W)
    dense_w_ = np.transpose(dense_w_, (3, 2, 0, 1))
    # (N_out, C_in, H, W) -> (N_out, C_in*H*W)
    dense_w_ = np.reshape(dense_w_, (10, 1600))

We deflatten into height, width and channels, then we transpose to PyTorch
order, and then we flatten again. The matmul operation itself is also
transposed in PyTorch compared to TensorFlow.

GGUF reverses the order of all dimensions. So if we save TensorFlow weights
that are ordered `(H, W, C_in, C_out)`, then in GGUF the data is intact, but
the dimensions are saved as `(C_out, C_in, W, H)`. As an example, for `H=W=3`,
`C_in=32`, `C_out=64`, the `gguf-dump` utility will print:

      3:      18432 |    64,    32,     3,     3 | F32     | kernel2

This must be read from right to left to recover the actual dimension order that
the data was saved in `(H=3, W=3, C_in=32, C_out=64)`. When we create a NumPy
array from GGUF data, we thus use `np.reshape(g.data, np.flip(g.shape))`, which
assumes a C-ordering of the NumPy array (last dimension contiguous), and
reverses the order of GGUF dimensions using `np.flip`.


## Inference

Load the GGUF model and run inference using PyTorch and TensorFlow:

    $ python inference.py
    Importing Python packages...
        Done.
    Loading MNIST test images...
        Done.
    Loading MNIST model GGUF...
        Done.
    Input digit index: 634
    ╔════════════════════════════════════════════════════════╗
    ║                                                        ║
    ║                                                        ║
    ║                                                        ║
    ║                                                        ║
    ║                                                        ║
    ║                                                        ║
    ║                                                        ║
    ║                    ▒▒██████████▓▓▒▒                    ║
    ║                ▓▓▓▓▓▓████▓▓▓▓▓▓████▒▒░░                ║
    ║            ░░▓▓████▓▓▒▒░░      ▒▒██████▒▒              ║
    ║            ▓▓████▒▒              ██████▒▒              ║
    ║          ▒▒██▓▓░░                ▓▓████▒▒              ║
    ║          ████▓▓                  ▒▒████░░              ║
    ║          ████▓▓                ▓▓▓▓████░░              ║
    ║          ▓▓████▒▒        ░░▒▒▓▓████████░░              ║
    ║            ▓▓████████████████████████▓▓                ║
    ║              ▒▒▓▓██████▓▓▓▓▒▒░░  ▒▒██▓▓                ║
    ║                                  ▓▓██▓▓                ║
    ║                                  ████▒▒                ║
    ║                                  ████▒▒                ║
    ║                                  ████▒▒                ║
    ║                                  ▓▓██▒▒                ║
    ║                                  ▒▒██▓▓                ║
    ║                                  ▒▒██▓▓                ║
    ║                                  ▒▒██▓▓                ║
    ║                                  ░░▓▓██░░              ║
    ║                                    ▒▒▓▓                ║
    ║                                                        ║
    ╚════════════════════════════════════════════════════════╝
    Reference value: 9
    Input shape: (28, 28)
    Output shape: (10,)
    PT: [2.6982696e-12 2.7950708e-14 1.8198607e-09 4.2483357e-08 2.9408625e-06
     1.2539382e-09 7.9030879e-18 2.3529321e-06 1.8065512e-07 9.9999440e-01]
    PT max: 9
    TF: tf.Tensor(
    [[2.6982696e-12 2.7950710e-14 1.8198641e-09 4.2483276e-08 2.9408570e-06
      1.2539311e-09 7.9029373e-18 2.3529278e-06 1.8065478e-07 9.9999440e-01]], shape=(1, 10), dtype=float32)
    TF max: 9
    Inferred value: 9
    Digit probabilities: [2.6982696e-12 2.7950708e-14 1.8198607e-09 4.2483357e-08 2.9408625e-06
     1.2539382e-09 7.9030879e-18 2.3529321e-06 1.8065512e-07 9.9999440e-01]
