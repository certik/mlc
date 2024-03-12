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
