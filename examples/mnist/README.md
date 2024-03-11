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
