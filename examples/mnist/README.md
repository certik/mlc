# MNIST

## Train using TensorFlow

Create and load Conda environment:

    mamba env create -f environment_tf.yml
    conda activate tf

Train the MNIST model using TensorFlow and save the model arrays in the GGUF
format:

    python mnist-tf.py train mnist-cnn-model
    python mnist-tf.py convert mnist-cnn-model

The training takes about 1min 20s on Apple M1 Max, it automatically runs in
parallel. The second command creates `mnist-cnn-model.gguf`.
