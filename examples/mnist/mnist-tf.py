# This file was taken from GGML (MIT license):
# https://github.com/ggerganov/ggml/blob/43a6d4af1971ee2912ff7bc2404011ff327b6a60/examples/mnist/mnist-cnn.py
import sys
import gguf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def train(model_name):
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()
    batch_size = 128
    epochs = 1
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    model.save(model_name)
    print("Keras model saved to '" + model_name + "'")

def convert(model_name):
    model = keras.models.load_model(model_name)
    gguf_model_name = model_name + ".gguf"
    gguf_writer = gguf.GGUFWriter(gguf_model_name, "mnist-cnn")

    kernel1 = model.layers[0].weights[0].numpy()
    #kernel1 = kernel1.astype(np.float16)
    gguf_writer.add_tensor("kernel1", kernel1)

    bias1 = model.layers[0].weights[1].numpy()
    gguf_writer.add_tensor("bias1", bias1)

    kernel2 = model.layers[2].weights[0].numpy()
    #kernel2 = kernel2.astype(np.float16)
    gguf_writer.add_tensor("kernel2", kernel2)

    bias2 = model.layers[2].weights[1].numpy()
    gguf_writer.add_tensor("bias2", bias2)

    dense_w = model.layers[-1].weights[0].numpy()
    gguf_writer.add_tensor("dense_w", dense_w)

    dense_b = model.layers[-1].weights[1].numpy()
    gguf_writer.add_tensor("dense_b", dense_b)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"Model converted and saved to '{gguf_model_name}'")

def convert_tests(model_name):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(x_test.dtype)
    print(y_test.dtype)

    gguf_model_name = model_name + ".gguf"
    gguf_writer = gguf.GGUFWriter(gguf_model_name, "mnist-cnn")

    gguf_writer.add_tensor("x_test", np.array(x_test, dtype=np.int8))
    gguf_writer.add_tensor("y_test", np.array(y_test, dtype=np.int8))

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"MNIST test data saved to '{gguf_model_name}'")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: %s <train|convert> <model_name>".format(sys.argv[0]))
        sys.exit(1)
    if sys.argv[1] == 'train':
        train(sys.argv[2])
    elif sys.argv[1] == 'convert':
        convert(sys.argv[2])
    elif sys.argv[1] == 'convert_tests':
        convert_tests(sys.argv[2])
    else:
        print("Usage: %s <train|convert> <model_name>".format(sys.argv[0]))
        sys.exit(1)
