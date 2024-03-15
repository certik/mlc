# C inference driver

## Prepare gguf files

Follow the instructions in `examples/mnist/README` to prepare the two gguf
files there.

## Build

To build it:

    conda activate tf
    python generate.py
    cmake .
    make
    ./mlc_clib
