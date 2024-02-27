# Machine Learning Compiler (mlc)

## Build

```
mamba env create -f environment_unix.yml
conda activate mlc
```

## Run tests:

```
pytest
```

Plot the computational graph:
```
dot -Tpng mnist-cnn.dot -o mnist-cnn.dot.png
```
