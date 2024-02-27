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
![mnist-cnn.dot.png](https://gist.githubusercontent.com/certik/8aaae7df1380c5ddf3f7931e315e58f6/raw/a84b69aa424fa3ed30c3999c45314736d079c549/mnist-cnn.dot.png)
