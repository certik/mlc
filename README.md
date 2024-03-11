# Machine Learning Compiler (mlc)

In top-level directory, `mlc`.

## Build

### (Recommended) Install `mamba` and `conda` from `miniforge`.

We have found that the `miniforge` versions of `mamba` and `conda`
work better than those you get from the `anaconda` graphical 
installer. You can make any of them work, but this seems 
smoothest to us.

https://github.com/conda-forge/miniforge

```
mamba env create -f environment_unix.yml
conda activate mlc
```

## Install `src`, or set `PYTHONPATH`

There are two ways to make the code accessible
and to reload it on each run: 

### Alternative 1: Install Editable

into the `mlc` environment

```
pip install -e .
```

### Alternative 2: Set `PYTHONPATH`

Either on-the-fly for each run, for example: 

```
PYTHONPATH="./src:$PYTHONPATH" pytest
```

or once per terminal session: 

```
export PYTHONPATH="./src:$PYTHONPATH"
```

before running tests.

## Run tests:

```
pytest
```

## See the IR

The `-s` option tells `pytest` to display `print` output.

```
pytest -s
```

## Plot the computational graph:

```
dot -Tpng mnist-cnn.dot -o mnist-cnn.dot.png
```
![mnist-cnn.dot.png](https://gist.githubusercontent.com/certik/8aaae7df1380c5ddf3f7931e315e58f6/raw/a84b69aa424fa3ed30c3999c45314736d079c549/mnist-cnn.dot.png)
