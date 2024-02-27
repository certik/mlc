from dataclasses import dataclass
from typing import Any

# Neural Network IR (NNIR)

# Basic building blocks for (neural network) graphs for machine learning.
# These nodes correspond to
# * https://pytorch.org/docs/stable/nn.html
# * https://www.tensorflow.org/api_docs/python/tf/keras/
#
# These nodes express the high-level neural network nodes, they do not contain
# any array specific details, such as layout, rank, order of dimensions,
# location (host / device), etc. These nodes contain neural network information
# using ML neural network terminology (e.g. `linear layer`), not array
# terminology (`matmul`).
#
# The NNIR gets transformed to HLIR which will contain all the array
# information.

## Linear Layers

@dataclass
class Linear:
    in_features: int
    out_features: int
    bias: bool

## Convolution Layers

@dataclass
class Conv2D:
    in_channels: int
    out_channels: int
    kernel_size: int
    bias: bool

## Pooling Layers

@dataclass
class MaxPool2D:
    kernel_size: list[int]

## Normalization Layers

@dataclass
class BatchNorm2D:
    num_features: int

@dataclass
class GroupNorm:
    num_groups: int
    num_channels: int

## Non-linear Activations

@dataclass
class ReLU:
    pass

@dataclass
class Softmax:
    pass

@dataclass
class Tanh:
    pass

@dataclass
class Sigmoid:
    pass

## Utilities

@dataclass
class Flatten:
    start_dim: int
    end_dim: int

# Note: This node does not belong to NN IR, rather it belongs to HLIR, which
# deals with array layouts. This node is not in torch.nn either.
#@dataclass
#class Transpose:
#    permutation: list[int]

## Transformers

@dataclass
class Transformer:
    d_model: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int

## Containers

@dataclass
class Sequential:
    layers: list[Any]
