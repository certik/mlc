from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

# High-Level (Array) IR (HLIR)

# Represents the computational graph using array operations with a complete
# high-level semantics. It uses array terminology (`matmul`) not ML terminology
# (`linear layer`). All ML semantic information is now lost, everything is fully
# expressed in terms of arrays and operations on them. In particular, the
# following array information is present in HLIR:
# * rank (1, 2, 3, 4, etc.)
# * shape (dimensions and their order),
# * element type (f32, f16, f8, f5, etc.)
# * location (host / device / L1 / register, etc.)
# * layout (strides, column order/row order) --- if we support it
#
# Consequently, it also contains explicit casts between element types, location,
# etc. It has nodes for padding and slicing an array.
#
# The initial HLIR (as immediately lowered from NNIR) keeps high level "special
# functions" such as matmul, conv2d, etc. With subsequent passes (either in HLIR
# or a separate representation) we lower these operation by providing an actual
# implementation or a specific backend API (kernel) call.

class Type(Enum):
    f8 = auto()
    f16 = auto()
    f32 = auto()
    f64 = auto()

class OpType(Enum):
    MatMul = auto()
    MatVec = auto()
    Add = auto()
    ReLU = auto()
    Softmax = auto()
    Conv2D = auto()
    BatchNorm2D = auto()
    MaxPool2D = auto()
    Reshape = auto()

class MemorySpace(Enum):
    host = auto()
    apu_L4 = auto()
    apu_L1 = auto()
    apu_VR = auto()

class ExecutionSpace(Enum):
    host = auto()
    apu_arc = auto()
    apu_mmb = auto()

@dataclass
class Array:
    name: str
    type: Type
    rank: int
    shape: list[int]
    memory_space: MemorySpace

@dataclass
class Operation:
    op_type: OpType
    args: list[Any]
    # Where the operation is executed:
    execution_space: ExecutionSpace
    # Properties of the result:
    rank: int
    shape: list[int]
    memory_space: MemorySpace
