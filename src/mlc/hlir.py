from dataclasses import dataclass
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

@dataclass
class Array:
    name: str
    rank: int
    shape: list[int]

@dataclass
class Operation:
    op_type: str
    rank: int
    shape: list[int]
    args: list[Any]
