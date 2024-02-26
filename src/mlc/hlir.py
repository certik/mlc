from dataclasses import dataclass
from typing import Any

# High-Level Array IR

# Represents all array operations using a complete high level semantics.
# Does not contain details about how and where operations are executed or where
# a given array is stored.

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
