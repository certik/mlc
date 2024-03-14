from dataclasses import dataclass
from enum import Enum, auto

@dataclass
class Type:
    pass

@dataclass
class u8(Type): pass

@dataclass
class u16(Type): pass

@dataclass
class u32(Type): pass

@dataclass
class u64(Type): pass

@dataclass
class i32(Type): pass

@dataclass
class i64(Type): pass

@dataclass
class f32(Type): pass

@dataclass
class f64(Type): pass

@dataclass
class Array:
    name: str
    element_type: Type
    shape: list[int] # compile time shape

@dataclass
class Instruction:
    pass

@dataclass
class conv2d(Instruction):
    in_channels: int
    out_channels: int
    kernel_size: int
    H: int
    W: int
    kernel: str
    bias: str
    x_in: str
    x_out: str

@dataclass
class relu(Instruction):
    in_channels: int
    H: int
    W: int
    x_in: str
    x_out: str

@dataclass
class max_pool_2d(Instruction):
    in_channels: int
    H: int
    W: int
    x_in: str
    x_out: str

@dataclass
class reshape(Instruction):
    shape: list[int]
    x_inout: str

# (m, n) x (n,) + (n,)
@dataclass
class saxpy(Instruction):
    m: int
    n: int
    # (m, n)
    A_name: str
    # (n,)
    x_name: str # input argument
    # (n,)
    y_name: str
    # (n,)
    out_name: str # output argument

@dataclass
class softmax(Instruction):
    n: int
    x_in: str
    x_out: str

@dataclass
class Inference:
    x_in: Array
    x_out: Array
    weights: list[Array]
    temporaries: list[Array]
    instructions: list[Instruction]
