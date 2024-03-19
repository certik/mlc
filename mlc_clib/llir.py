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
class f16(Type): pass

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
class conv2d_f16(Instruction):
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
class relu_f16(Instruction):
    n: int
    x_in: str
    x_out: str

@dataclass
class relu_32K_f16(Instruction):
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
class max_pool_2d_f16(Instruction):
    in_channels: int
    H: int
    W: int
    x_in: str
    x_out: str

@dataclass
class reshape(Instruction):
    shape: list[int]
    x_inout: str

# x_out(m) = A(m, n) x x_in(n) + y(m)
@dataclass
class saxpy(Instruction):
    m: int
    n: int
    A: str # matrix A(m, n)
    y: str # vector y(n)
    x_in: str # input vector x_in(n)
    x_out: str # output vector x_out(m)

# x_out(m) = A(m, n) x x_in(n) + y(m)
@dataclass
class saxpy_f16(Instruction):
    m: int
    n: int
    A: str # matrix A(m, n)
    y: str # vector y(n)
    x_in: str # input vector x_in(n)
    x_out: str # output vector x_out(m)

@dataclass
class softmax(Instruction):
    n: int
    x_in: str
    x_out: str

@dataclass
class softmax_f16(Instruction):
    n: int
    x_in: str
    x_out: str

@dataclass
class pad_32K_copy(Instruction):
    old_size: int
    x_in: str
    x_out: str

@dataclass
class section_32K_copy(Instruction):
    new_size: int
    x_in: str
    x_out: str

@dataclass
class cast_f32_f16(Instruction):
    n: str
    x_in: str
    x_out: str

@dataclass
class cast_32K_f32_f16(Instruction):
    x_in: str
    x_out: str

@dataclass
class cast_f16_f32(Instruction):
    n: str
    x_in: str
    x_out: str

@dataclass
class cast_32K_f16_f32(Instruction):
    x_in: str
    x_out: str

#############################################################

@dataclass
class Inference:
    x_in: Array
    x_out: Array
    weights: list[Array]
    temporaries: list[Array]
    instructions: list[Instruction]
