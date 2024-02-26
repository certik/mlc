from dataclasses import dataclass

# High Level IR

@dataclass
class Operation:
    op_type: str
    rank: int
    shape: list[int]
