from llir import (Inference, Array, i32, i64, f32, f64, u8, u16, u32, u64)

def convert_LL_type(t):
    if isinstance(t, i32):
        stype = "int32_t"
    elif isinstance(t, i64):
        stype = "int64_t"
    elif isinstance(t, f32):
        stype = "float"
    elif isinstance(t, f64):
        stype = "double"
    elif isinstance(t, u8):
        stype = "uint8_t"
    elif isinstance(t, u16):
        stype = "uint16_t"
    elif isinstance(t, u32):
        stype = "uint32_t"
    elif isinstance(t, u64):
        stype = "uint64_t"
    else:
        raise Exception("Unsupported LLIR type: %s" % str(t))
    return stype

def convert_LL_type2(t):
    if isinstance(t, Array):
        stype = convert_LL_type(t.element_type)
    else:
        stype = convert_LL_type(t)
    return stype

class LLToCPUVisitor:

    def __init__(self):
        self.src = ""
        self.indent = " "*4
        self.instructions = []

    def visit(self, x):
        supported_nodes = ["Inference", "conv2d",
                           "relu", "max_pool_2d",
                           "reshape", "saxpy",
                           "softmax"
                           ]
        node_name = type(x).__name__
        if node_name in supported_nodes:
            eval("self.visit_%s(x)" % node_name)
        else:
            raise Exception("Unsupported LLIR node: %s" % node_name)

    def visit_Inference(self, x):
        self.cpu_c = ""
        self.cpu_h = ""

def ll_to_cpu(ll: Inference):
    v = LLToCPUVisitor()
    v.visit_Inference(ll)
    return v.cpu_c, v.cpu_h
