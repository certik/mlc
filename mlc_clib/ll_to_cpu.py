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
        self.weights = {w.name: w for w in x.weights}
        self.tmpinout = {w.name: w for w in x.x_in} | \
            {w.name: w for w in x.x_out} | \
            {w.name: w for w in x.temporaries}

        self.inf_body = ""
        for instruction in x.instructions:
            self.visit(instruction)

        args = []
        for var in x.x_in + x.x_out + x.weights + x.temporaries:
            args.append(f"f32 *{var.name} /*{var.shape}*/")
        self.inf_calc_args = "        " + ",\n        ".join(args)

        args = []
        for var in x.x_in + x.x_out + x.weights:
            args.append(f"f32 *{var.name} /*{var.shape}*/")
        self.inf_args = "        " + ",\n        ".join(args)

        args = []
        for var in x.temporaries:
            args.append(f"*{var.name}")
        assert len(args) > 0
        self.inf_body_decl1 = ", ".join(args)

        args = []
        for var in x.temporaries:
            args.append(f"&{var.name}")
        self.inf_body_args1 = ", ".join(args)

        args = []
        for var in x.x_in + x.x_out + x.weights + x.temporaries:
            args.append(var.name)
        self.inf_body_args2 = ", ".join(args)

        args = []
        for var in x.temporaries:
            args.append(f"f32 **{var.name} /*{var.shape}*/")
        self.inf_alloc_temp_args = "        " + ",\n        ".join(args)

        args = []
        for var in x.temporaries:
            dims = "*".join([str(s) for s in var.shape])
            args.append(f"*{var.name} = malloc({dims}*sizeof(f32));")
        self.inf_alloc_temp_body = "    " + "\n    ".join(args)

        self.cpu_c = f"""\
// This file was generated using `generate.py`. Do not modify by hand.

#include <stdio.h>
#include <stdlib.h>

#include "inference-generated.h"
#include "kernels.h"

void inference_calculation(
{self.inf_calc_args}
    ) {{
{self.inf_body}}}

void allocate_temporaries(
{self.inf_alloc_temp_args}
) {{
{self.inf_alloc_temp_body}
}}

void inference(
{self.inf_args}
) {{
    f32 {self.inf_body_decl1};
    allocate_temporaries({self.inf_body_args1});
    inference_calculation({self.inf_body_args2});
}}
"""
        self.cpu_h = ""

    def visit_conv2d(self, x):
        self.inf_body += f"""\
    conv2d({x.in_channels}, {x.out_channels}, {x.kernel_size}, {x.H}, {x.W},
        {x.kernel}, // {self.weights[x.kernel].shape}
        {x.bias}, // {self.weights[x.bias].shape}
        {x.x_in}, // {self.tmpinout[x.x_in].shape}
        {x.x_out} // {self.tmpinout[x.x_out].shape}
    );
"""

    def visit_relu(self, x):
        self.inf_body += f"""\
    relu({x.in_channels}, {x.H}, {x.W},
        {x.x_in}, // {self.tmpinout[x.x_in].shape}
        {x.x_out} // {self.tmpinout[x.x_out].shape}
    );
"""

    def visit_max_pool_2d(self, x):
        self.inf_body += f"""\
    max_pool_2d({x.in_channels}, {x.H}, {x.W},
        {x.x_in}, // {self.tmpinout[x.x_in].shape}
        {x.x_out} // {self.tmpinout[x.x_out].shape}
    );
"""

    def visit_reshape(self, x):
        self.inf_body += f"""\
    // NOOP: Reshape({x.x_inout}, {x.shape})
"""

    def visit_saxpy(self, x):
        self.inf_body += f"""\
    saxpy({x.m}, {x.n},
        {x.A}, // {self.weights[x.A].shape}
        {x.x_in}, // {self.tmpinout[x.x_in].shape}
        {x.y}, // {self.weights[x.y].shape}
        {x.x_out} // {self.tmpinout[x.x_out].shape}
    );
"""

    def visit_softmax(self, x):
        self.inf_body += f"""\
    softmax({x.n},
        {x.x_in}, // {self.tmpinout[x.x_in].shape}
        {x.x_out} // {self.tmpinout[x.x_out].shape}
    );
"""

def ll_to_cpu(ll: Inference):
    v = LLToCPUVisitor()
    v.visit_Inference(ll)
    return v.cpu_c, v.cpu_h
