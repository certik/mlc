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
        self.cpu_c = f"""\
#include <stdio.h>
#include <stdlib.h>

#include "inference-generated.h"
#include "kernels.h"

void inference_calculation(
        f32 *in,      // (1, 28, 28)
        f32 *out,     // (10,)
        f32 *kernel1, // (32, 1, 3, 3)
        f32 *bias1,   // (32,)
        f32 *kernel2, // (32, 64, 3, 3)
        f32 *bias2,   // (64,)
        f32 *dense_w, // (1600, 10)
        f32 *dense_b, // (10,)
        f32 *out2,    // (32, 26, 26)
        f32 *out3,    // (32, 26, 26)
        f32 *out4,    // (32, 13, 13)
        f32 *out5,    // (64, 11, 11)
        f32 *out6,    // (64, 11, 11)
        f32 *out7,    // (64, 5, 5)
        f32 *out8     // (10,)
    ) {{
{self.inf_body}}}

void allocate_temporaries(
        f32 **out2,
        f32 **out3,
        f32 **out4,
        f32 **out5,
        f32 **out6,
        f32 **out7,
        f32 **out8
) {{
    *out2 = malloc(32*26*26*sizeof(f32));
    *out3 = malloc(32*26*26*sizeof(f32));
    *out4 = malloc(32*13*13*sizeof(f32));
    *out5 = malloc(64*11*11*sizeof(f32));
    *out6 = malloc(64*11*11*sizeof(f32));
    *out7 = malloc(64*5*5*sizeof(f32));
    *out8 = malloc(10*sizeof(f32));
}}

void inference(
        f32 *in,      // (1, 28, 28)
        f32 *out,     // (10,)
        f32 *kernel1, // (32, 1, 3, 3)
        f32 *bias1,   // (32,)
        f32 *kernel2, // (3, 3, 32, 64)
        f32 *bias2,   // (64,)
        f32 *dense_w, // (1600, 10)
        f32 *dense_b  // (10,)
) {{
    f32 *out2, *out3, *out4, *out5, *out6, *out7, *out8;
    allocate_temporaries(&out2, &out3, &out4, &out5, &out6, &out7, &out8);
    inference_calculation(in, out, kernel1, bias1, kernel2, bias2,
        dense_w, dense_b,
        out2, out3, out4, out5, out6, out7, out8
        );
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
        {x.A_name}, // {self.weights[x.A_name].shape}
        {x.x_in}, // {self.tmpinout[x.x_in].shape}
        {x.y_name}, // {self.weights[x.y_name].shape}
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
