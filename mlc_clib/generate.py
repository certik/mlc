from llir import (Inference, Array, f32, conv2d, relu, max_pool_2d, reshape,
        saxpy, softmax)
from ll_to_cpu import ll_to_cpu

ll = Inference(
        x_in = [
            Array("in", f32(), (1, 28, 28)),
        ],
        x_out = [
            Array("out", f32(), (10,)),
        ],
        weights=[
            Array("kernel1", f32(), (32, 1, 3, 3)),
            Array("bias1", f32(), (32,)),
            Array("kernel2", f32(), (64, 32, 3, 3)),
            Array("bias2", f32(), (64,)),
            Array("dense_w", f32(), (10, 1600)),
            Array("dense_b", f32(), (10,)),
        ],
        temporaries=[
            Array("tmp2", f32(), (32, 26, 26)),
            Array("tmp3", f32(), (32, 26, 26)),
            Array("tmp4", f32(), (32, 13, 13)),
            Array("tmp5", f32(), (64, 11, 11)),
            Array("tmp6", f32(), (64, 11, 11)),
            Array("tmp7", f32(), (64, 5, 5)),
            Array("tmp8", f32(), (10,)),
        ],
        # Verify pass: the array arguments fully determine the parameters
        instructions=[
            conv2d(1, 32, 3, 28, 28, "kernel1", "bias1", "in", "tmp2"),
            relu(32, 26, 26, "tmp2", "tmp3"),
            max_pool_2d(32, 26, 26, "tmp3", "tmp4"),
            conv2d(32, 64, 3, 13, 13, "kernel2", "bias2", "tmp4", "tmp5"),
            relu(64, 11, 11, "tmp5", "tmp6"),
            max_pool_2d(64, 11, 11, "tmp6", "tmp7"),
            reshape((1600,), "tmp7"),
            saxpy(10, 1600, "dense_w", "dense_b", "tmp7", "tmp8"),
            softmax(10, "tmp8", "out"),
        ]
    )

cpu_c, cpu_h = ll_to_cpu(ll)
open("inference-generated2.c", "w").write(cpu_c)
open("inference-generated2.h", "w").write(cpu_h)
