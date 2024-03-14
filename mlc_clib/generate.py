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
            Array("kernel2", f32(), (32, 64, 3, 3)),
            Array("bias2", f32(), (64,)),
            Array("dense_w", f32(), (1600,10)),
            Array("dense_b", f32(), (10,)),
        ],
        temporaries=[
            Array("out2", f32(), (32, 26, 26)),
            Array("out3", f32(), (32, 26, 26)),
            Array("out4", f32(), (32, 13, 13)),
            Array("out5", f32(), (64, 11, 11)),
            Array("out6", f32(), (64, 11, 11)),
            Array("out7", f32(), (64, 5, 5)),
            Array("out8", f32(), (10,)),
        ],
        # Verify pass: the array arguments fully determine the parameters
        instructions=[
            conv2d(1, 32, 3, 28, 28, "kernel1", "bias1", "in", "out"),
            relu(32, 26, 26, "out2", "out3"),
            max_pool_2d(32, 26, 26, "out3", "out4"),
            conv2d(32, 64, 3, 13, 13, "kernel2", "bias2", "out4", "out5"),
            relu(64, 11, 11, "out5", "out6"),
            max_pool_2d(64, 11, 11, "out6", "out7"),
            reshape((1600,), "out7"),
            saxpy(1600, 10, "dense_w", "dense_b", "out7", "out8"),
            softmax(10, "out8", "out"),
        ]
    )

cpu_c, cpu_h = ll_to_cpu(ll)
open("inference-generated2.c", "w").write(cpu_c)
open("inference-generated2.h", "w").write(cpu_h)
