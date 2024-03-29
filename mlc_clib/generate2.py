from llir import (Inference, Array, f16, f32, conv2d, relu, max_pool_2d,
        reshape, saxpy, softmax, pad_32K_copy, cast_32K_f16_f32,
        cast_32K_f32_f16, section_32K_copy, relu_32K_f16, batch_norm_2d)
from ll_to_cpu import ll_to_cpu

ll = Inference(
        x_in = [
            Array("in", f32(), (1, 28, 28)),
        ],
        x_out = [
            Array("out", f32(), (10,)),
        ],
        weights=[
            Array("kernel1", f32(), (32, 1, 5, 5)),
            Array("bias1", f32(), (32,)),
            Array("kernel2", f32(), (32, 32, 5, 5)),
            Array("bias2", f32(), (32,)),
            Array("batchnorm1_gamma", f32(), (32,)),
            Array("batchnorm1_beta", f32(), (32,)),
            Array("batchnorm1_moving_mean", f32(), (32,)),
            Array("batchnorm1_moving_variance", f32(), (32,)),

            Array("kernel3", f32(), (64, 32, 3, 3)),
            Array("bias3", f32(), (64,)),
            Array("kernel4", f32(), (64, 64, 3, 3)),
            Array("bias4", f32(), (64,)),
            Array("batchnorm2_gamma", f32(), (64,)),
            Array("batchnorm2_beta", f32(), (64,)),
            Array("batchnorm2_moving_mean", f32(), (64,)),
            Array("batchnorm2_moving_variance", f32(), (64,)),

            Array("dense_w", f32(), (10, 576)),
            Array("dense_b", f32(), (10,)),
        ],
        temporaries=[
            Array("tmp1", f32(), (32, 24, 24)),
            Array("tmp2", f32(), (32, 24, 24)),
            Array("tmp3", f32(), (32, 20, 20)),
            Array("tmp4", f32(), (32, 20, 20)),
            Array("tmp5", f32(), (32, 20, 20)),
            Array("tmp6", f32(), (32, 10, 10)),

            Array("tmp7", f32(), (64, 8, 8)),
            Array("tmp8", f32(), (64, 8, 8)),
            Array("tmp9", f32(), (64, 6, 6)),
            Array("tmp10", f32(), (64, 6, 6)),
            Array("tmp11", f32(), (64, 6, 6)),
            Array("tmp12", f32(), (64, 3, 3)),

            Array("tmp13", f32(), (10,)),
        ],
        instructions=[
            conv2d(1, 32, 5, 28, 28, "kernel1", "bias1", "in", "tmp1"), # (32, 24, 24)
            relu(32, 24, 24, "tmp1", "tmp2"), # (32, 24, 24)
            conv2d(32, 32, 5, 24, 24, "kernel2", "bias2", "tmp2", "tmp3"), # (32, 20, 20)
            relu(32, 20, 20, "tmp3", "tmp4"), # (32, 20, 20)
            batch_norm_2d(32, 20, 20, "tmp4", "tmp5",
                "batchnorm1_gamma", "batchnorm1_beta",
                "batchnorm1_moving_mean", "batchnorm1_moving_variance"
                          ), # (32, 20, 20)
            max_pool_2d(32, 20, 20, "tmp5", "tmp6"), # (32, 10, 10)

            conv2d(32, 64, 3, 10, 10, "kernel3", "bias3", "tmp6", "tmp7"), # (64, 8, 8)
            relu(64, 8, 8, "tmp7", "tmp8"), # (64, 8, 8)
            conv2d(64, 64, 3, 8, 8, "kernel4", "bias4", "tmp8", "tmp9"), # (64, 6, 6)
            relu(64, 6, 6, "tmp9", "tmp10"), # (64, 6, 6)
            batch_norm_2d(64, 6, 6, "tmp10", "tmp11",
                "batchnorm2_gamma", "batchnorm2_beta",
                "batchnorm2_moving_mean", "batchnorm2_moving_variance"
                ), # (64, 6, 6)
            max_pool_2d(64, 6, 6, "tmp11", "tmp12"), # (64, 3, 3)

            saxpy(10, 576, "dense_w", "dense_b", "tmp12", "tmp13"), # (10)
            softmax(10, "tmp13", "out"), # (10)
        ]
    )

cpu_c, cpu_h = ll_to_cpu(ll)
open("inference-generated.c", "w").write(cpu_c)
open("inference-generated.h", "w").write(cpu_h)
