from llir import (Inference, Array, f16, f32, conv2d, relu, max_pool_2d,
        reshape, saxpy, saxpy_f16, softmax, pad_32K_copy, cast_32K_f16_f32,
        cast_32K_f32_f16, section_32K_copy, relu_32K_f16, cast_f32_f16,
        softmax_f16, cast_f16_f32, max_pool_2d_f16)
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
            Array("tmp5b", f32(), (32768,)),
            Array("tmp5c", f16(), (32768,)),
            Array("tmp5d", f16(), (32768,)),
            Array("tmp5e", f32(), (32768,)),

            Array("tmp6", f32(), (64, 11, 11)),
            Array("tmp6c", f16(), (64, 11, 11)),

            Array("tmp7", f16(), (64, 5, 5)),
            Array("tmp7b", f32(), (32768,)),
            Array("tmp7c", f16(), (1600,)),
            Array("tmp7d", f16(), (10,)),
            Array("tmp7e", f32(), (10,)),

            Array("tmp8", f32(), (10,)),
            Array("tmp8b", f16(), (10,)),
            Array("tmp8c", f16(), (10,)),
        ],
        # Verify pass: the array arguments fully determine the parameters
        instructions=[
            conv2d(1, 32, 3, 28, 28, "kernel1", "bias1", "in", "tmp2"),
            relu(32, 26, 26, "tmp2", "tmp3"),
            max_pool_2d(32, 26, 26, "tmp3", "tmp4"),
            conv2d(32, 64, 3, 13, 13, "kernel2", "bias2", "tmp4", "tmp5"),

            #relu(64, 11, 11, "tmp5", "tmp6"),
            pad_32K_copy(7744, "tmp5", "tmp5b"),
            cast_32K_f32_f16("tmp5b", "tmp5c"),
            relu_32K_f16("tmp5c", "tmp5d"),
            cast_32K_f16_f32("tmp5d", "tmp5e"),
            section_32K_copy(7744, "tmp5e", "tmp6"),

            cast_f32_f16(7744, "tmp6", "tmp6c"),
            max_pool_2d_f16(64, 11, 11, "tmp6c", "tmp7"),
            saxpy_f16(10, 1600, "dense_w", "dense_b", "tmp7", "tmp7d"),
            softmax_f16(10, "tmp7d", "tmp8c"),

            cast_f16_f32(10, "tmp8c", "out"),
        ]
    )

cpu_c, cpu_h = ll_to_cpu(ll)
open("inference-generated.c", "w").write(cpu_c)
open("inference-generated.h", "w").write(cpu_h)
