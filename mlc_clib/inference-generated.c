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
    ) {
    // Conv2D
    conv2d(1, 32, 3,
        28, 28,
        kernel1, // (32, 1, 3, 3)
        bias1, // (32,)
        in, // (1, 28, 28)
        out2 // (32, 26, 26)
        );

    // ReLU
    relu(32, 26, 26,
        out2, // (32, 26, 26)
        out3  // (32, 26, 26)
        );

    // MaxPool2D
    max_pool_2d(32, 26, 26,
        out3, // (32, 26, 26)
        out4  // (32, 13, 13)
        );

    // Conv2D
    conv2d(32, 64, 3,
        13, 13,
        kernel2, // (32, 64, 3, 3)
        bias2, // (32,)
        out4, // (32, 13, 13)
        out5 // (64, 11, 11)
        );

    // ReLU
    relu(64, 11, 11,
        out5, // (64, 11, 11)
        out6  // (64, 11, 11)
        );

    // MaxPool2D
    max_pool_2d(64, 11, 11,
        out6, // (64, 11, 11)
        out7  // (64, 5, 5)
        );

    // Flatten: out7 (64, 5, 5) -> (1600,)

    // Linear
    saxpy(10, 1600,
            dense_w,  // (10, 1600)
            out7,     // (1600,)
            dense_b,  // (10,)
            out8      // (10,)
        );

    // Softmax
    softmax(10,
            out8, // (10,)
            out   // (10,)
        );
}

void allocate_temporaries(
        f32 **out2,
        f32 **out3,
        f32 **out4,
        f32 **out5,
        f32 **out6,
        f32 **out7,
        f32 **out8
) {
    *out2 = malloc(32*26*26*sizeof(f32));
    *out3 = malloc(32*26*26*sizeof(f32));
    *out4 = malloc(32*13*13*sizeof(f32));
    *out5 = malloc(64*11*11*sizeof(f32));
    *out6 = malloc(64*11*11*sizeof(f32));
    *out7 = malloc(64*5*5*sizeof(f32));
    *out8 = malloc(10*sizeof(f32));
}

void inference(
        f32 *in,      // (1, 28, 28)
        f32 *out,     // (10,)
        f32 *kernel1, // (32, 1, 3, 3)
        f32 *bias1,   // (32,)
        f32 *kernel2, // (3, 3, 32, 64)
        f32 *bias2,   // (64,)
        f32 *dense_w, // (1600, 10)
        f32 *dense_b  // (10,)
) {
    f32 *out2, *out3, *out4, *out5, *out6, *out7, *out8;
    allocate_temporaries(&out2, &out3, &out4, &out5, &out6, &out7, &out8);
    inference_calculation(in, out, kernel1, bias1, kernel2, bias2,
        dense_w, dense_b,
        out2, out3, out4, out5, out6, out7, out8
        );
}
