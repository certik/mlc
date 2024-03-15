#include <stdio.h>
#include <stdlib.h>

#include "inference-generated.h"
#include "kernels.h"

void inference_calculation(
        f32 *in /*(1, 28, 28)*/,
        f32 *out /*(10,)*/,
        f32 *kernel1 /*(32, 1, 3, 3)*/,
        f32 *bias1 /*(32,)*/,
        f32 *kernel2 /*(64, 32, 3, 3)*/,
        f32 *bias2 /*(64,)*/,
        f32 *dense_w /*(10, 1600)*/,
        f32 *dense_b /*(10,)*/,
        f32 *tmp2 /*(32, 26, 26)*/,
        f32 *tmp3 /*(32, 26, 26)*/,
        f32 *tmp4 /*(32, 13, 13)*/,
        f32 *tmp5 /*(64, 11, 11)*/,
        f32 *tmp6 /*(64, 11, 11)*/,
        f32 *tmp7 /*(64, 5, 5)*/,
        f32 *tmp8 /*(10,)*/
    ) {
    conv2d(1, 32, 3, 28, 28,
        kernel1, // (32, 1, 3, 3)
        bias1, // (32,)
        in, // (1, 28, 28)
        tmp2 // (32, 26, 26)
    );
    relu(32, 26, 26,
        tmp2, // (32, 26, 26)
        tmp3 // (32, 26, 26)
    );
    max_pool_2d(32, 26, 26,
        tmp3, // (32, 26, 26)
        tmp4 // (32, 13, 13)
    );
    conv2d(32, 64, 3, 13, 13,
        kernel2, // (64, 32, 3, 3)
        bias2, // (64,)
        tmp4, // (32, 13, 13)
        tmp5 // (64, 11, 11)
    );
    relu(64, 11, 11,
        tmp5, // (64, 11, 11)
        tmp6 // (64, 11, 11)
    );
    max_pool_2d(64, 11, 11,
        tmp6, // (64, 11, 11)
        tmp7 // (64, 5, 5)
    );
    // NOOP: Reshape(tmp7, (1600,))
    saxpy(10, 1600,
        dense_w, // (10, 1600)
        tmp7, // (64, 5, 5)
        dense_b, // (10,)
        tmp8 // (10,)
    );
    softmax(10,
        tmp8, // (10,)
        out // (10,)
    );
}

void allocate_temporaries(
        f32 **tmp2 /*(32, 26, 26)*/,
        f32 **tmp3 /*(32, 26, 26)*/,
        f32 **tmp4 /*(32, 13, 13)*/,
        f32 **tmp5 /*(64, 11, 11)*/,
        f32 **tmp6 /*(64, 11, 11)*/,
        f32 **tmp7 /*(64, 5, 5)*/,
        f32 **tmp8 /*(10,)*/
) {
    *tmp2 = malloc(32*26*26*sizeof(f32));
    *tmp3 = malloc(32*26*26*sizeof(f32));
    *tmp4 = malloc(32*13*13*sizeof(f32));
    *tmp5 = malloc(64*11*11*sizeof(f32));
    *tmp6 = malloc(64*11*11*sizeof(f32));
    *tmp7 = malloc(64*5*5*sizeof(f32));
    *tmp8 = malloc(10*sizeof(f32));
}

void inference(
        f32 *in /*(1, 28, 28)*/,
        f32 *out /*(10,)*/,
        f32 *kernel1 /*(32, 1, 3, 3)*/,
        f32 *bias1 /*(32,)*/,
        f32 *kernel2 /*(64, 32, 3, 3)*/,
        f32 *bias2 /*(64,)*/,
        f32 *dense_w /*(10, 1600)*/,
        f32 *dense_b /*(10,)*/
) {
    f32 *tmp2, *tmp3, *tmp4, *tmp5, *tmp6, *tmp7, *tmp8;
    allocate_temporaries(&tmp2, &tmp3, &tmp4, &tmp5, &tmp6, &tmp7, &tmp8);
    inference_calculation(in, out, kernel1, bias1, kernel2, bias2, dense_w, dense_b, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);
}
