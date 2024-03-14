#include "kernels.h"

void inference(
        f32 *in,      // (1, 28, 28)
        f32 *out,     // (10,)
        f32 *kernel1, // (3, 3, 1, 32)
        f32 *bias1,   // (32,)
        f32 *kernel2, // (3, 3, 32, 64)
        f32 *bias2,   // (64,)
        f32 *dense_w, // (1600, 10)
        f32 *dense_b // (10,)
    );
