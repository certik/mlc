//
// Created by Brian Beckman on 3/10/24.
//


#include <math.h>

#include "kernels.h"

/*
 * Index computations, from multi-dimensional indices to linear
 * indices. The rightmost index increases fastest. These
 * expressions do not actually depend on, D1, the size of the
 * slowest dimension, but it is included in the signature for
 * symmetry.
 */
#define I2(D1, D2, i, j) ((i)*(D2)+(j))
#define I3(D1, D2, D3, i, j, k) ((i)*(D3)*(D2)+(j)*(D3)+(k))
#define I4(D1, D2, D3, D4, i, j, k, l) ((i)*(D4)*(D3)*(D2)+(j)*(D4)*(D3)+(k)*(D4)+(l))

// A -> B
/*
 * Permute the dimensions as follows:
 *
 *     (1 2 3 4) ->
 *     (4 3 1 2)
 *
 * TODO: This is a hard-coded special case.
 */
void transpose(int n1, int n2, int n3, int n4, const f32 *A,
               int t1, int t2, int t3, int t4, f32 *B) {
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < n3; k++) {
                for (int l = 0; l < n4; l++) {
                    // TODO: for now the transposition is
                    // hardwired to (3,2,0,1)
                    B[I4(n4, n3, n1, n2, l, k, i, j)] =
                        A[I4(n1, n2, n3, n4, i, j, k, l)];
                }
            }
        }
    }
}

// (h, w)
/*
 * Note: Accumulate (+=) into `out`. `Out` must be initialized by
 * the caller.
 *
 * https://paperswithcode.com/method/convolution
 *
 * The code below is specialized to a square kernel (boxcar) of
 * odd size (1, 3, 5, ..). It reduces the iterations by k-1 and
 * indexes the boxcar from 0. For an even-sized boxcar, decrease
 * the outer iteration count by 2 * floor(k / 2), which also
 * happens to work for odd-sized boxcars.
 */
void conv2d_kernel(int kernel_size, int in_h, int in_w,
                   const f32 *weight, // (3,3)
                   const f32 *x, // (in_h,in_w)
                   f32 *out // (out_w,out_h)
) {
    int out_w = in_w - (kernel_size - 1);
    int out_h = in_h - (kernel_size - 1);
    for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    out[I2(out_h, out_w, h, w)]
                        += weight[I2(3, 3, i, j)]
                        * x[I2(in_h, in_w, h + i, w + j)];
                }
            }
        }
    }
}

void conv2d_kernel_f16(int kernel_size, int in_h, int in_w,
                   const f32 *weight, // (3,3)
                   const f16 *x, // (in_h,in_w)
                   f16 *out // (out_w,out_h)
) {
    int out_w = in_w - (kernel_size - 1);
    int out_h = in_h - (kernel_size - 1);
    for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    out[I2(out_h, out_w, h, w)]
                        += (f32)(weight[I2(3, 3, i, j)])
                        * x[I2(in_h, in_w, h + i, w + j)];
                }
            }
        }
    }
}


// (batch, channel, h, w)
/*
 * Multiple-In-Channel, Multiple-out-channel 2D Convolution.
 *
 * https://paperswithcode.com/method/convolution
 *
 * Note: Accumulate (+=) into `out`. `Out` must be initialized by
 * the caller.
 *
 * The code below is specialized to a square kernel (boxcar) of
 * odd size (1, 3, 5, ..). It reduces the iterations by k-1 and
 * indexes the boxcar from 0. For an even-sized boxcar, decrease
 * the outer iteration count by 2 * floor(k / 2), which also
 * happens to work for odd-sized boxcars.
 */
void conv2d(int in_channels, int out_channels, int kernel_size,
            int in_h, int in_w,
            f32 *weight, // (out_channels,in_channels,3,3)
            const f32 *bias, // (out_channels,)
            f32 *x, // (in_channels,in_h,in_w)
            f32 *out // (out_channels,out_h,out_w)
) {
    int out_w = in_w - (kernel_size - 1);
    int out_h = in_h - (kernel_size - 1);
    f32 s[out_h * out_w];
    for (int c = 0; c < out_channels; c++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                s[I2(out_h, out_w, i, j)] = 0;
            }
        }
        for (int k = 0; k < in_channels; k++) {
            conv2d_kernel(kernel_size, in_h, in_w,
                          &weight[I4(out_channels, in_channels, 3, 3, c, k, 0, 0)],
                          &x[I3(in_channels, in_h, in_w, k, 0, 0)],
                          s);
        }
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                out[I3(out_channels, out_h, out_w, c, i, j)]
                    = bias[c] + s[I2(out_h, out_w, i, j)];
            }
        }
    }
}

void conv2d_no_bias(int in_channels, int out_channels, int kernel_size,
            int in_h, int in_w,
            f32 *weight, // (out_channels,in_channels,3,3)
            f32 *x, // (in_channels,in_h,in_w)
            f32 *out // (out_channels,out_h,out_w)
) {
    int out_w = in_w - (kernel_size - 1);
    int out_h = in_h - (kernel_size - 1);
    f32 s[out_h * out_w];
    for (int c = 0; c < out_channels; c++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                s[I2(out_h, out_w, i, j)] = 0;
            }
        }
        for (int k = 0; k < in_channels; k++) {
            conv2d_kernel(kernel_size, in_h, in_w,
                          &weight[I4(out_channels, in_channels, 3, 3, c, k, 0, 0)],
                          &x[I3(in_channels, in_h, in_w, k, 0, 0)],
                          s);
        }
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                out[I3(out_channels, out_h, out_w, c, i, j)]
                    = s[I2(out_h, out_w, i, j)];
            }
        }
    }
}

void conv2d_f16(int in_channels, int out_channels, int kernel_size,
            int in_h, int in_w,
            f32 *weight, // (out_channels,in_channels,3,3)
            const f32 *bias, // (out_channels,)
            f16 *x, // (in_channels,in_h,in_w)
            f16 *out // (out_channels,out_h,out_w)
) {
    int out_w = in_w - (kernel_size - 1);
    int out_h = in_h - (kernel_size - 1);
    f16 s[out_h * out_w];
    for (int c = 0; c < out_channels; c++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                s[I2(out_h, out_w, i, j)] = 0;
            }
        }
        for (int k = 0; k < in_channels; k++) {
            conv2d_kernel_f16(kernel_size, in_h, in_w,
                          &weight[I4(out_channels, in_channels, 3, 3, c, k, 0, 0)],
                          &x[I3(in_channels, in_h, in_w, k, 0, 0)],
                          s);
        }
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                out[I3(out_channels, out_h, out_w, c, i, j)]
                    = (f16)(bias[c]) + s[I2(out_h, out_w, i, j)];
            }
        }
    }
}

// (channel, h, w)
/*
 * ReLU (rectified linear units) is defined as f(x) = max(0, x),
 * the identity function with the left (negative) half of the real
 * line zero'd out. Its purpose in training is to prevent
 * saturation of gradients (https://paperswithcode.com/method/relu).
 */
void relu(int in_channels, int in_h, int in_w,
          const f32 *x, // (in_channels,in_h,in_w)
          f32 *out // (in_channels,in_h,in_w)
) {
    for (int c = 0; c < in_channels; c++) {
        for (int i = 0; i < in_h; i++) {
            for (int j = 0; j < in_w; j++) {
                f32 val = x[I3(in_channels, in_h, in_w, c, i, j)];
                if (val > 0) {
                    out[I3(in_channels, in_h, in_w, c, i, j)] = val;
                } else {
                    out[I3(in_channels, in_h, in_w, c, i, j)] = 0;
                }
            }
        }
    }
}

/*
 * Find largest value.
 */
f32 max(int n, const f32 *x) {
    f32 maxval = -1e10f;
    for (int i = 0; i < n; i++) {
        if (x[i] > maxval) maxval = x[i];
    }
    return maxval;
}

f16 max_f16(int n, const f16 *x) {
    f16 maxval = -1e10f;
    for (int i = 0; i < n; i++) {
        if (x[i] > maxval) maxval = x[i];
    }
    return maxval;
}

/*
 * Find index of largest value.
 */
int argmax(int n, const f32 *x) {
    f32 maxval = -1e10f;
    int idx = -1;
    for (int i = 0; i < n; i++) {
        if (x[i] > maxval) {
            maxval = x[i];
            idx = i;
        }
    }
    return idx;
}

f32 sum(int n, const f32 *x) {
    f32 sumval = 0;
    for (int i = 0; i < n; i++) {
        sumval += x[i];
    }
    return sumval;
}

f16 sum_f16(int n, const f16 *x) {
    f16 sumval = 0;
    for (int i = 0; i < n; i++) {
        sumval += x[i];
    }
    return sumval;
}

/*
 * https://paperswithcode.com/method/softmax
 */
void softmax(int n,
             f32 *x,  // (n,)
             f32 *out // (n,)
) {
    f32 maxval = max(n, x);
    for (int i = 0; i < n; i++) {
        out[i] = (f32)exp(x[i] - maxval);
    }
    f32 sumval = sum(n, out);
    for (int i = 0; i < n; i++) {
        out[i] = out[i] / sumval;
    }
}

void softmax_f16(int n,
             f16 *x,  // (n,)
             f16 *out // (n,)
) {
    f16 maxval = max_f16(n, x);
    for (int i = 0; i < n; i++) {
        out[i] = (f16)exp(x[i] - maxval);
    }
    f16 sumval = sum_f16(n, out);
    for (int i = 0; i < n; i++) {
        out[i] = out[i] / sumval;
    }
}

/*
 * Down-sample an input array by a factor of 2 in width and
 * height, saving only the maximum value of the input map.
 *
 * https://paperswithcode.com/method/max-pooling
 */
void max_pool_2d(int in_channels, int in_h, int in_w,
                 const f32 *x, // (in_channels, in_h, in_w)
                 f32 *out // (in_channels, in_h/2, in_w/2)
) {
    int out_w = in_w / 2;
    int out_h = in_h / 2;
    for (int c = 0; c < in_channels; c++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                f32 max = -1e10f;
                for (int i2 = 0; i2 < 2; i2++) {
                    for (int j2 = 0; j2 < 2; j2++) {
                        f32 val = x[I3(in_channels, in_h, in_w, c, 2 * i + i2, 2 * j + j2)];
                        if (val > max) {
                            max = val;
                        }
                    }
                }
                out[I3(in_channels, out_h, out_w, c, i, j)] = max;
            }
        }
    }
}

void max_pool_2d_f16(int in_channels, int in_h, int in_w,
                 const f16 *x, // (in_channels, in_h, in_w)
                 f16 *out // (in_channels, in_h/2, in_w/2)
) {
    int out_w = in_w / 2;
    int out_h = in_h / 2;
    for (int c = 0; c < in_channels; c++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                f16 max = -1e10f;
                for (int i2 = 0; i2 < 2; i2++) {
                    for (int j2 = 0; j2 < 2; j2++) {
                        f16 val = x[I3(in_channels, in_h, in_w, c, 2 * i + i2, 2 * j + j2)];
                        if (val > max) {
                            max = val;
                        }
                    }
                }
                out[I3(in_channels, out_h, out_w, c, i, j)] = max;
            }
        }
    }
}

void batch_norm_2d(int in_channels, int in_h, int in_w,
                 const f32 *x, // (in_channels, in_h, in_w)
                 f32 *out // (in_channels, in_h/2, in_w/2)
) {
    int out_w = in_w / 2;
    int out_h = in_h / 2;
    for (int c = 0; c < in_channels; c++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                f32 max = -1e10f;
                for (int i2 = 0; i2 < 2; i2++) {
                    for (int j2 = 0; j2 < 2; j2++) {
                        f32 val = x[I3(in_channels, in_h, in_w, c, 2 * i + i2, 2 * j + j2)];
                        if (val > max) {
                            max = val;
                        }
                    }
                }
                out[I3(in_channels, out_h, out_w, c, i, j)] = max;
            }
        }
    }
}

// out = matmul(A, x) + y
void saxpy(int m, int n,
           const f32 *A,  // (m, n)
           const f32 *x,  // (n,)
           const f32 *y,  // (m,)
           f32 *out // (m,)
) {
    for (int i = 0; i < m; i++) {
        out[i] = 0;
        for (int j = 0; j < n; j++) {
            out[i] += A[I2(m, n, i, j)] * x[j];
        }
        out[i] += y[i];
    }
}

void saxpy_no_bias(int m, int n,
           const f32 *A,  // (m, n)
           const f32 *x,  // (n,)
           f32 *out // (m,)
) {
    for (int i = 0; i < m; i++) {
        out[i] = 0;
        for (int j = 0; j < n; j++) {
            out[i] += A[I2(m, n, i, j)] * x[j];
        }
    }
}

void saxpy_f16(int m, int n,
           const f32 *A,  // (m, n)
           const f16 *x,  // (n,)
           const f32 *y,  // (m,)
           f16 *out // (m,)
) {
    for (int i = 0; i < m; i++) {
        out[i] = (f16)0;
        for (int j = 0; j < n; j++) {
            out[i] += (f16)(A[I2(m, n, i, j)]) * x[j];
        }
        out[i] += (f16)(y[i]);
    }
}

void relu_f16(
        int n,
        const f16 *x, // (n)
        f16 *out // (n)
        ) {
    for (int i = 0; i < n; i++) {
        f16 val = x[i];
        if (val > 0) {
            out[i] = val;
        } else {
            out[i] = 0;
        }
    }
}

void relu_32K_f16(
        const f16 *x, // (32768)
        f16 *out // (32768)
        ) {
    for (int i = 0; i < 32768; i++) {
        f16 val = x[i];
        if (val > 0) {
            out[i] = val;
        } else {
            out[i] = 0;
        }
    }
}

void pad_32K_copy(
        int old_size,
        const f32 *x, // (old_size)
        f32 *out // (32768)
        ) {
    for (int i = 0; i < old_size; i++) {
        out[i] = x[i];
    }
}

void section_32K_copy(
        int new_size,
        const f32 *x, // (32768)
        f32 *out // (new_size)
        ) {
    for (int i = 0; i < new_size; i++) {
        out[i] = x[i];
    }
}

void cast_f32_f16(
        int n,
        const f32 *x, // (n)
        f16 *out // (n)
        ) {
    for (int i = 0; i < n; i++) {
        out[i] = x[i];
    }
}

void cast_32K_f32_f16(
        const f32 *x, // (32768)
        f16 *out // (32768)
        ) {
    for (int i = 0; i < 32768; i++) {
        out[i] = x[i];
    }
}

void cast_f16_f32(
        int n,
        const f16 *x, // (n)
        f32 *out // (n)
        ) {
    for (int i = 0; i < n; i++) {
        out[i] = x[i];
    }
}

void cast_32K_f16_f32(
        const f16 *x, // (32768)
        f32 *out // (32768)
        ) {
    for (int i = 0; i < 32768; i++) {
        out[i] = x[i];
    }
}
