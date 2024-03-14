#include <stdio.h>
#include <string.h>
#include <math.h>

#include "kernels.h"
#include "display.h"
#include "gguf.h"

/*
 * On Mac, run "leaks" on the executable.
 *
 * leaks --atExit -- ./mlc_clib
 */

#define I2(D1, D2, i, j) ((i)*(D2)+(j))
#define I3(D1, D2, D3, i, j, k) ((i)*(D3)*(D2)+(j)*(D3)+(k))
#define I4(D1, D2, D3, D4, i, j, k, l) ((i)*(D4)*(D3)*(D2)+(j)*(D4)*(D3)+(k)*(D4)+(l))

void print_A(f32 *A) {
    for (int i = 0; i < 10; i++) {
        printf("%f ", A[i]);
    }
    printf("\n");
}

// A -> B
void transpose(int n1, int n2, int n3, int n4, f32 *A,
            int t1, int t2, int t3, int t4, f32 *B) {
    for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
    for (int k = 0; k < n3; k++) {
    for (int l = 0; l < n4; l++) {
        // TODO: for now the transposition is hardwired to (3,2,0,1)
        B[I4(n4,n3,n1,n2,l,k,i,j)] = A[I4(n1,n2,n3,n4,i,j,k,l)];
    }}}}
}

// (h, w)
// Note: this is appending (+=) to `out`, must be initialized.
void conv2d_kernel(int kernel_size, int in_h, int in_w,
        f32 *weight, // (3,3)
        f32 *x, // (in_h,in_w)
        f32 *out // (out_w,out_h)
        )
{
    int out_w = in_w - (kernel_size-1);
    int out_h = in_h - (kernel_size-1);
    for (int h = 0; h < out_h; h++) {
    for (int w = 0; w < out_w; w++) {
            for (int i=0; i<kernel_size; i++) {
            for (int j=0; j<kernel_size; j++) {
                out[I2(out_h,out_w,h,w)] += weight[I2(3,3,i,j)] * \
                    x[I2(in_h,in_w,h+i,w+j)];
            }}
    }}
}

// (batch, channel, h, w)
void conv2d(int in_channels, int out_channels, int kernel_size,
        int in_h, int in_w,
        f32 *weight, // (out_channels,in_channels,3,3)
        f32 *bias, // (out_channels,)
        f32 *x, // (in_channels,in_h,in_w)
        f32 *out // (out_channels,out_h,out_w)
        )
{
    int out_w = in_w - (kernel_size-1);
    int out_h = in_h - (kernel_size-1);
    f32 s[out_h*out_w];
    for (int c = 0; c < out_channels; c++) {
        for (int i=0; i<out_h; i++) {
        for (int j=0; j<out_w; j++) {
            s[I2(out_h,out_w,i,j)] = 0;
        }}
        for (int k = 0; k < in_channels; k++) {
            conv2d_kernel(kernel_size, in_h, in_w,
                    &weight[I4(out_channels,in_channels,3,3,c,k,0,0)],
                    &x[I3(in_channels,in_h,in_w,k,0,0)],
                    s);
        }
        for (int i=0; i<out_h; i++) {
        for (int j=0; j<out_w; j++) {
            out[I3(out_channels,out_h,out_w,c,i,j)] = bias[c] + \
                s[I2(out_h,out_w,i,j)];
        }}
    }
}

// (channel, h, w)
void relu(int in_channels, int in_h, int in_w,
        f32 *x, // (in_channels,in_h,in_w)
        f32 *out // (in_channels,in_h,in_w)
        )
{
    for (int c = 0; c < in_channels; c++) {
        for (int i=0; i<in_h; i++) {
        for (int j=0; j<in_w; j++) {
            f32 val = x[I3(in_channels,in_h,in_w,c,i,j)];
            if (val > 0) {
                out[I3(in_channels,in_h,in_w,c,i,j)] = val;
            } else {
                out[I3(in_channels,in_h,in_w,c,i,j)] = 0;
            }
        }}
    }
}

f32 max(int n, f32 *x)
{
    f32 maxval = -1e10;
    for (int i=0; i<n; i++) {
        if (x[i] > maxval) maxval = x[i];
    }
    return maxval;
}

f32 sum(int n, f32 *x)
{
    f32 sumval = 0;
    for (int i=0; i<n; i++) {
        sumval += x[i];
    }
    return sumval;
}

void softmax(int n,
        f32 *x,  // (n,)
        f32 *out // (n,)
        )
{
    f32 maxval = max(n, x);
    for (int i=0; i<n; i++) {
        out[i] = exp(x[i] - maxval);
    }
    f32 sumval = sum(n, out);
    for (int i=0; i<n; i++) {
        out[i] = out[i] / sumval;
    }
}

void max_pool_2d(int in_channels, int in_h, int in_w,
        f32 *x, // (in_channels,in_h,in_w)
        f32 *out // (in_channels,in_h/2,in_w/2)
        )
{
    int out_w = in_w/2;
    int out_h = in_h/2;
    for (int c = 0; c < in_channels; c++) {
        for (int i=0; i<out_h; i++) {
        for (int j=0; j<out_w; j++) {
            f32 max = -1e10;
            for (int i2=0; i2<2; i2++) {
            for (int j2=0; j2<2; j2++) {
                f32 val = x[I3(in_channels,in_h,in_w,c,2*i+i2,2*j+j2)];
                if (val > max) max = val;
            }}
            out[I3(in_channels,out_h,out_w,c,i,j)] = max;
        }}
    }
}

// out = matmul(A, x) + y
void saxpy(int m, int n,
        f32 *A,  // (m, n)
        f32 *x,  // (n,)
        f32 *y,  // (m,)
        f32 *out // (m,)
        )
{
    for (int i=0; i<m; i++) {
        out[i] = 0;
        for (int j=0; j<n; j++) {
            out[i] += A[I2(m,n,i,j)] * x[j];
        }
        out[i] += y[i];
    }
}

int main() {
    // Follow the instructions in the README. The `mnist-tf` script will
    // generate two GGUF files:
    // * mnist-cnn-model.gguf (trained ML weights)
    // * mnist-tests.gguf (10,000 MNIST test images)


    // Read test images

    struct gguf_context ctx_test;
    int r = gguf_read("../examples/mnist/mnist-tests.gguf", &ctx_test);
    if (r != 0) {
        printf("GGUF file not read; return code = %d\n", r);
        return r;
    }

    int digit_idx = 4212;
    f32 *pDigits;
    uint8_t *digit_ref_bytes;

    {
        assert(ctx_test.infos[0].ne[0] == 28);
        assert(ctx_test.infos[0].ne[1] == 28);
        assert(ctx_test.infos[0].ne[2] == 10000);
        assert(ctx_test.infos[0].type == GGML_TYPE_I8);
        uint8_t *pDigits_u8 = (uint8_t *) (ctx_test.data + ctx_test.infos[0].offset);

        int ndigits = 10000;
        int width = 28;
        int height = 28;
        size_t digit_size = width * height;
        size_t nitems = ndigits * digit_size;

        size_t digits_size = nitems * sizeof(f32);  // plural
        pDigits = malloc(digits_size);
        for (int i = 0; i < ndigits; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    pDigits[i*height*width+j*width+k]
                        = (f32)(pDigits_u8[i*height*width+j*width+k]) / 255.;
                }
            }
        }

        // Draw 4201'th digit in the file.
        draw_digit(pDigits + (digit_idx * digit_size));
    }
    {
        assert(ctx_test.infos[1].ne[0] == 10000);
        assert(ctx_test.infos[1].type == GGML_TYPE_I8);
        digit_ref_bytes = (uint8_t *) (ctx_test.data + ctx_test.infos[1].offset);

        size_t ndigits = 10000;
        assert(sizeof(uint8_t) == 1);
        assert(digit_ref_bytes != NULL);

        printf("Reference value: %u\n", digit_ref_bytes[digit_idx]);
    }

    // Read the model file

    struct gguf_context ctx;
    r = gguf_read("../examples/mnist/mnist-cnn-model.gguf", &ctx);
    if (r != 0) {
        printf("GGUF file not read; return code = %d\n", r);
        return r;
    }
    printf("File read successfuly.\n");
    printf("Magic:'%c%c%c%c'\n", ctx.header.magic[0], ctx.header.magic[1],
           ctx.header.magic[2], ctx.header.magic[3]);
    printf("Version: %d\n", ctx.header.version);
    printf("Number of kv pairs: %llu\n", ctx.header.n_kv);
    for (size_t i = 0; i < ctx.header.n_kv; i++) {
        char tmp[256];
        strncpy(tmp, ctx.kv[i].key.data, ctx.kv[i].key.n);
        tmp[ctx.kv[i].key.n] = 0;
        char v[256] = "";
        if (ctx.kv[i].type == GGUF_TYPE_STRING) {
            strncpy(v, ctx.kv[i].value.str.data, ctx.kv[i].value.str.n);
            v[ctx.kv[i].value.str.n] = 0;
        }
        printf("    %zu: %s = %s\n", i, tmp, v);
    }
    printf("Data Offset: %zu\n", ctx.offset);
    printf("Data Size:   %zu\n", ctx.size);
    printf("Number of arrays: %llu\n", ctx.header.n_tensors);
    for (size_t i = 0; i < ctx.header.n_tensors; i++) {
        char tmp[256];
        strncpy(tmp, ctx.infos[i].name.data, ctx.infos[i].name.n);
        tmp[ctx.infos[i].name.n] = 0;
        printf("    %zu: %s ndim=%d shape=(%llu,%llu,%llu,%llu) type=%s offset=%llu\n",
               i, tmp,
               ctx.infos[i].n_dims,
               ctx.infos[i].ne[0], ctx.infos[i].ne[1],
               ctx.infos[i].ne[2], ctx.infos[i].ne[3],
               ggml_type_name(ctx.infos[i].type),
               ctx.infos[i].offset
        );
        if (ctx.infos[i].type == GGML_TYPE_F32) {
            if (ctx.infos[i].ne[0] >= 5) {
                printf("        First few elements of f32 array:\n");
                float *A = (float *) (ctx.data + ctx.infos[i].offset);
                printf("            [%f, %f, %f, %f, %f]\n",
                       A[0], A[1], A[2], A[3], A[4]);
            }
        }
    }

    // (3, 3, 1, 32)
    f32 *kernel1 = (f32*) (ctx.data + ctx.infos[0].offset);
    // (32,)
    f32 *bias1 = (f32*) (ctx.data + ctx.infos[1].offset);
    // (3, 3, 32, 64)
    f32 *kernel2 = (f32*) (ctx.data + ctx.infos[2].offset);
    // (32,)
    f32 *bias2 = (f32*) (ctx.data + ctx.infos[3].offset);
    // (1600, 10)
    f32 *dense_w = (f32*) (ctx.data + ctx.infos[4].offset);
    // (10,)
    f32 *dense_b = (f32*) (ctx.data + ctx.infos[5].offset);

    // (28, 28)
    f32 *out = pDigits + digit_idx*28*28;

    // Conv2D
    // (32, 1, 3, 3)
    f32 *kernel1_ = malloc(32*1*3*3*sizeof(f32));
    transpose(3, 3, 1, 32, kernel1, 3, 2, 0, 1, kernel1_);
    f32 *out2 = malloc(32*26*26*sizeof(f32));
    conv2d(1, 32, 3,
        28, 28,
        kernel1_, // (32, 1, 3, 3)
        bias1, // (32,)
        out, // (1, 28, 28)
        out2 // (32, 26, 26)
        );

    // ReLU
    f32 *out3 = malloc(32*26*26*sizeof(f32));
    relu(32, 26, 26,
        out2, // (32, 26, 26)
        out3  // (32, 26, 26)
        );

    // MaxPool2D
    f32 *out4 = malloc(32*13*13*sizeof(f32));
    max_pool_2d(32, 26, 26,
        out3, // (32, 26, 26)
        out4  // (32, 13, 13)
        );

    // Conv2D
    // (32, 1, 3, 3)
    f32 *kernel2_ = malloc(32*64*3*3*sizeof(f32));
    transpose(3, 3, 32, 64, kernel2, 3, 2, 0, 1, kernel2_);
    f32 *out5 = malloc(64*11*11*sizeof(f32));
    conv2d(32, 64, 3,
        13, 13,
        kernel2_, // (32, 64, 3, 3)
        bias2, // (32,)
        out4, // (32, 13, 13)
        out5 // (64, 11, 11)
        );

    // ReLU
    f32 *out6 = malloc(64*11*11*sizeof(f32));
    relu(64, 11, 11,
        out5, // (64, 11, 11)
        out6  // (64, 11, 11)
        );

    // MaxPool2D
    f32 *out7 = malloc(64*5*5*sizeof(f32));
    max_pool_2d(64, 11, 11,
        out6, // (64, 11, 11)
        out7  // (64, 5, 5)
        );

    // Flatten: out7 (64, 5, 5) -> (1600,)

    // Linear
    f32 *dense_w_ = malloc(64*5*5*10*sizeof(f32));
    transpose(5, 5, 64, 10, dense_w, 3, 2, 0, 1, dense_w_);
    f32 *out8 = malloc(10*sizeof(f32));
    saxpy(10, 1600,
            dense_w_, // (10, 1600)
            out7,     // (1600,)
            dense_b,  // (10,)
            out8      // (10,)
        );

    // Softmax
    f32 *out9 = malloc(10*sizeof(f32));
    softmax(10,
            out8, // (10,)
            out9  // (10,)
        );

    print_A(out9);

    return 0;
}

