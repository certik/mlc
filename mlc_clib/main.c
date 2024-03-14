#include <stdio.h>
#include <string.h>

#include "kernels.h"
#include "display.h"
#include "gguf.h"

/*
 * On Mac, run "leaks" on the executable.
 *
 * leaks --atExit -- ./mlc_clib
 */

#define I2(D1, D2, i, j) ((i)*(D2)+(j))
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

int main() {
    // Follow the instructions in the
    // mlc/examples/mnist/READMEmd, namely
    //
    //    python mnist-tf.py train mnist-cnn-model
    //    python mnist-tf.py convert mnist-cnn-model
    //    python mnist-tf.py convert_tests mnist-tests
    //
    // The `mnist-tf` script will
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
                        = (f32)(pDigits_u8[i*height*width+j*width+k]) / 255.f;
                }
            }
        }

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

    // (28, 28)
    f32 *out = pDigits + digit_idx*28*28;

    print_A(&out[I2(28, 28, 14, 13)]);
    print_A(kernel1);

    // (32, 1, 3, 3)
    f32 *kernel1_ = malloc(32*1*3*3*sizeof(f32));
    transpose(3, 3, 1, 32, kernel1, 3, 2, 0, 1, kernel1_);
    print_A(kernel1_);

    return 0;
}

