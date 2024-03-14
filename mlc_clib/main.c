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


void print_A(f32 *A) {
    for (int i = 0; i < 10; i++) {
        printf("%f ", A[i]);
    }
    printf("\n");
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

    // Read the model file

    struct gguf_context ctx;
    r = gguf_read("../examples/mnist/mnist-cnn-model.gguf", &ctx);
    if (r != 0) {
        printf("GGUF file not read; return code = %d\n", r);
        return r;
    }
    /*
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
    */

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

    for (int digit_idx_i=0; digit_idx_i < 13; digit_idx_i++) {
        int digit_idx = 4212 + digit_idx_i;
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

            printf("Reference value: %u; digit index %d\n",
                    digit_ref_bytes[digit_idx], digit_idx);
        }


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

        printf("Digit probabilities:\n");
        print_A(out9);
        printf("Inferred value: %d\n", argmax(10, out9));
    }

    return 0;
}
