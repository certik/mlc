#include <stdio.h>
#include <string.h>

#include "inference-generated2.h"
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
    // mlc/examples/mnist/README.md, namely
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
    assert(ctx_test.infos[0].ne[0] == 28);
    assert(ctx_test.infos[0].ne[1] == 28);
    assert(ctx_test.infos[0].ne[2] == 10000);
    assert(ctx_test.infos[0].type == GGML_TYPE_I8);
    uint8_t *pDigits_u8 = (uint8_t *) (ctx_test.data + ctx_test.infos[0].offset);
    size_t digit_w = ctx_test.infos[0].ne[0];
    size_t digit_h = ctx_test.infos[0].ne[1];
    int ndigits = ctx_test.infos[0].ne[2];
    f32 *pDigits = malloc(ndigits * digit_w * digit_h * sizeof(f32));
    // Convert test data from u8 to f32
    for (int i = 0; i < ndigits * digit_w * digit_h; i++) {
        pDigits[i] = (f32)(pDigits_u8[i]) / 255.f;
    }
    assert(ctx_test.infos[1].ne[0] == 10000);
    assert(ctx_test.infos[1].type == GGML_TYPE_I8);
    uint8_t *digit_ref_bytes = (uint8_t *) (ctx_test.data + ctx_test.infos[1].offset);

    // Read the model file
    struct gguf_context ctx;
    r = gguf_read("../examples/mnist/mnist-cnn-beautiful-model.gguf", &ctx);
    if (r != 0) {
        printf("GGUF file not read; return code = %d\n", r);
        return r;
    }
    f32 *kernel1 = (f32*) (ctx.data + ctx.infos[0].offset);
    f32 *bias1 = (f32*) (ctx.data + ctx.infos[1].offset);
    f32 *kernel2 = (f32*) (ctx.data + ctx.infos[2].offset);
    f32 *bias2 = (f32*) (ctx.data + ctx.infos[3].offset);
    f32 *batchnorm1_gamma = (f32*) (ctx.data + ctx.infos[4].offset);
    f32 *batchnorm1_beta = (f32*) (ctx.data + ctx.infos[5].offset);
    f32 *batchnorm1_moving_mean = (f32*) (ctx.data + ctx.infos[6].offset);
    f32 *batchnorm1_moving_variance = (f32*) (ctx.data + ctx.infos[7].offset);

    f32 *kernel3 = (f32*) (ctx.data + ctx.infos[8].offset);
    f32 *bias3 = (f32*) (ctx.data + ctx.infos[9].offset);
    f32 *kernel4 = (f32*) (ctx.data + ctx.infos[10].offset);
    f32 *bias4 = (f32*) (ctx.data + ctx.infos[11].offset);
    f32 *batchnorm2_gamma = (f32*) (ctx.data + ctx.infos[12].offset);
    f32 *batchnorm2_beta = (f32*) (ctx.data + ctx.infos[13].offset);
    f32 *batchnorm2_moving_mean = (f32*) (ctx.data + ctx.infos[14].offset);
    f32 *batchnorm2_moving_variance = (f32*) (ctx.data + ctx.infos[15].offset);

    f32 *dense_w = (f32*) (ctx.data + ctx.infos[16].offset);
    f32 *dense_b = (f32*) (ctx.data + ctx.infos[17].offset);


    for (int digit_idx_i=0; digit_idx_i < 11; digit_idx_i++) {
        int digit_idx = 4213 + digit_idx_i;
        draw_digit(pDigits + (digit_idx * digit_w * digit_h));
        int reference_value = digit_ref_bytes[digit_idx];
        printf("Reference value: %u; digit index %d\n",
                reference_value, digit_idx);

        // (28, 28)
        f32 *in = pDigits + digit_idx*28*28;
        f32 *out = malloc(10*sizeof(f32));

        inference(in, out,
                kernel1, bias1, kernel2, bias2,
                batchnorm1_gamma, batchnorm1_beta,
                batchnorm1_moving_mean, batchnorm1_moving_variance,
                kernel3, bias3, kernel4, bias4,
                batchnorm2_gamma, batchnorm2_beta,
                batchnorm2_moving_mean, batchnorm2_moving_variance,
                dense_w, dense_b);

        printf("Digit probabilities:\n");
        print_A(out);
        int inferred_value = argmax(10, out);
        printf("Inferred value: %d\n", inferred_value);
        if (inferred_value != reference_value) {
            printf("FAIL: Inferred value does not match reference value.\n");
            return 1;
        }
    }

    return 0;
}
