#include <stdio.h>
#include <string.h>

#include "inference-generated.h"
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
    r = gguf_read("../examples/mnist/mnist-cnn-model.gguf", &ctx);
    if (r != 0) {
        printf("GGUF file not read; return code = %d\n", r);
        return r;
    }
    // (32, 1, 3, 3) -- row major, C-order, (C_out, C_in, H, W)
    f32 *kernel1 = (f32*) (ctx.data + ctx.infos[0].offset);
    // (32,)
    f32 *bias1 = (f32*) (ctx.data + ctx.infos[1].offset);
    // (64, 32, 3, 3)
    f32 *kernel2 = (f32*) (ctx.data + ctx.infos[2].offset);
    // (64,)
    f32 *bias2 = (f32*) (ctx.data + ctx.infos[3].offset);
    // (10, 1600)
    f32 *dense_w = (f32*) (ctx.data + ctx.infos[4].offset);
    // (10,)
    f32 *dense_b = (f32*) (ctx.data + ctx.infos[5].offset);

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
                kernel1, bias1,
                kernel2, bias2,
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
