#include <stdio.h>
#include <string.h>

#include "kernels.h"
#include "gguf.h"

/*
 * On Mac, run "leaks" on the executable.
 *
 * leaks --atExit -- ./mlc_clib
 */


int main() {
    struct gguf_context ctx;
    int r = gguf_read("examples/mnist/mnist-cnn-model.gguf", &ctx);
    if (r == 0) {
        printf("File read successfuly.\n");
        printf("Magic:'%c%c%c%c'\n", ctx.header.magic[0], ctx.header.magic[1],
                ctx.header.magic[2], ctx.header.magic[3]);
        printf("Version: %d\n", ctx.header.version);
        printf("Number of kv pairs: %llu\n", ctx.header.n_kv);
        for (size_t i=0; i < ctx.header.n_kv; i++) {
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
        for (size_t i=0; i < ctx.header.n_tensors; i++) {
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
                    float *A = (float*) (ctx.data + ctx.infos[i].offset);
                    printf("            [%f, %f, %f, %f, %f]\n",
                            A[0], A[1], A[2], A[3], A[4]);
                }
            }
        }
    }
    return r;
}

void test_1D_MLCA() {
    int64_t cols = 42;
    pMLCA pmlca = alloc1D(cols);
    for (int64_t col = 0; col < cols; col++) {
        put1D(pmlca, col, ((f32)(col)));
    }
    dumpMLCA(pmlca);
    freeMLCA(pmlca);
}

void test_2D_MLCA() {
    int64_t cols = 7;
    int64_t rows = 6;
    pMLCA pmlca = alloc2D(rows, cols);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            put2D(pmlca, row, col,
                  ((f32)((row + 1) * (col + 1))));
        }
    }
    dumpMLCA(pmlca);
    freeMLCA(pmlca);
}

void test_3D_MLCA() {
    int64_t sheets = 4;
    int64_t cols = 7;
    int64_t rows = 6;
    pMLCA pmlca = alloc3D(sheets, rows, cols);
    for (int sheet = 0; sheet < sheets; sheet++) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                put3D(pmlca,
                      sheet, row, col,
                      ((f32)((sheet + 1) * (row + 1) * (col + 1))));
            }
        }
    }
    dumpMLCA(pmlca);
    freeMLCA(pmlca);
}

void test_4D_MLCA() {
    int64_t blocks = 3;
    int64_t sheets = 4;
    int64_t cols = 7;
    int64_t rows = 6;
    pMLCA pmlca = alloc4D(blocks, sheets, rows, cols);
    for (int block = 0; block < blocks; block++) {
        for (int sheet = 0; sheet < sheets; sheet++) {
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    put4D(pmlca, block, sheet, row, col,
                          ((f32) ((block + 1) * (sheet + 1)
                          * (row + 1) * (col + 1))));
                }
            }
        }
    }
    dumpMLCA(pmlca);
    freeMLCA(pmlca);
}

