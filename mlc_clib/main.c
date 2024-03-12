#include <stdio.h>
#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "kernels.h"

void test_1D_MLCA();
void test_2D_MLCA();
void test_3D_MLCA();
void test_4D_MLCA();

/*
 * On Mac, run "leaks" on the executable.
 *
 * leaks --atExit -- ./mlc_clib
 */


#define GGUF_MAGIC "GGUF"
#define GGUF_MAX_DIMS           4

struct gguf_str {
    uint64_t n;  // GGUFv2
    char * data;
};

enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT,       // marks the end of the enum
};

    enum ggml_type {
        GGML_TYPE_F32  = 0,
        GGML_TYPE_F16  = 1,
        GGML_TYPE_Q4_0 = 2,
        GGML_TYPE_Q4_1 = 3,
        // GGML_TYPE_Q4_2 = 4, support has been removed
        // GGML_TYPE_Q4_3 (5) support has been removed
        GGML_TYPE_Q5_0 = 6,
        GGML_TYPE_Q5_1 = 7,
        GGML_TYPE_Q8_0 = 8,
        GGML_TYPE_Q8_1 = 9,
        // k-quantizations
        GGML_TYPE_Q2_K = 10,
        GGML_TYPE_Q3_K = 11,
        GGML_TYPE_Q4_K = 12,
        GGML_TYPE_Q5_K = 13,
        GGML_TYPE_Q6_K = 14,
        GGML_TYPE_Q8_K = 15,
        GGML_TYPE_IQ2_XXS = 16,
        GGML_TYPE_IQ2_XS  = 17,
        GGML_TYPE_IQ3_XXS = 18,
        GGML_TYPE_IQ1_S   = 19,
        GGML_TYPE_IQ4_NL  = 20,
        GGML_TYPE_IQ3_S   = 21,
        GGML_TYPE_IQ2_S   = 22,
        GGML_TYPE_IQ4_XS  = 23,
        GGML_TYPE_I8,
        GGML_TYPE_I16,
        GGML_TYPE_I32,
        GGML_TYPE_COUNT,
    };


union gguf_value {
    uint8_t  uint8;
    int8_t   int8;
    uint16_t uint16;
    int16_t  int16;
    uint32_t uint32;
    int32_t  int32;
    float    float32;
    uint64_t uint64;
    int64_t  int64;
    double   float64;
    bool     bool_;

    struct gguf_str str;

    struct {
        enum gguf_type type;

        uint64_t n;  // GGUFv2
        void * data;
    } arr;
};

struct gguf_kv {
    struct gguf_str key;

    enum  gguf_type  type;
    union gguf_value value;
};

struct gguf_header {
    char magic[4];

    uint32_t version;
    uint64_t n_tensors; // GGUFv2
    uint64_t n_kv;      // GGUFv2
};

struct gguf_context {
    struct gguf_header header;

    struct gguf_kv          * kv;
    struct gguf_tensor_info * infos;

    size_t alignment;
    size_t offset;    // offset of `data` from beginning of file
    size_t size;      // size of `data` in bytes

    //uint8_t * padding;
    void * data;
};

struct gguf_tensor_info {
    struct gguf_str name;
    uint32_t n_dims;
    uint64_t ne[GGUF_MAX_DIMS];
    enum ggml_type type;
    uint64_t offset; // offset from start of `data`, must be a multiple of `ALIGNMENT`
    // for writing API
    const void * data;
    size_t size;
};


static bool gguf_fread_el(FILE * file, void * dst, size_t size,
        size_t * offset)
{
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    return n == size;
}


int gguf_read(const char *fname, struct gguf_context *ctx)
{
    FILE * file = fopen(fname, "rb");
    if (!file) {
        fprintf(stderr, "%s: failed to open `%s`\n", __func__, fname);
        return 1;
    }
    size_t offset = 0;

    // Magic
    {
        char magic[4];
        gguf_fread_el(file, &magic, sizeof(magic), &offset);
        for (uint32_t i = 0; i < sizeof(magic); i++) {
            if (magic[i] != GGUF_MAGIC[i]) {
                fprintf(stderr, "%s: invalid magic characters '%c%c%c%c'\n", __func__, magic[0], magic[1], magic[2], magic[3]);
                fclose(file);
                return 2;
            }
        }
        strncpy(ctx->header.magic, magic, 4);
    }

    // Header
    {
        bool ok = true;

        ctx->kv    = NULL;
        ctx->infos = NULL;
        ctx->data  = NULL;

        ok = ok && gguf_fread_el(file, &ctx->header.version,
                sizeof(ctx->header.version),   &offset);
        ok = ok && gguf_fread_el(file, &ctx->header.n_tensors,
                sizeof(ctx->header.n_tensors), &offset);
        ok = ok && gguf_fread_el(file, &ctx->header.n_kv,
                sizeof(ctx->header.n_kv),      &offset);

        if (ctx->header.version == 1) {
            fprintf(stderr, "%s: GGUFv1 is no longer supported. please use a more up-to-date version\n", __func__);
            fclose(file);
            return 3;
        }

        // sanity-checks to prevent from integer/buffer overflows
        ok = ok && (ctx->header.n_tensors
                < (SIZE_MAX/2)/sizeof(struct gguf_tensor_info));
        ok = ok && (ctx->header.n_kv
                < (SIZE_MAX/2)/sizeof(struct gguf_kv));

        if (!ok) {
            fprintf(stderr, "%s: failed to read header\n", __func__);
            fclose(file);
            return 4;
        }
    }
    return 0;
}

int main() {
    struct gguf_context ctx;
    int r = gguf_read("examples/mnist/mnist-cnn-model.gguf", &ctx);
    if (r == 0) {
        printf("File read successfuly.\n");
        printf("Magic:'%c%c%c%c'\n", ctx.header.magic[0], ctx.header.magic[1],
                ctx.header.magic[2], ctx.header.magic[3]);
        printf("Version: %d\n", ctx.header.version);
        printf("Number of arrays: %llu\n", ctx.header.n_tensors);
        printf("Number of kv pairs: %llu\n", ctx.header.n_kv);
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

