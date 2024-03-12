#include <stdio.h>
#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

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
#define GGUF_DEFAULT_ALIGNMENT 32
#define GGUF_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))



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

static const size_t GGUF_TYPE_SIZE[GGUF_TYPE_COUNT] = {                                                  
    [GGUF_TYPE_UINT8]   = sizeof(uint8_t),                                                                                        
    [GGUF_TYPE_INT8]    = sizeof(int8_t),            
    [GGUF_TYPE_UINT16]  = sizeof(uint16_t),                  
    [GGUF_TYPE_INT16]   = sizeof(int16_t),          
    [GGUF_TYPE_UINT32]  = sizeof(uint32_t),                            
    [GGUF_TYPE_INT32]   = sizeof(int32_t),                      
    [GGUF_TYPE_FLOAT32] = sizeof(float),                                                                                   
    [GGUF_TYPE_BOOL]    = sizeof(bool),                                      
    [GGUF_TYPE_STRING]  = sizeof(struct gguf_str),
    [GGUF_TYPE_UINT64]  = sizeof(uint64_t),                                          
    [GGUF_TYPE_INT64]   = sizeof(int64_t),    
    [GGUF_TYPE_FLOAT64] = sizeof(double),         
    [GGUF_TYPE_ARRAY]   = 0, // undefined                         
};

static size_t gguf_type_size(enum gguf_type type) {
    return GGUF_TYPE_SIZE[type];
}


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

typedef struct {
    const char      * type_name;
    int               blck_size;
    size_t            type_size;
    bool              is_quantized;
    int64_t           nrows; // number of rows to process simultaneously;
} ggml_type_traits_t;


static const ggml_type_traits_t type_traits[GGML_TYPE_COUNT] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_size                = 1,
        .type_size                = sizeof(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_size                = 1,
        .type_size                = sizeof(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = sizeof(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = sizeof(float),
        .is_quantized             = false,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = sizeof(float)/2,
        .is_quantized             = false,
        .nrows                    = 1,
    },
};

int ggml_blck_size(enum ggml_type type) {
    return type_traits[type].blck_size;
}

size_t ggml_type_size(enum ggml_type type) {
    return type_traits[type].type_size;
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    assert(ne % ggml_blck_size(type) == 0);
    return ggml_type_size(type)*ne/ggml_blck_size(type);
}

const char * ggml_type_name(enum ggml_type type) {
    return type_traits[type].type_name;
}

static bool gguf_fread_el(FILE * file, void * dst, size_t size,
        size_t * offset)
{
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    return n == size;
}

static bool gguf_fread_str(FILE * file, struct gguf_str * p, size_t * offset) {
    p->n    = 0;
    p->data = NULL;
    bool ok = true;
    ok = ok && gguf_fread_el(file, &p->n, sizeof(p->n), offset);
    // early exit if string length is invalid, prevents from integer overflow
    if (p->n == SIZE_MAX) {
        fprintf(stderr, "%s: invalid string length (%llu)\n", __func__, p->n);
        return false;
    }
    p->data = calloc(p->n + 1, 1);
    ok = ok && gguf_fread_el(file,  p->data, p->n, offset);
    return ok;
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

    // kv pairs
    {
        bool ok = true;

        ctx->kv = malloc(ctx->header.n_kv * sizeof(struct gguf_kv));

        for (uint64_t i = 0; i < ctx->header.n_kv; ++i) {
            struct gguf_kv * kv = &ctx->kv[i];

            //fprintf(stderr, "%s: reading kv %d\n", __func__, i);

            ok = ok && gguf_fread_str(file, &kv->key,                    &offset);
            ok = ok && gguf_fread_el (file, &kv->type, sizeof(kv->type), &offset);

            //fprintf(stderr, "%s: reading kv with key %s\n", __func__, kv->key.data);

            switch (kv->type) {
                case GGUF_TYPE_UINT8:   ok = ok && gguf_fread_el (file, &kv->value.uint8,   sizeof(kv->value.uint8),   &offset); break;
                case GGUF_TYPE_INT8:    ok = ok && gguf_fread_el (file, &kv->value.int8,    sizeof(kv->value.int8),    &offset); break;
                case GGUF_TYPE_UINT16:  ok = ok && gguf_fread_el (file, &kv->value.uint16,  sizeof(kv->value.uint16),  &offset); break;
                case GGUF_TYPE_INT16:   ok = ok && gguf_fread_el (file, &kv->value.int16,   sizeof(kv->value.int16),   &offset); break;
                case GGUF_TYPE_UINT32:  ok = ok && gguf_fread_el (file, &kv->value.uint32,  sizeof(kv->value.uint32),  &offset); break;
                case GGUF_TYPE_INT32:   ok = ok && gguf_fread_el (file, &kv->value.int32,   sizeof(kv->value.int32),   &offset); break;
                case GGUF_TYPE_FLOAT32: ok = ok && gguf_fread_el (file, &kv->value.float32, sizeof(kv->value.float32), &offset); break;
                case GGUF_TYPE_UINT64:  ok = ok && gguf_fread_el (file, &kv->value.uint64,  sizeof(kv->value.uint64),  &offset); break;
                case GGUF_TYPE_INT64:   ok = ok && gguf_fread_el (file, &kv->value.int64,   sizeof(kv->value.int64),   &offset); break;
                case GGUF_TYPE_FLOAT64: ok = ok && gguf_fread_el (file, &kv->value.float64, sizeof(kv->value.float64), &offset); break;
                case GGUF_TYPE_BOOL:    ok = ok && gguf_fread_el (file, &kv->value.bool_,   sizeof(kv->value.bool_),   &offset); break;
                case GGUF_TYPE_STRING:  ok = ok && gguf_fread_str(file, &kv->value.str,                                &offset); break;
                case GGUF_TYPE_ARRAY:
                    {
                        ok = ok && gguf_fread_el(file, &kv->value.arr.type, sizeof(kv->value.arr.type), &offset);
                        ok = ok && gguf_fread_el(file, &kv->value.arr.n,    sizeof(kv->value.arr.n),    &offset);

                        switch (kv->value.arr.type) {
                            case GGUF_TYPE_UINT8:
                            case GGUF_TYPE_INT8:
                            case GGUF_TYPE_UINT16:
                            case GGUF_TYPE_INT16:
                            case GGUF_TYPE_UINT32:
                            case GGUF_TYPE_INT32:
                            case GGUF_TYPE_FLOAT32:
                            case GGUF_TYPE_UINT64:
                            case GGUF_TYPE_INT64:
                            case GGUF_TYPE_FLOAT64:
                            case GGUF_TYPE_BOOL:
                                {
                                    // prevent from integer overflow in the malloc below
                                    if (kv->value.arr.n >= SIZE_MAX/gguf_type_size(kv->value.arr.type)) {
                                        fprintf(stderr, "%s: array size is too large (%llu)\n", __func__, kv->value.arr.n);
                                        fclose(file);
                                        return 5;
                                    }

                                    kv->value.arr.data = malloc(kv->value.arr.n * gguf_type_size(kv->value.arr.type));

                                    ok = ok && gguf_fread_el(file, kv->value.arr.data, kv->value.arr.n * gguf_type_size(kv->value.arr.type), &offset);
                                } break;
                            case GGUF_TYPE_STRING:
                                {
                                    // prevent from integer overflow in the malloc below
                                    if (kv->value.arr.n >= SIZE_MAX/sizeof(struct gguf_str)) {
                                        fprintf(stderr, "%s: array size is too large (%llu)\n", __func__, kv->value.arr.n);
                                        fclose(file);
                                        return 6;
                                    }

                                    kv->value.arr.data = malloc(kv->value.arr.n * sizeof(struct gguf_str));

                                    for (uint64_t j = 0; j < kv->value.arr.n; ++j) {
                                        ok = ok && gguf_fread_str(file, &((struct gguf_str *) kv->value.arr.data)[j], &offset);
                                    }
                                } break;
                            case GGUF_TYPE_ARRAY:
                            default: fprintf(stderr, "invalid type\n"); return 8; break;
                        }
                    } break;
                default: fprintf(stderr, "invalid type\n"); return 9; break;
            }

            if (!ok) {
                break;
            }
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read key-value pairs\n", __func__);
            fclose(file);
            return 7;
        }
    }

    // read the array infos
    {
        bool ok = true;

        ctx->infos = malloc(ctx->header.n_tensors * sizeof(struct gguf_tensor_info));

        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct gguf_tensor_info * info = &ctx->infos[i];

            for (int j = 0; j < GGUF_MAX_DIMS; ++j) {
                info->ne[j] = 1;
            }

            ok = ok && gguf_fread_str(file, &info->name,                          &offset);
            ok = ok && gguf_fread_el (file, &info->n_dims, sizeof(info->n_dims),  &offset);

            ok = ok && (info->n_dims <= GGUF_MAX_DIMS);

            for (uint32_t j = 0; j < info->n_dims; ++j) {
                ok = ok && gguf_fread_el(file, &info->ne[j], sizeof(info->ne[j]), &offset);
            }

            ok = ok && gguf_fread_el (file, &info->type,   sizeof(info->type),    &offset);
            ok = ok && gguf_fread_el (file, &info->offset, sizeof(info->offset),  &offset);

            //gguf_tensor_info_sanitize(info);

            if (!ok) {
                fprintf(stderr, "%s: failed to read array info\n", __func__);
                fclose(file);
                return 10;
            }
        }
    }

    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;

    // Not needed for now
    /*
    int alignment_idx = gguf_find_key(ctx, "general.alignment");
    if (alignment_idx != -1) {
        ctx->alignment = gguf_get_val_u32(ctx, alignment_idx);
    }
    */

    // we require the data section to be aligned, so take into account any padding
    {
        const size_t offset_pad = offset % ctx->alignment;

        if (offset_pad != 0) {
            offset += ctx->alignment - offset_pad;
            fseek(file, offset, SEEK_SET);
        }
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = offset;

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct gguf_tensor_info * info = &ctx->infos[i];

            const int64_t ne =
                (int64_t) info->ne[0] *
                (int64_t) info->ne[1] *
                (int64_t) info->ne[2] *
                (int64_t) info->ne[3];

            if (ne % ggml_blck_size(info->type) != 0) {
                fprintf(stderr, "%s: array '%s' of type %d (%s) number of elements (%llu) is not a multiple of block size (%d)\n",
                        __func__, info->name.data, (int)info->type, ggml_type_name(info->type), ne, ggml_blck_size(info->type));
                fclose(file);
                return 11;
            }

            const size_t size_cur = ggml_row_size(info->type, ne);

            ctx->size += GGUF_PAD(size_cur, ctx->alignment);
        }
    }

    // Read Data
    {
        bool ok = true;
        ctx->data = malloc(ctx->size);
        ok = ok && gguf_fread_el(file, ctx->data, ctx->size, &offset);
        if (!ok) {
            fprintf(stderr, "%s: failed to read array data\n", __func__);
            fclose(file);
            return 12;
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

