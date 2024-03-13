//
// Created by Brian Beckman on 3/12/24.
//

#ifndef MLC_CLIB_GGUF_H
#define MLC_CLIB_GGUF_H

#include <stdio.h>
#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

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


int ggml_blck_size(enum ggml_type type);

size_t ggml_type_size(enum ggml_type type);

size_t ggml_row_size(enum ggml_type type, int64_t ne);

const char * ggml_type_name(enum ggml_type type);

static bool gguf_fread_el(
        FILE * file, void * dst, size_t size, size_t * offset);

static bool gguf_fread_str(FILE * file, struct gguf_str * p, size_t * offset);

int gguf_read(const char *fname, struct gguf_context *ctx);

#endif //MLC_CLIB_GGUF_H
