#include <stdio.h>
#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdbool.h>

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

int main() {
    int fd = open("examples/mnist/mnist-cnn-model.gguf", O_RDONLY);
    FILE * f = fdopen(fd, "r");
    int fclosed = fclose(f);
    assert(fclosed == 0);
    return fclosed;
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

