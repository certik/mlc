//
// Created by Brian Beckman on 3/10/24.
//


#include <stdlib.h>
#include <stdio.h>
#include <string.h>  // for memset
#include "kernels.h"
#include "assert.h"
#include "uuid/uuid.h"


void generate_name(pMLCA result);

void test_linkage() {
    printf("Hello from kernels.c !\n");
}


/*   ____ ___  _   _ _____ ___ ____ _   _  ___  _   _ ____         */
/*  / ___/ _ \| \ | |_   _|_ _/ ___| | | |/ _ \| | | / ___|        */
/* | |  | | | |  \| | | |  | | |  _| | | | | | | | | \___ \        */
/* | |__| |_| | |\  | | |  | | |_| | |_| | |_| | |_| |___) |       */
/*  \____\___/|_| \_| |_| |___\____|\___/ \___/ \___/|____/        */
/*  _   _  ___  _   _      ____ _____ ____  ___ ____  _____ ____   */
/* | \ | |/ _ \| \ | |    / ___|_   _|  _ \|_ _|  _ \| ____|  _ \  */
/* |  \| | | | |  \| |____\___ \ | | | |_) || || | | |  _| | | | | */
/* | |\  | |_| | |\  |_____|__) || | |  _ < | || |_| | |___| |_| | */
/* |_| \_|\___/|_| \_|    |____/ |_| |_| \_\___|____/|_____|____/  */


/*
 * Motivated by https://github.com/ggerganov/ggml/blob/43a6d4af1971ee2912ff7bc2404011ff327b6a60/include/ggml/ggml.h#L556
 */


/*
 * Count number of leading non-zero dimensions in ne.
 */
int rank(pMLCA pmlca) {
    assert(pmlca != NULL);
    int result = 0;
    for (int r = 0; r < MLC_MAX_RANK; r++) {
        if (pmlca->ne[r] == 0) {
            return result;
        }
        result += 1;
    }
    return result;
}


int64_t cols(pMLCA pmlca) {
    assert(pmlca != NULL);
    return pmlca->ne[MLC_COLS_IDX];
}


int64_t rows(pMLCA pmlca) {
    assert(pmlca != NULL);
    return pmlca->ne[MLC_ROWS_IDX];
}


int64_t sheets(pMLCA pmlca) {
    assert(pmlca != NULL);
    return pmlca->ne[MLC_SHEETS_IDX];
}


int64_t blocks(pMLCA pmlca) {
    assert(pmlca != NULL);
    return pmlca->ne[MLC_BLOCKS_IDX];
}


void validateMLCA(pMLCA pmlca) {
    assert(pmlca != NULL);
    assert(rank(pmlca) >= 1);
    assert(pmlca->data != NULL);
    // It must have columns.
    assert(cols(pmlca) <= MLC_MAX_COLS);
    assert(cols(pmlca) >= 1);
    if (pmlca->ne[MLC_ROWS_IDX] != 0) {
        assert(rank(pmlca) >= 2);
        assert(rows(pmlca) >= 1);
        assert(rows(pmlca) <= MLC_MAX_ROWS);
    } else {
        // If it doesn't have rows, it must not have sheets and blocks.
        assert(pmlca->ne[MLC_SHEETS_IDX] == 0);
        assert(pmlca->ne[MLC_BLOCKS_IDX] == 0);
    }
    if (pmlca->ne[MLC_SHEETS_IDX] != 0) {
        assert(rank(pmlca) >= 3);
        assert(sheets(pmlca) >= 1);
        assert(sheets(pmlca) <= MLC_MAX_SHEETS);
    } else {
        // If it doesn't have sheets, it must not have blocks.
        assert(pmlca->ne[MLC_BLOCKS_IDX] == 0);
    }
    if (pmlca->ne[3] != 0) {
        assert(rank(pmlca) == 4);
        assert(blocks(pmlca) >= 1);
        assert(blocks(pmlca) <= MLC_MAX_BLOCKS);
    }
    int64_t total_elts = 1;
    for (int r = 0; r < rank(pmlca); r++) {
        total_elts *= pmlca->ne[r];
    }
    int64_t total_size = ((int64_t) (total_elts * MLC_DATUM_SIZE));
    assert(total_size == pmlca->total_size);
}


f32 get1D(pMLCA pmlca, int64_t col) {
    validateMLCA(pmlca);
    assert(rank(pmlca) == 1);
    f32 result = ((f32 *) (pmlca->data))[col];
    return result;
}


f32 get2D(pMLCA pmlca, int64_t row, int64_t col) {
    validateMLCA(pmlca);
    assert(rank(pmlca) == 2);
    int64_t index = (row * cols(pmlca)) + col;
    f32 result = ((f32 *) (pmlca->data))[index];
    return result;
}


f32 get3D(pMLCA pmlca, int64_t sheet,
          int64_t row, int64_t col) {
    validateMLCA(pmlca);
    assert(rank(pmlca) == 3);
    int64_t index = (sheet * rows(pmlca) * cols(pmlca))
                    + (row * cols(pmlca)) + col;
    f32 result = ((f32 *) (pmlca->data))[index];
    return result;
}


f32 get4D(pMLCA pmlca,
        int64_t block, int64_t sheet,
        int64_t row, int64_t col) {
    validateMLCA(pmlca);
    assert(rank(pmlca) == 4);
    int64_t index =
            (block * sheets(pmlca) * rows(pmlca) * cols(pmlca))
            + (sheet * rows(pmlca) * cols(pmlca))
            + (row * cols(pmlca)) + col;
    f32 result = ((f32 *) (pmlca->data))[index];
    return result;
}


void put1D(pMLCA pmlca, int64_t col, f32 val) {
    validateMLCA(pmlca);
    assert(rank(pmlca) == 1);
    f32 *pointer = ((f32 *) (pmlca->data));
    pointer[col] = val;
}


void put2D(pMLCA pmlca, int64_t row, int64_t col, f32 val) {
    validateMLCA(pmlca);
    assert(rank(pmlca) == 2);
    f32 *pointer = ((f32 *) (pmlca->data));
    int64_t index = (row * cols(pmlca)) + col;
    pointer[index] = val;
}


void put3D(pMLCA pmlca, int64_t sheet,
           int64_t row, int64_t col, f32 val) {
    validateMLCA(pmlca);
    assert(rank(pmlca) == 3);
    f32 *pointer = ((f32 *) (pmlca->data));
    int64_t index = (sheet * rows(pmlca) * cols(pmlca))
                    + (row * cols(pmlca)) + col;
    pointer[index] = val;
}


void put4D(pMLCA pmlca,
           int64_t block, int64_t sheet,
           int64_t row, int64_t col, f32 val) {
    validateMLCA(pmlca);
    assert(rank(pmlca) == 4);
    f32 *pointer = ((f32 *) (pmlca->data));
    int64_t index =
            (block * sheets(pmlca) * rows(pmlca) * cols(pmlca))
            + (sheet * rows(pmlca) * cols(pmlca))
            + (row * cols(pmlca)) + col;
    pointer[index] = val;
}


pMLCA alloc1D(int64_t cols) {
    assert(cols <= MLC_MAX_COLS);
    pMLCA result = malloc(MLC_SIZE);
    assert(result != NULL);
    size_t total_size = cols * MLC_DATUM_SIZE;
    result->total_size = total_size;
    result->data = malloc(total_size);
    assert(result->data != NULL);
    memset(result->ne, 0, MLC_MAX_RANK * sizeof(result->ne[0]));
    result->ne[MLC_COLS_IDX] = cols;
    generate_name(result);
    validateMLCA(result);
    return result;
}


pMLCA alloc2D(int64_t rows, int64_t cols) {
    assert(rows <= MLC_MAX_ROWS);
    assert(cols <= MLC_MAX_COLS);
    pMLCA result = malloc(MLC_SIZE);
    assert(result != NULL);
    size_t total_size = rows * cols * MLC_DATUM_SIZE;
    result->total_size = total_size;
    result->data = malloc(total_size);
    assert(result->data != NULL);
    memset(result->ne, 0, MLC_MAX_RANK * sizeof(result->ne[0]));
    result->ne[MLC_COLS_IDX] = cols;
    result->ne[MLC_ROWS_IDX] = rows;
    generate_name(result);
    validateMLCA(result);
    return result;
}


pMLCA alloc3D(int64_t sheets,
              int64_t rows, int64_t cols) {
    assert(rows <= MLC_MAX_ROWS);
    assert(cols <= MLC_MAX_COLS);
    assert(sheets <= MLC_MAX_SHEETS);
    pMLCA result = malloc(MLC_SIZE);
    assert(result != NULL);
    size_t total_size = sheets * rows * cols * MLC_DATUM_SIZE;
    result->total_size = total_size;
    result->data = malloc(total_size);
    assert(result->data != NULL);
    memset(result->ne, 0, MLC_MAX_RANK * sizeof(result->ne[0]));
    result->ne[MLC_COLS_IDX] = cols;
    result->ne[MLC_ROWS_IDX] = rows;
    result->ne[MLC_SHEETS_IDX] = sheets;
    generate_name(result);
    validateMLCA(result);
    return result;
}


pMLCA alloc4D(int64_t blocks, int64_t sheets,
              int64_t rows, int64_t cols) {
    assert(rows <= MLC_MAX_ROWS);
    assert(cols <= MLC_MAX_COLS);
    assert(sheets <= MLC_MAX_SHEETS);
    assert(blocks <= MLC_MAX_BLOCKS);
    pMLCA result = malloc(MLC_SIZE);
    assert(result != NULL);
    size_t total_size = blocks * sheets * rows * cols * MLC_DATUM_SIZE;
    result->total_size = total_size;
    result->data = malloc(total_size);
    assert(result->data != NULL);
    memset(result->ne, 0, MLC_MAX_RANK * sizeof(result->ne[0]));
    result->ne[MLC_COLS_IDX] = cols;
    result->ne[MLC_ROWS_IDX] = rows;
    result->ne[MLC_SHEETS_IDX] = sheets;
    result->ne[MLC_BLOCKS_IDX] = blocks;
    generate_name(result);
    validateMLCA(result);
    return result;
}


pMLCA freeMLCA(pMLCA pmlca) {
    validateMLCA(pmlca);
    memset(pmlca->data, 0, pmlca->total_size);
    free(pmlca->data);
    memset(pmlca->name, 0, MLC_MAX_NAME);
    memset(pmlca->ne, 0, MLC_MAX_RANK * (sizeof(int64_t)));
    pmlca->total_size = 0;
    free(pmlca);
    return NULL;
}


void generate_name(pMLCA result) {
    unsigned char buffer[17] = {0};
    uuid_generate(buffer);
    // result->name is 64 chars; unparse write 37
    uuid_unparse(buffer, result->name);
}


void dumpMLCA(pMLCA pmlca) {
    validateMLCA(pmlca);
    switch (rank(pmlca)) {
        case 1:
            printf("\n1D CONTIG DUMP\n");
            printf("Array name = %s\n", pmlca->name);
            printf("cols = %lld, total_size = %ld\n", cols(pmlca), pmlca->total_size);
            for (int col = 0; col < cols(pmlca); col++) {
                printf("%.0f ", get1D(pmlca, col));
            }
            printf("\n");
            break;
        case 2:
            printf("\n2D CONTIG DUMP\n");
            printf("Array name = %s\n", pmlca->name);
            printf("cols = %lld, rows = %lld, total_size = %ld\n",
                   cols(pmlca), rows(pmlca), pmlca->total_size);
            for (int row = 0; row < rows(pmlca); row++) {
                printf("row[%d]\t", row);
                for (int col = 0; col < cols(pmlca); col++) {
                    printf("%.0f ", get2D(pmlca, row, col));
                }
                printf("\n");
            }
            break;
        case 3:
            printf("\n3D CONTIG DUMP\n");
            printf("Array name = %s\n", pmlca->name);
            printf("cols = %lld, rows = %lld, sheets = %lld, total_size = %ld\n",
                   cols(pmlca), rows(pmlca), sheets(pmlca), pmlca->total_size);
            for (int sheet = 0; sheet < sheets(pmlca); sheet++) {
                printf("\n");
                for (int row = 0; row < rows(pmlca); row++) {
                    printf("sheet[%d], row[%d]\t", sheet, row);
                    for (int col = 0; col < cols(pmlca); col++) {
                        printf("%.0f ", get3D(
                                pmlca, sheet,
                                row, col));
                    }
                    printf("\n");
                }
            }
            break;
        case 4:
            printf("\n4D CONTIG DUMP\n");
            printf("Array name = %s\n", pmlca->name);
            printf("cols = %lld, rows = %lld, sheets = %lld, blocks = %lld, total_size = %ld\n",
                   cols(pmlca), rows(pmlca), sheets(pmlca), blocks(pmlca),
                   pmlca->total_size);
            for (int block = 0; block < blocks(pmlca); block++) {
                printf("\n");
                for (int sheet = 0; sheet < sheets(pmlca); sheet++) {
                    printf("\n");
                    for (int row = 0; row < rows(pmlca); row++) {
                        printf("block[%d], sheet[%d], row[%d]\t", block, sheet, row);
                        for (int col = 0; col < cols(pmlca); col++) {
                            printf("%.0f ", get4D(
                                    pmlca, block, sheet,
                                    row, col));
                        }
                        printf("\n");
                    }
                }
            }
            break;
        default:
            assert(0);
    }
}


