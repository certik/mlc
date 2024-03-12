#include <stdio.h>

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

int main() {
    test_linkage();
    test_1D_MLCA();
    test_2D_MLCA();
    test_3D_MLCA();
    test_4D_MLCA();
    return 0;
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

