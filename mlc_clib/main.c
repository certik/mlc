#include <stdio.h>

#include "kernels.h"

void test_4D();
void test_3D();
void test_2D();
void test_1D();

/*
 * On Mac, run "leaks" on the executable.
 *
 * leaks --atExit -- ./mlc_clib
 */

int main() {
    test_linkage();
    test_1D();
    test_2D();
    test_3D();
    test_4D();
    return 0;
}

void test_1D() {
    int cols = 42;
    pA1 pa1 = A1_alloc(cols);
    for (int i = 0; i < cols; i++) {
        pa1->p_storage[i] = ((f32)(i));
    }
    A1_dump(pa1);
    A1_free(pa1);
    // DON'T DO THIS!
    // A1_dump(pa1);
}

void test_2D() {
    int cols = 7;
    int rows = 6;
    pA2 pa2 = A2_alloc(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            pa2->pp_storage[i][j] = ((f32)((i + 1) * (j + 1)));
        }
    }
    A2_dump(pa2);
    A2_free(pa2);
}

void test_3D() {
    int sheets = 4;
    int cols = 7;
    int rows = 6;
    pA3 pa3 = A3_alloc(rows, cols, sheets);
    for (int k = 0; k < sheets; k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                pa3->ppp_storage[k][i][j] = ((f32) ((i + 1) * (j + 1) * (k + 1)));
            }
        }
    }
    A3_dump(pa3);
    A3_free(pa3);
}

void test_4D() {
    int blocks = 3;
    int sheets = 4;
    int cols = 7;
    int rows = 6;
    pA4 pa4 = A4_alloc(rows, cols, sheets, blocks);
    for (int l = 0; l < blocks; l ++) {
        for (int k = 0; k < sheets; k++) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    pa4->pppp_storage[l][k][i][j] = 
                            ((f32) ((i + 1) * (j + 1) * (k + 1) * (l + 1)));
                }
            }
        }
    }
    A4_dump(pa4);
    A4_free(pa4);
}
