#include <stdio.h>

#include "kernels.h"

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
    return 0;
}

void test_1D() {
    pA1 pa1 = A1_alloc(42);
    for (int i = 0; i < 42; i++) {
        pa1->p_storage[i] = ((f32)(i));
    }
    A1_dump(pa1);
    A1_free(pa1);
    // DON'T DO THIS!
    // A1_dump(pa1);
}

void test_2D() {
    pA2 pa2 = A2_alloc(6, 7);
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 7; j++) {
            pa2->pp_storage[i][j] = ((f32)((i + 1) * (j + 1)));
        }
    }
    A2_dump(pa2);
    A2_free(pa2);
}

void test_3D() {
    pA3 pa3 = A3_alloc(6, 7, 4);
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 7; j++) {
                pa3->ppp_storage[k][i][j] = ((f32) ((i + 1) * (j + 1) * (k + 1)));
            }
        }
    }
    A3_dump(pa3);
    A3_free(pa3);
}
