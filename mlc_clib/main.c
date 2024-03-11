#include <stdio.h>

#include "kernels.h"

/*
 * On Mac, run "leaks" on the executable.
 *
 * leaks --atExit -- ./mlc_clib
 */

int main() {
    test_linkage();
    pA1 pa1 = A1_alloc(42);
    for (int i = 0; i < 42; i++) {
        pa1->p_storage[i] = ((f32)(i));
    }
    A1_dump(pa1);
    A1_free(pa1);
    // DON'T DO THIS!
    // A1_dump(pa1);

    pA2 pa2 = A2_alloc(6, 7);
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 7; j++) {
            pa2->pp_storage[i][j] = ((f32)((i + 1) * (j + 1)));
        }
    }
    A2_dump(pa2);
    A2_free(pa2);
    return 0;
}
