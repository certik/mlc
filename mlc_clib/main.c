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
    return 0;
}
