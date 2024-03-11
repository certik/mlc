//
// Created by Brian Beckman on 3/10/24.
//


#include <stdlib.h>
#include <stdio.h>
#include <string.h>  // for memset
#include "kernels.h"
#include "assert.h"


void test_linkage() {
    printf("Hello from kernels.c !");
}


/*
 * No advanced error handling. Just assert that everything is ok.
 */
pA1 A1_alloc(int dim) {
    assert(dim >= 1);
    pA1 result = malloc(sizeof(A1));
    assert(result != NULL);
    result->dim = dim;
    result->p_storage = malloc(dim * sizeof(f32));
    assert(result->p_storage != NULL);
    return result;
}


/*
 * Return a null pointer to help calling code to avoid using
 * deallocated memory.
 *
 * On Mac, run "leaks" on the executable.
 *
 * leaks --atExit -- ./mlc_clib
 */
pA1 A1_free(pA1 it) {
    A1_validate(it);
    memset(it->p_storage, 0, it->dim * sizeof(f32));
    free(it->p_storage);
    it->p_storage = NULL;
    it->dim = 0;
    free(it);
    return NULL;
}


void A1_dump(pA1 it) {
    A1_validate(it);
    printf("\n");
    for (int i = 0; i < it->dim; i++) {
        printf("%.0f ", it->p_storage[i]);
    }
    printf("\n");
}


void A1_validate(pA1 it) {
    assert(it != NULL);
    assert(it->dim >= 1);
    assert(it->p_storage != NULL);
}


pA2 A2_alloc(int rows, int cols) {
    assert(rows >= 1);
    assert(cols >= 1);
    pA2 result = malloc(sizeof(A2));
    assert(result != NULL);
    result->rows = rows;
    result->pp_storage = malloc(rows * sizeof(f32 *));
    assert(result->pp_storage != NULL);
    for (int i = 0; i < rows; i++) {
        result->pp_storage[i] = malloc(cols * sizeof(f32));
        assert(result->pp_storage[i] != NULL);
    }
    return result;
}


pA2 A2_free(pA2 it) {
    A2_validate(it);
    for (int i = 0; i < it->rows; i++) {
//        memset(it->pp_storage[i], it->cols, )
    }
}


void A2_validate(pA2 it) {
    assert(it != NULL);
    assert(it->rows >= 1);
    assert(it->cols >= 1);
    assert(it->pp_storage != NULL);
    for (int i = 0; i < it->rows; i++) {
        assert(it->pp_storage[i] != NULL);
    }
}


void A2_dump(pA2 it) {

}
