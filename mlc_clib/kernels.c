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
    pA1 result = malloc(dim * sizeof(A1));
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
    memset(it->p_storage, it->dim, sizeof(f32));
    it->dim = 0;
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
        printf("%f ", it->p_storage[i]);
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
}


pA2 A2_free(pA2 it) {

}


void A2_validate(pA2 it) {

}


void A2_dump(pA2 it) {

}
