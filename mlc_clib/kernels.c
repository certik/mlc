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


/*                _       _  */
/*  _ _ __ _ _ _ | |_____/ | */
/* | '_/ _` | ' \| / /___| | */
/* |_| \__,_|_||_|_\_\   |_| */


pA1 A1_alloc(int cols) {
    assert(cols >= 1);
    pA1 result = malloc(sizeof(A1));
    assert(result != NULL);
    result->cols = cols;
    result->p_storage = malloc(cols * sizeof(f32));
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
    memset(it->p_storage, 0, it->cols * sizeof(f32));
    free(it->p_storage);
    it->p_storage = NULL;
    it->cols = 0;
    free(it);
    return NULL;
}


void A1_dump(pA1 it) {
    A1_validate(it);
    printf("\n1D DUMP\n");
    printf("cols = %d\n", it->cols);
    for (int i = 0; i < it->cols; i++) {
        printf("%.0f ", it->p_storage[i]);
    }
    printf("\n");
}


void A1_validate(pA1 it) {
    assert(it != NULL);
    assert(it->cols >= 1);
    assert(it->p_storage != NULL);
}


/*                _      ___  */
/*  _ _ __ _ _ _ | |____|_  ) */
/* | '_/ _` | ' \| / /___/ /  */
/* |_| \__,_|_||_|_\_\  /___| */


pA2 A2_alloc(int rows, int cols) {
    assert(rows >= 1);
    assert(cols >= 1);
    pA2 result = malloc(sizeof(A2));
    assert(result != NULL);
    result->pp_storage = malloc(rows * sizeof(f32 *));
    assert(result->pp_storage != NULL);
    for (int i = 0; i < rows; i++) {
        result->pp_storage[i] = malloc(cols * sizeof(f32));
        assert(result->pp_storage[i] != NULL);
    }
    result->rows = rows;
    result->cols = cols;
    return result;
}


pA2 A2_free(pA2 it) {
    A2_validate(it);
    for (int i = 0; i < it->rows; i++) {
        memset(it->pp_storage[i], 0, it->cols * sizeof(f32));
        free(it->pp_storage[i]);
    }
    memset(it->pp_storage, 0, it->rows * sizeof(f32 *));
    free(it->pp_storage);
    it->rows = 0;
    it->cols = 0;
    free(it);
    return NULL;
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
    A2_validate(it);
    printf("\n2D DUMP\n");
    printf("rows = %d, cols = %d\n", it->rows, it->cols);
    for (int i = 0; i < it->rows; i++) {
        printf("row[%d]\t", i);
        for (int j = 0; j < it->cols; j++) {
            printf("%.0f ", it->pp_storage[i][j]);
        }
        printf("\n");
    }
}


/*                _      ____ */
/*  _ _ __ _ _ _ | |____|__ / */
/* | '_/ _` | ' \| / /___|_ \ */
/* |_| \__,_|_||_|_\_\  |___/ */


pA3 A3_alloc(int rows, int cols, int sheets) {
    assert(rows >= 1);
    assert(cols >= 1);
    assert(sheets >= 1);
    pA3 result = malloc(sizeof(A3));
    assert(result != NULL);
    result->ppp_storage = malloc(sheets * sizeof(f32 **));
    assert(result->ppp_storage != NULL);
    for (int k = 0; k < sheets; k++) {
        result->ppp_storage[k] = malloc(rows * sizeof(f32 *));
        assert(result->ppp_storage[k] != NULL);
        for (int i = 0; i < rows; i++) {
            result->ppp_storage[k][i] = malloc(cols * sizeof(f32));
            assert(result->ppp_storage[k][i] != NULL);
        }
    }
    result->rows = rows;
    result->cols = cols;
    result->sheets = sheets;
    return result;
}


pA3 A3_free(pA3 it) {
    A3_validate(it);
    for (int k = 0; k < it->sheets; k++) {
        for (int i = 0; i < it->rows; i++) {
            memset(it->ppp_storage[k][i], 0, it->cols * sizeof(f32));
            free(it->ppp_storage[k][i]);
        }
        memset(it->ppp_storage[k], 0, it->rows * sizeof(f32 *));
        free(it->ppp_storage[k]);
    }
    free(it->ppp_storage);
    it->sheets = 0;
    it->rows = 0;
    it->cols = 0;
    free(it);
    return NULL;
}


void A3_validate(pA3 it) {
    assert(it != NULL);
    assert(it->rows >= 1);
    assert(it->cols >= 1);
    assert(it->sheets >= 1);
    assert(it->ppp_storage != NULL);
    for (int k = 0; k < it->sheets; k++) {
        assert(it->ppp_storage[k] != NULL);
        for (int i = 0; i < it->rows; i++) {
            assert(it->ppp_storage[k][i] != NULL);
        }
    }
}


void A3_dump(pA3 it) {
    A3_validate(it);
    printf("\n3D DUMP\n");
    printf("rows = %d, cols = %d, sheets = %d\n",
           it->rows, it->cols, it->sheets);
    for (int k = 0; k < it->sheets; k++) {
        printf("\n");
        for (int i = 0; i < it->rows; i++) {
            printf("sheet[%d], row[%d]\t", k, i);
            for (int j = 0; j < it->cols; j++) {
                printf("%.0f ", it->ppp_storage[k][i][j]);
            }
            printf("\n");
        }
    }
}


/*                _       _ _   */
/*  _ _ __ _ _ _ | |_____| | |  */
/* | '_/ _` | ' \| / /___|_  _| */
/* |_| \__,_|_||_|_\_\     |_|  */


pA4 A4_alloc(int rows, int cols, int sheets, int blocks) {
    assert(rows >= 1);
    assert(cols >= 1);
    assert(sheets >= 1);
    assert(blocks >= 1);
    pA4 result = malloc(sizeof(A4));
    assert(result != NULL);
    result->pppp_storage = malloc(sheets * sizeof(f32 ***));
    assert(result->pppp_storage != NULL);
    for (int l = 0; l < blocks; l++) {
        result->pppp_storage[l] = malloc(sheets * sizeof(f32 **));
        for (int k = 0; k < sheets; k++) {
            result->pppp_storage[l][k] = malloc(rows * sizeof(f32 *));
            assert(result->pppp_storage[l][k] != NULL);
            for (int i = 0; i < rows; i++) {
                result->pppp_storage[l][k][i] = malloc(cols * sizeof(f32));
                assert(result->pppp_storage[l][k][i] != NULL);
            }
        }
    }
    result->rows = rows;
    result->cols = cols;
    result->sheets = sheets;
    result->blocks = blocks;
    return result;
}


pA4 A4_free(pA4 it) {
    A4_validate(it);
    for (int l = 0; l < it->blocks; l++) {
        for (int k = 0; k < it->sheets; k++) {
            for (int i = 0; i < it->rows; i++) {
                memset(it->pppp_storage[l][k][i], 0, it->cols * sizeof(f32));
                free(it->pppp_storage[l][k][i]);
            }
            memset(it->pppp_storage[l][k], 0, it->rows * sizeof(f32 *));
            free(it->pppp_storage[l][k]);
        }
        memset(it->pppp_storage[l], 0, it->sheets * sizeof(f32 **));
        free(it->pppp_storage[l]);
    }
    free(it->pppp_storage);
    it->blocks = 0;
    it->sheets = 0;
    it->rows = 0;
    it->cols = 0;
    free(it);
    return NULL;
}


void A4_validate(pA4 it) {
    assert(it != NULL);
    assert(it->rows >= 1);
    assert(it->cols >= 1);
    assert(it->sheets >= 1);
    assert(it->blocks >= 1);
    assert(it->pppp_storage != NULL);
    for (int l = 0; l < it->blocks; l++) {
        assert(it->pppp_storage[l] != NULL);
        for (int k = 0; k < it->sheets; k++) {
            assert(it->pppp_storage[l][k] != NULL);
            for (int i = 0; i < it->rows; i++) {
                assert(it->pppp_storage[l][k][i] != NULL);
            }
        }
    }
}


void A4_dump(pA4 it) {
    A4_validate(it);
    printf("\n4D DUMP\n");
    printf("blocks = %d, sheets = %d, rows = %d, cols = %d\n",
           it->blocks, it->sheets, it->rows, it->cols);
    for (int l = 0; l < it->blocks; l++) {
        printf("\n");
        for (int k = 0; k < it->sheets; k++) {
            printf("\n");
            for (int i = 0; i < it->rows; i++) {
                printf("block[%d], sheet[%d], row[%d]\t",
                       l, k, i);
                for (int j = 0; j < it->cols; j++) {
                    printf("%.0f ", it->pppp_storage[l][k][i][j]);
                }
                printf("\n");
            }
        }
    }
}


/*
 * A typical conv_kernel1 is 5 x 5. The Input must be reduced and
 * on the left and on the right. The actual innermost convolution
 * for MNIST is a 5x5 boxcar kernel on a 28x28 grid, and,
 * actually, 1x32 of such things. The beginning index of the
 * raster process is 2, and the ending index is 25 = (cols - 1) -
 * 2. For a square boxcar of odd dimensions n x n, let b =
 * floor(n/2) =def= n//2. This is 2 when n == 5. The raster
 * summing begins at index b and ends at index (col - 1) - b
 * (inclusive). The results are, in general of dims (rows - 2b)
 * and (cols - 2b)
 *
 *                             1                   2
 *         0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7
 *      0 |. . . . .|. . . . . . . . . . . . . . . . . .|. . . . .|
 *      1 |. . . . .|. . . . . . . . . . . . . . . . . .|. . . . .|
 *      2 |. . o . .|. .~~~~>. . . .~~~~>. . . .~~~~>. .|. . o . .|
 *      3 |. . . . .|. . . . . . . . . . . . . . . . . .|. . . . .|
 *      4 |. . . . .|. . . . . . . . . . . . . . . . . .|. . . . .|
 *      5  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 *      6  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 *          .       .       .       .       .       .       .
 *            .       .       .       .       .       .       .
 *              .       .       .       .       .       .       .
 *     20  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 *      1  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 *      2  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 *      3  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 *      4  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 *      5  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 *      6  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 *      7  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 */


pA3 Conv2D(pA4 boxcars, pA2 Input) {
    pA3 result = NULL;
    return result;
}
