//
// Created by Brian Beckman on 3/10/24.
//

#ifndef MLC_CLIB_KERNELS_H
#define MLC_CLIB_KERNELS_H


void test_linkage();

/*
 * For the specs of all operations, see the file
 * "test_beautiful.mnist.specs.txt".
 */



/*
 * All the type=<Type...> specs are for f32s, so that's the
 * only type we need here.
 */


typedef float f32;


/*
 * There are arrays of rank 1, 2, 3, and 4 in the spec.
 *
 * Internally, to start, let's have a non-contiguous storage
 * model, where a 2-D array is an array of pointers to 1-D arrays,
 * and 1-D arrays are contiguous. This model can be re- placed
 * with a contiguous model, in which a 2-D array is a contiguous
 * block and its array of pointers pp_storage are strided into it.
 */


/*                _       _  */
/*  _ _ __ _ _ _ | |_____/ | */
/* | '_/ _` | ' \| / /___| | */
/* |_| \__,_|_||_|_\_\   |_| */


typedef struct array_1 {
    f32 * p_storage;
    int cols;
} A1, * pA1;


pA1 A1_alloc(int cols);
/*
 * Return a null pointer from "free" to help calling code to avoid
 * using deallocated memory.
 */
pA1 A1_free(pA1 it);
void A1_validate(pA1 it);
void A1_dump(pA1 it);


/*                _      ___  */
/*  _ _ __ _ _ _ | |____|_  ) */
/* | '_/ _` | ' \| / /___/ /  */
/* |_| \__,_|_||_|_\_\  /___| */


typedef struct array_2 {
    f32 ** pp_storage;
    int rows;
    int cols;
} A2, * pA2;


pA2 A2_alloc(int rows, int cols);
pA2 A2_free(pA2 it);
void A2_validate(pA2 it);
void A2_dump(pA2 it);


/*                _      ____ */
/*  _ _ __ _ _ _ | |____|__ / */
/* | '_/ _` | ' \| / /___|_ \ */
/* |_| \__,_|_||_|_\_\  |___/ */


typedef struct array_3 {
    f32 *** ppp_storage;
    int rows;
    int cols;
    int sheets;
} A3, * pA3;


pA3 A3_alloc(int rows, int cols, int sheets);
pA3 A3_free(pA3 it);
void A3_validate(pA3 it);
void A3_dump(pA3 it);


/*                _       _ _   */
/*  _ _ __ _ _ _ | |_____| | |  */
/* | '_/ _` | ' \| / /___|_  _| */
/* |_| \__,_|_||_|_\_\     |_|  */


typedef struct array_4 {
    f32 **** pppp_storage;
    int rows;
    int cols;
    int sheets;
    int blocks;
} A4, * pA4;


pA4 A4_alloc(int rows, int cols, int sheets, int blocks);
pA4 A4_free(pA4 it);
void A4_validate(pA4 it);
void A4_dump(pA4 it);


/*
 * The innermost operation is a Conv2D with the following spec:
 *
 * (Operation(
 *     op_type=<OpType.Conv2D: 6>,
 *     args=(
 *         Array(
 *             name='conv_kernel1',
 *             type=<Type.f32: 3>,
 *             rank=2,
 *             shape=(5, 5, 1, 32),
 *             memory_space=<MemorySpace.host: 1>),
 *         Array(
 *             name='Input',
 *             type=<Type.f32: 3>,
 *             rank=2,
 *             shape=(28, 28),
 *             memory_space=<MemorySpace.host: 1>)),
 *     execution_space=<ExecutionSpace.host: 1>,
 *     type=<Type.f32: 3>,
 *     rank=3,
 *     shape=[24, 24, 32],
 *     memory_space=<MemorySpace.host: 1>),)
 */


pA3 Conv2D(pA4 conv_kernel1, pA2 Input);


#endif //MLC_CLIB_KERNELS_H
