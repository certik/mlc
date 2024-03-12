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
 * There are arrays of rank 1, 2, 3, and 4 in the spec. No fancy
 * error management for now, just assert that everything is OK.
 */


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


#define MLC_MAX_RANK    4

#define MLC_COLS_IDX    0
#define MLC_ROWS_IDX    1
#define MLC_SHEETS_IDX  2
#define MLC_BLOCKS_IDX  3

#define MLC_DATUM_SIZE (sizeof(f32))

// Guesses -- full-boat is 256 GB
#define MLC_MAX_COLS    1024
#define MLC_MAX_ROWS    1024
#define MLC_MAX_SHEETS   128
#define MLC_MAX_BLOCKS    64

#define MLC_MAX_NAME   64  // actual name < 64; last byte for string-term NULL

// like ggml_tensor at the URL above, just less general.
typedef struct mlc_array {
    /*
     * [42, 0, 0, 0] is a rank-1 array, with 42 columns
     * [ 7, 6, 0, 0] is a rank-2 array, with 7 columns, 6 rows
     * [ 7, 6, 4, 0] is a rank-3 array, 7 cols, 6 rows, 4 sheets
     * [ 7, 6, 4, 3] is a rank-4 array, 7 cols, 6 rows, 4 sheets, 3 blocks
     */
    int64_t ne[MLC_MAX_RANK];
//  size_t  nb[MLC_MAX_RANK];  // strides (unused)
    void *  data;
    char    name[MLC_MAX_NAME];
    size_t  total_size;  // for clearning on dealloc
} MLCA, * pMLCA;

#define MLC_SIZE (sizeof(MLCA))


/*
 * Count number of leading non-zero dimensions in ne.
 */
int rank(pMLCA pmlca);
int64_t cols(pMLCA pmlca);
int64_t rows(pMLCA pmlca);
int64_t sheets(pMLCA pmlca);
int64_t blocks(pMLCA pmlca);


f32 get1D(pMLCA pmlca, int64_t col);
f32 get2D(pMLCA pmlca, int64_t row, int64_t col);
f32 get3D(pMLCA pmlca, int64_t sheet, int64_t row, int64_t col);
f32 get4D(pMLCA pmlca, int64_t block, int64_t sheet, int64_t row, int64_t col);


void put1D(pMLCA pmlca, int64_t col, f32 val);
void put2D(pMLCA pmlca, int64_t row, int64_t col, f32 val);
void put3D(pMLCA pmlca, int64_t sheet, int64_t row, int64_t col, f32 val);
void put4D(pMLCA pmlca, int64_t block, int64_t sheet, int64_t row, int64_t col, f32 val);


pMLCA alloc1D(int64_t cols);
pMLCA alloc2D(int64_t rows, int64_t cols);
pMLCA alloc3D(int64_t sheets, int64_t rows, int64_t cols);
pMLCA alloc4D(int64_t blocks, int64_t sheets, int64_t rows, int64_t cols);


void validateMLCA(pMLCA pmlca);
pMLCA freeMLCA(pMLCA pmlca);
void dumpMLCA(pMLCA pmlca);


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


#endif //MLC_CLIB_KERNELS_H
