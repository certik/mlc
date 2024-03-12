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


//# specs of function calls:
//
//.Sequential(layers=[
//Conv2D(in_channels=1, out_channels=32, kernel_size=5, bias=False),
//ReLU(),
//    Conv2D(in_channels=32, out_channels=32, kernel_size=5, bias=False),
//    ReLU(),
//    BatchNorm2D(num_features=32),
//    MaxPool2D(kernel_size=(2, 2)),
//    Conv2D(in_channels=32, out_channels=64, kernel_size=3, bias=False),
//    ReLU(),
//    Conv2D(in_channels=64, out_channels=64, kernel_size=3, bias=False),
//    ReLU(),
//    BatchNorm2D(num_features=64),
//    MaxPool2D(kernel_size=(2, 2)),
//    Flatten(start_dim=1, end_dim=-1),
//    Linear(in_features=576, out_features=10,
//       bias=False)])
//
//# execution-tree trace:
//
// let inner_rel_conv1 = Operation(ReLU,
//    args=(
//      Operation(Conv2D,
//        args=(
//          Array('conv_kernel1', (5, 5, 1, 32)),
//          Array('Input', (28, 28))),
//        [24, 24, 32])),
//    [24, 24, 32])
//
// let inner_rel_conv2 = Operation(ReLU,
//    args=(
//      Operation(Conv2D,
//        args=(
//          Array('conv_kernel2', (5, 5, 32, 32)),
//          inner_rel_conv1),
//        [20, 20, 32])),
//    [20, 20, 32])),
//
// let inner_poolnorm2 = Operation(MaxPool2D,
//    args=(
//      Operation(BatchNorm2D,
//        args=(
//          inner_rel_conv2,
//          [20, 20, 32])),
//    [10, 10, 32])),
//
// let rel_conv3 = Operation(ReLU,
//      args=(
//      Operation(Conv2D,
//        args=(
//          Array('conv_kernel3', (3, 3, 32, 64)),
//          inner_poolnorm2,
//          [8, 8, 64])),
//      [8, 8, 64])),
//
// let rel_conv4 = Operation(ReLU,
//      args=(
//      Operation(Conv2D,
//        args=(
//          Array('conv_kernel4', (3, 3, 64, 64)),
//          rel_conv3,
//          [6, 6, 64])),
//      [6, 6, 64])),
//
//
// let result = Operation(MatVec,
//    args=(
//    Operation(Reshape,
//      args=(
//      Operation(MaxPool2D,
//        args=(
//        Operation(BatchNorm2D,
//          args=(rel_conv4),
//          [6, 6, 64])),
//        [3, 3, 64])),
//    (576)),
//  Array('linear_w1', (576, 10))),
//  (10))





#endif //MLC_CLIB_KERNELS_H
