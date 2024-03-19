//
// Created by Brian Beckman on 3/10/24.
//

#ifndef MLC_CLIB_KERNELS_H
#define MLC_CLIB_KERNELS_H

/*
 * All the type=<Type...> specs are for f32s, so that's the
 * only type we need here.
 */


typedef float f32;
typedef _Float16 f16;

void transpose(int n1, int n2, int n3, int n4, const f32 *A,
            int t1, int t2, int t3, int t4, f32 *B);
void conv2d_kernel(int kernel_size, int in_h, int in_w,
        const f32 *weight, // (3,3)
        const f32 *x, // (in_h,in_w)
        f32 *out // (out_w,out_h)
        );
void conv2d(int in_channels, int out_channels, int kernel_size,
        int in_h, int in_w,
        f32 *weight, // (out_channels,in_channels,3,3)
        const f32 *bias, // (out_channels,)
        f32 *x, // (in_channels,in_h,in_w)
        f32 *out // (out_channels,out_h,out_w)
        );
void conv2d_no_bias(int in_channels, int out_channels, int kernel_size,
        int in_h, int in_w,
        f32 *weight, // (out_channels,in_channels,3,3)
        f32 *x, // (in_channels,in_h,in_w)
        f32 *out // (out_channels,out_h,out_w)
        );
void conv2d_f16(int in_channels, int out_channels, int kernel_size,
            int in_h, int in_w,
            f32 *weight, // (out_channels,in_channels,3,3)
            const f32 *bias, // (out_channels,)
            f16 *x, // (in_channels,in_h,in_w)
            f16 *out // (out_channels,out_h,out_w)
);
void relu(int in_channels, int in_h, int in_w,
        const f32 *x, // (in_channels,in_h,in_w)
        f32 *out // (in_channels,in_h,in_w)
        );
void relu_f16(
        int n,
        const f16 *x, // (n)
        f16 *out // (n)
        );
void relu_32K_f16(
        const f16 *x, // (32768)
        f16 *out // (32768)
        );
f32 max(int n, const f32 *x);
int argmax(int n, const f32 *x);
f32 sum(int n, const f32 *x);
void softmax(int n,
        f32 *x,  // (n,)
        f32 *out // (n,)
        );
void softmax_f16(int n,
             f16 *x,  // (n,)
             f16 *out // (n,)
);
void max_pool_2d(int in_channels, int in_h, int in_w,
        const f32 *x, // (in_channels,in_h,in_w)
        f32 *out // (in_channels,in_h/2,in_w/2)
        );
void max_pool_2d_f16(int in_channels, int in_h, int in_w,
                 const f16 *x, // (in_channels, in_h, in_w)
                 f16 *out // (in_channels, in_h/2, in_w/2)
        );
void batch_norm_2d(int in_channels, int in_h, int in_w,
        const f32 *x,
        f32 *out
        );
// out = matmul(A, x) + y
void saxpy(int m, int n,
        const f32 *A,  // (m, n)
        const f32 *x,  // (n,)
        const f32 *y,  // (m,)
        f32 *out // (m,)
        );
void saxpy_no_bias(int m, int n,
        const f32 *A,  // (m, n)
        const f32 *x,  // (n,)
        f32 *out // (m,)
        );
void saxpy_f16(int m, int n,
        const f32 *A,  // (m, n)
        const f16 *x,  // (n,)
        const f32 *y,  // (m,)
        f16 *out // (m,)
        );

void pad_32K_copy(
        int old_size,
        const f32 *x, // (old_size)
        f32 *out // (32768)
        );

void section_32K_copy(
        int new_size,
        const f32 *x, // (32768)
        f32 *out // (new_size)
        );

void cast_f32_f16(
        int n,
        const f32 *x, // (n)
        f16 *out // (n)
        );

void cast_32K_f32_f16(
        const f32 *x, // (32768)
        f16 *out // (32768)
        );

void cast_f16_f32(
        int n,
        const f16 *x, // (n)
        f32 *out // (n)
        );

void cast_32K_f16_f32(
        const f16 *x, // (32768)
        f32 *out // (32768)
        );


void test_linkage();

/*
 * For the specs of all operations, see the file
 * "test_beautiful.mnist.specs.txt".
 */





/*
 * There are arrays of rank 1, 2, 3, and 4 in the spec. No fancy
 * error management for now, just assert that everything is OK.
 */


/*
 * Our particular example gguf file:
 *
 * File read successfuly.
 * Magic:'GGUF'
 * Version: 3
 * Number of kv pairs: 1
 *     0: general.architecture = mnist-cnn
 *     Data Offset: 384
 *     Data Size:   139328
 *     Number of arrays: 6
 *       0: kernel1 ndim=4 shape=(32,1,3,3) type=f32 offset=0
 *           First few elements of f32 array:
 *               [0.146283, 0.033849, 0.200462, -0.395606, -0.251901]
 *       1: bias1 ndim=1 shape=(32,1,1,1) type=f32 offset=1152
 *           First few elements of f32 array:
 *               [-0.012165, -0.040562, -0.000860, -0.014375, -0.029884]
 *       2: kernel2 ndim=4 shape=(64,32,3,3) type=f32 offset=1280
 *           First few elements of f32 array:
 *               [0.085590, 0.054133, 0.127438, -0.014899, -0.134709]
 *       3: bias2 ndim=1 shape=(64,1,1,1) type=f32 offset=75008
 *           First few elements of f32 array:
 *               [-0.115071, -0.032611, 0.020584, -0.130953, 0.019709]
 *       4: dense_w ndim=2 shape=(10,1600,1,1) type=f32 offset=75264
 *           First few elements of f32 array:
 *               [-0.048662, 0.024389, -0.047063, 0.066041, 0.132236]
 *       5: dense_b ndim=1 shape=(10,1,1,1) type=f32 offset=139264
 *           First few elements of f32 array:
 *               [0.040235, 0.139814, -0.007806, -0.031162, -0.078248]
 */


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
