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
