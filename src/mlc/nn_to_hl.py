from mlc import nnir, hlir

def create_matmul(A, B):
    if A.rank == 1:
        assert A.rank == 1
        assert B.rank == 2
        assert A.shape[0] == B.shape[0]
        return hlir.Operation(hlir.OpType.MatVec, rank=1,
                    shape=(B.shape[1],),
                    args=(A, B),
                    type=hlir.Type.f32,
                    execution_space=hlir.ExecutionSpace.host,
                    memory_space=hlir.MemorySpace.host
        )
    elif A.rank == 2:
        assert A.rank == 2
        assert B.rank == 2
        assert A.shape[1] == B.shape[0]
        return hlir.Operation(hlir.OpType.MatMul, rank=2,
                    shape=(A.shape[0], B.shape[1]),
                    args=(A, B),
                    type=hlir.Type.f32,
                    execution_space=hlir.ExecutionSpace.host,
                    memory_space=hlir.MemorySpace.host
        )
    else:
        raise Exception("Only rank 1 and 2 supported in matmul for now")

class NNToHLVisitor:

    def __init__(self, in_shape):
        self.hl = hlir.Array("Input", hlir.Type.f32, len(in_shape), in_shape,
                            hlir.MemorySpace.host)
        self.conv_counter = 0
        self.linear_counter = 0

    def visit(self, x):
        supported_nodes = ["Sequential", "Linear", "Conv2D", "ReLU", "Softmax",
                            "MaxPool2D", "Flatten", "BatchNorm2D"]
        node_name = type(x).__name__
        if node_name in supported_nodes:
            eval("self.visit_%s(x)" % node_name)
        else:
            raise Exception("Unsupported NN IR node: %s" % node_name)

    def visit_Linear(self, x: nnir.Linear):
        self.linear_counter += 1
        weight = hlir.Array("linear_w%d" % self.linear_counter, hlir.Type.f32,
                            2, (x.in_features, x.out_features),
                            hlir.MemorySpace.host)
        self.hl = create_matmul(self.hl, weight)
        if x.bias:
            bias = hlir.Array("linear_b%d" % self.linear_counter, hlir.Type.f32,
                            self.hl.rank, self.hl.shape,
                            hlir.MemorySpace.host)
            self.hl = hlir.Operation(hlir.OpType.Add, rank=self.hl.rank,
                            shape=self.hl.shape,
                            args=(self.hl, bias),
                            type=hlir.Type.f32,
                            execution_space=hlir.ExecutionSpace.host,
                            memory_space=hlir.MemorySpace.host
            )

    def visit_Conv2D(self, x: nnir.Conv2D):
        assert self.hl.rank == 2 or self.hl.rank == 3
        if self.hl.rank == 2:
            in_channels = 1
        else:
            assert self.hl.rank == 3
            in_channels = self.hl.shape[2]
        assert in_channels == x.in_channels

        new_shape = list(self.hl.shape)[:]
        new_shape[0] -= (x.kernel_size-1)
        new_shape[1] -= (x.kernel_size-1)
        if len(new_shape) == 2:
            new_shape.append(x.out_channels)
        else:
            new_shape[2] = x.out_channels

        self.conv_counter += 1
        kernel = hlir.Array("conv_kernel%d" % self.conv_counter, hlir.Type.f32,
            2, (x.kernel_size, x.kernel_size, x.in_channels, x.out_channels),
            hlir.MemorySpace.host)
        self.hl = hlir.Operation(hlir.OpType.Conv2D,
                    (kernel, self.hl),
                    hlir.ExecutionSpace.host,
                    type=hlir.Type.f32,
                    rank=len(new_shape),
                    shape=new_shape,
                    memory_space=hlir.MemorySpace.host
        )
        if x.bias:
            bias = hlir.Array("conv_b%d" % self.conv_counter, hlir.Type.f32,
                            self.hl.rank, self.hl.shape,
                            hlir.MemorySpace.host)
            self.hl = hlir.Operation(hlir.OpType.Add, rank=self.hl.rank,
                            shape=self.hl.shape,
                            args=(self.hl, bias),
                            type=hlir.Type.f32,
                            execution_space=hlir.ExecutionSpace.host,
                            memory_space=hlir.MemorySpace.host
            )

    def visit_ReLU(self, x: nnir.ReLU):
        self.hl = hlir.Operation(hlir.OpType.ReLU,
                    args=(self.hl,),
                    execution_space=hlir.ExecutionSpace.host,
                    type=hlir.Type.f32,
                    rank=self.hl.rank,
                    shape=self.hl.shape,
                    memory_space=hlir.MemorySpace.host
        )

    def visit_Softmax(self, x: nnir.Softmax):
        self.hl = hlir.Operation(hlir.OpType.Softmax, rank=self.hl.rank,
                    shape=self.hl.shape,
                    args=(self.hl,),
                    type=hlir.Type.f32,
                    execution_space=hlir.ExecutionSpace.host,
                    memory_space=hlir.MemorySpace.host
        )

    def visit_MaxPool2D(self, x: nnir.MaxPool2D):
        assert self.hl.rank >= 2
        new_shape = list(self.hl.shape)[:]
        new_shape[0] //= 2
        new_shape[1] //= 2
        self.hl = hlir.Operation(hlir.OpType.MaxPool2D,
                    rank=self.hl.rank,
                    shape=new_shape,
                    args=(self.hl,),
                    type=hlir.Type.f32,
                    execution_space=hlir.ExecutionSpace.host,
                    memory_space=hlir.MemorySpace.host
        )

    def visit_BatchNorm2D(self, x: nnir.BatchNorm2D):
        assert self.hl.rank >= 2
        self.hl = hlir.Operation(hlir.OpType.BatchNorm2D,
                    rank=self.hl.rank,
                    shape=self.hl.shape,
                    args=(self.hl,),
                    type=hlir.Type.f32,
                    execution_space=hlir.ExecutionSpace.host,
                    memory_space=hlir.MemorySpace.host
                    )

#    def visit_Transpose(self, x: nnir.Transpose):
#        new_shape = []
#        for i in range(len(self.hl.shape)):
#            new_shape.append(self.hl.shape[x.permutation[i]])
#        self.hl = hlir.Operation("Transpose", self.hl.rank,
#                    shape=new_shape,
#                    args=(self.hl,))

    def visit_Flatten(self, x: nnir.Flatten):
        assert x.start_dim == 1
        assert x.end_dim == -1
        s = 1
        for i in range(len(self.hl.shape)):
            s = s*self.hl.shape[i]
        self.hl = hlir.Operation(hlir.OpType.Reshape, rank=1,
                    shape=(s,),
                    args=(self.hl,),
                    type=hlir.Type.f32,
                    execution_space=hlir.ExecutionSpace.host,
                    memory_space=hlir.MemorySpace.host
                )

    def visit_Sequential(self, x: nnir.Sequential):
        for layer in x.layers:
            self.visit(layer)

def nn_to_hl(nn: nnir.Sequential, in_shape: list[int]):
    v = NNToHLVisitor(in_shape)
    v.visit_Sequential(nn)
    return v.hl
