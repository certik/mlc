from mlc import nnir, hlir

def create_matmul(A, B):
    if A.rank == 1:
        assert A.rank == 1
        assert B.rank == 2
        assert A.shape[0] == B.shape[0]
        return hlir.Operation("MatVec", 1,
                    shape=(B.shape[1],),
                    args=(A, B))
    elif A.rank == 2:
        assert A.rank == 2
        assert B.rank == 2
        assert A.shape[1] == B.shape[0]
        return hlir.Operation("MatMul", 2,
                    shape=(A.shape[0], B.shape[1]),
                    args=(A, B))
    else:
        raise Exception("Only rank 1 and 2 supported in matmul for now")

class NNToHLVisitor:

    def __init__(self, in_shape):
        self.hl = hlir.Array("Input", len(in_shape), in_shape)
        self.kernel_counter = 0
        self.weight_counter = 0

    def visit(self, x):
        supported_nodes = ["Sequential", "Linear", "Conv2D", "ReLU", "Softmax",
                            "MaxPool2D", "Transpose", "Flatten", "BatchNorm2D"]
        node_name = type(x).__name__
        if node_name in supported_nodes:
            eval("self.visit_%s(x)" % node_name)
        else:
            raise Exception("Unsupported NN IR node: %s" % node_name)

    def visit_Linear(self, x: nnir.Linear):
        self.weight_counter += 1
        weight = hlir.Array("Weight%d" % self.weight_counter, 2,
                            (x.in_features, x.out_features))
        self.hl = create_matmul(self.hl, weight)
        if x.bias:
            bias = hlir.Array("Bias%d" % self.weight_counter, self.hl.rank,
                            self.hl.shape)
            self.hl = hlir.Operation("Add", self.hl.rank,
                            self.hl.shape,
                            args=(self.hl, bias))

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

        self.kernel_counter += 1
        kernel = hlir.Array("Kernel%d" % self.kernel_counter, 2,
                            (x.out_channels, x.in_channels,
                                x.kernel_size, x.kernel_size))
        self.hl = hlir.Operation("Conv2D", len(new_shape),
                    shape=new_shape,
                    args=(self.hl, kernel))
        if x.bias:
            bias = hlir.Array("Bias%d" % self.weight_counter, self.hl.rank,
                            self.hl.shape)
            self.hl = hlir.Operation("Add", self.hl.rank,
                            self.hl.shape,
                            args=(self.hl, bias))

    def visit_ReLU(self, x: nnir.ReLU):
        self.hl = hlir.Operation("ReLU", self.hl.rank,
                    shape=self.hl.shape,
                    args=(self.hl,))

    def visit_Softmax(self, x: nnir.Softmax):
        self.hl = hlir.Operation("Softmax", self.hl.rank,
                    shape=self.hl.shape,
                    args=(self.hl,))

    def visit_MaxPool2D(self, x: nnir.MaxPool2D):
        assert self.hl.rank >= 2
        new_shape = list(self.hl.shape)[:]
        new_shape[0] //= 2
        new_shape[1] //= 2
        self.hl = hlir.Operation("MaxPool2D", self.hl.rank,
                    shape=new_shape,
                    args=(self.hl,))

    def visit_BatchNorm2D(self, x: nnir.BatchNorm2D):
        assert self.hl.rank >= 2
        self.hl = hlir.Operation("BatchNorm2D", self.hl.rank,
                    shape=self.hl.shape,
                    args=(self.hl,))

    def visit_Transpose(self, x: nnir.Transpose):
        new_shape = []
        for i in range(len(self.hl.shape)):
            new_shape.append(self.hl.shape[x.permutation[i]])
        self.hl = hlir.Operation("Transpose", self.hl.rank,
                    shape=new_shape,
                    args=(self.hl,))

    def visit_Flatten(self, x: nnir.Flatten):
        assert x.start_dim == 1
        assert x.end_dim == -1
        s = 1
        for i in range(len(self.hl.shape)):
            s = s*self.hl.shape[i]
        self.hl = hlir.Operation("Reshape", 1,
                    shape=(s,),
                    args=(self.hl,))

    def visit_Sequential(self, x: nnir.Sequential):
        for layer in x.layers:
            self.visit(layer)

def nn_to_hl(nn: nnir.Sequential, in_shape: list[int]):
    v = NNToHLVisitor(in_shape)
    v.visit_Sequential(nn)
    return v.hl
