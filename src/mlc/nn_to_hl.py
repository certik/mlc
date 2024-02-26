from mlc import nnir, hlir

def create_matmul(A, B):
    assert A.rank == 2
    assert B.rank == 2
    assert A.shape[1] == B.shape[0]
    return hlir.Operation("matmul", 2,
                shape=(A.shape[0], B.shape[1]),
                args=(A, B))

class NNToHLVisitor:

    def __init__(self, in_shape):
        self.hl = None
        self.in_shape = in_shape

    def visit(self, x):
        if isinstance(x, nnir.Sequential):
            self.visit_Sequential(x)
        elif isinstance(x, nnir.Linear):
            self.visit_Linear(x)
        else:
            raise Exception("Unsupported NN IR node: %s" % type(x).__name__)

    def visit_Linear(self, x: nnir.Linear):
        assert x.bias == False
        weight = hlir.Array(2, (x.in_features, x.out_features))
        self.hl = create_matmul(self.hl, weight)

    def visit_Sequential(self, x: nnir.Sequential):
        self.hl = hlir.Array(2, self.in_shape)
        for layer in x.layers:
            self.visit(layer)

    def visit_XX():
        pass

def nn_to_hl(nn: nnir.Sequential, in_shape: list[int]):
    v = NNToHLVisitor(in_shape)
    v.visit_Sequential(nn)
    return v.hl
