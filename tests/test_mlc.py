from mlc.model import Operation

def test_op():
    o = Operation("matmul", 2, [100, 200])
    print(o)
