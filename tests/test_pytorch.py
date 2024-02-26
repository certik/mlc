import torch

def test_linear():
    class Test(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential (
                torch.nn.Linear(256, 2048, bias=False),
                torch.nn.Linear(2048, 768, bias=False)
                )

        def forward(self, x):
            x = self.model(x)
            return x

    py_model = Test()
    inp = torch.randn(100,256)
    print ("input dims: ", inp.size())

    torch_out = py_model(inp)
    print ("output dims: ", torch_out.size())
    print(torch_out.dtype)
    assert inp.shape == (100, 256)
    assert torch_out.shape == (100, 768)
