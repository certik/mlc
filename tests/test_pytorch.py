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

def test_ggml_mnist():
    class Test(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential (
                torch.nn.Conv2d(1, 32, 3, bias=True),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),
                torch.nn.Conv2d(32, 64, 3, bias=True),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),
                torch.nn.Flatten(0, -1),
                torch.nn.Linear(1600, 10, bias=True),
                torch.nn.Softmax(dim=0),
                )

        def forward(self, x):
            return self.model(x)

    py_model = Test()
    inp = torch.randn(1,28,28)
    print ("input dims: ", inp.size())

    torch_out = py_model(inp)
    print ("output dims: ", torch_out.size())
    print(torch_out.dtype)
    assert inp.shape == (1, 28, 28)
    assert torch_out.shape == (10,)
