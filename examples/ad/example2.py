import numpy as np
import torch
np.random.seed(1)
w0 = np.random.randn(5, 3).astype(np.float32)
b0 = np.random.randn(3).astype(np.float32)

################################################################################
# PyTorch

x = torch.ones(5)  # input array
y = torch.zeros(3)  # expected output
w = torch.tensor(w0, requires_grad=True)
b = torch.tensor(b0, requires_grad=True)
z = torch.matmul(x, w)+b
loss1 = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
loss2 = torch.nn.functional.l1_loss(z, y)
loss3 = torch.sum(torch.abs(z-y))
print("PyTorch")
print(loss1)
print(loss2)
print(loss3)

print("Derivatives")
loss3.backward()
print(w.grad)
print(b.grad)

print()

################################################################################
# NumPy

x = np.ones(5)  # input array
y = np.zeros(3)  # expected output
w = w0
b = b0
z = np.matmul(x, w) + b
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def binary_cross_entropy_with_logits(a, b):
    return -np.mean(b * np.log(sigmoid(a)) + (1 - b) * np.log(1 - sigmoid(a)))
loss1 = binary_cross_entropy_with_logits(z, y)
loss2 = np.mean(np.abs(z-y))
loss3 = np.sum(np.abs(z-y))
print("NumPy")
print(loss1)
print(loss2)
print(loss3)
