import numpy as np
import torch
np.random.seed(1)
x0 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y0 = np.zeros(3, dtype=np.float32)
w0 = np.random.randn(5, 3).astype(np.float32)
b0 = np.random.randn(3).astype(np.float32)

################################################################################
# PyTorch

x = torch.tensor(x0)  # input array
y = torch.tensor(y0)  # expected output
w = torch.tensor(w0, requires_grad=True)
b = torch.tensor(b0, requires_grad=True)
z = torch.matmul(x, w)+b
#z.retain_grad()
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

# d|x|/dx = sign(x)

# L3 = sum |z-y|
# z = X W + b
# z = (z1, z2, ..., zn) = (z^i)
# W = (w^ij)
# X = (x^i)
# ∂(XW)^k/∂w^ij = ∂x_lw^lk/∂w^ij = x_l ∂w^lk/∂w^ij = x_l delta_li delta_kj
#     = x_i delta_kj

# ∂L3/∂z1 = ∂|z1-y1|/∂z1 = sign(z1)
# Index notation:
# ∂L3/∂zi = ∂sum|z-y|/∂zi = sign(zi)
# Vector notation:
# ∂L3/∂z = ∂sum|z-y|/∂z = sign(z)

# Index notation:
# ∂L3/∂bi = ∂L3/∂z^j * ∂z^j/∂b^i = sign(z^j) * delta_{ji} = sign(zi)
# Vector notation:
# ∂L3/∂b = ∂L3/∂z * ∂z/∂b = sign(z) * 1

# Index notation:
# ∂L3/∂wij = ∂L3/∂z^k * ∂z^k/∂w^ij = sign(z^k) * x_i delta_{kj}
#     = x_i * sign(z^j)

print()

################################################################################
# NumPy

x = x0  # input array
y = y0  # expected output
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
