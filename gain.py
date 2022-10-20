import torch
from torch import nn

x = torch.randn(10, 100) * 2
x.requires_grad = True
bn = nn.BatchNorm1d(100)

y = bn(x)
# y = x + 1

print(torch.std_mean(x))
print(torch.std_mean(y))

y[0].sum().backward()
print(x.grad[1])
