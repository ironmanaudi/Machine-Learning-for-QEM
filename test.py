import torch
import numpy as np

rnn = torch.nn.GRUCell(10, 20)
input = torch.randn((6, 3, 10))
c= torch.Tensor([[0.4,0.3]])


x = torch.randn(2, 3)
y = torch.cat((x, x, x, x), 0)
y = torch.reshape(y, (4,2,3))
print(y)
















