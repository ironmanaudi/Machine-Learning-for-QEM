import torch
import numpy as np

rnn = torch.nn.GRUCell(10, 20)
input = torch.randn((6, 3, 10))
c= torch.Tensor([[0.4,0.3]])


x = torch.randn(2, 3)
y = torch.cat((x, x, x, x), 0)
y = torch.reshape(y, (4,2,3))
#print(y)



import torch
import torch.nn.functional as F
# 定义两个矩阵
x = torch.randn((4, 5))
y = torch.randn((4, 5))
# 因为要用y指导x,所以求x的对数概率，y的概率
logp_x = F.log_softmax(x, dim=-1)
p_y = F.softmax(y, dim=-1)
 
kl_sum = F.kl_div(logp_x, p_y, reduction='sum')
kl_mean = F.kl_div(logp_x, p_y, reduction='mean')
   
a = torch.ones(5)*0.5


b = torch.load("./myDict.pth")
print(b)







