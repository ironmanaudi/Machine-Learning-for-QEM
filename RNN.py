from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import math
import torch 
import os
from torch.autograd import Variable
import time
import torch.utils.data as Data
from torch import autograd
from torch import Tensor
from torch.nn import Parameter as Param
import inspect
from torch.nn import Parameter
import data_generation as dg

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(precision=None, threshold=5000, edgeitems=None, linewidth=None, profile=None)

#define parameters
num_qubits = 4
depth = 3
max_operands = 2
input_size = 2
hidden_size = 2
single_error = 1e-3
double_error = 1e-3
measure = 1024

#define hyperparameters
lr = 3e-4
batch_num = 30
size = 128 #batch size
shots = 8192 #sampling shots
num_qubits = 4
depth = 3
max_operands = 2
prob_one =0.05
prob_two = 0.1

#generate training and testing data
train_ideal, train_noisy = data_load(batch_num, size, shots, num_qubits, depth, max_operands, prob_one, prob_two)
test_ideal, test_noisy = data_load(batch_num=1, size, shots, num_qubits, depth, max_operands, prob_one, prob_two)


class QEM(torch.nn.Module):
    def __init__(self, num_qubits):
        super(QEM, self).__init__()
        self.num_qubits = num_qubits
        self.output = []
        self.rnn = torch.nn.GRUCell(num_qubits, num_qubits)
    
    def forward(self, data):
        hx = torch.randn(BATCH_SIZE, self.num_qubits)
        
        for i in range(num_qubits):
            hx = self.rnn(data[i], hx)
            
            #transform hx to y
            y = torch.log(hx)
            
            self.output.append(y)
        
        return self.output

device = torch.device('cpu')
mitigator = QEM().to(device)
optimizer = torch.optim.Adam(mitigator.parameters(), lr, weight_decay=0)
criterion = torch.nn.KLDivLoss()


def train(epoch):
    mitigator.train()
    
    for datas in train_loader:
        datas = datas.to(device)
        optimizer.zero_grad()
        loss = criterion(mitigator(datas), datas) #input change
        loss.backward(retain_graph=True)
        optimizer.step()
        
    if epoch % 1 == 0:
        torch.save(mitigator.state_dict(), './model/model_parameters_epoch%d.pkl' % (epoch))
        
    return loss

def test(model_a):
    model_a.eval()
    loss = 0
    for datas in test_loader:
        datas = datas.to(device)
        pred = model_a(datas)
        
        loss += criterion(pred, datas).item()
        
    return loss

    
if __name__ == '__main__':
    training = 1

