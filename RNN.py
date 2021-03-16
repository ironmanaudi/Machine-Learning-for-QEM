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
batch_num = 30
batch_num_test = 10
size = 128 #batch size
shots = 8192 #sampling shots
num_qubits = 4
depth = 3
max_operands = 2
prob_one =0.05
prob_two = 0.1

#define hyperparameters
lr = 3e-4
input_size = 2
hidden_size = 2


#generate training and testing data
train_ideal, train_noisy = data_load(batch_num, size, shots, num_qubits, depth, max_operands, prob_one, prob_two)
test_ideal, test_noisy = data_load(batch_num_test, size, shots, num_qubits, depth, max_operands, prob_one, prob_two)


class QEM(torch.nn.Module):
    def __init__(self, num_qubits):
        super(QEM, self).__init__()
        self.num_qubits = num_qubits
        self.rnn = torch.nn.GRUCell(2, 2).double()
        self.W = torch.nn.Parameter(Variable(torch.ones((2, 2)).double()))
        self.alpha = torch.nn.Parameter(Variable(torch.Tensor([[0, 0]]).double()))

    def forward(self, data):
        #reshape data
        data = torch.transpose(data, 0, 1)
        hx = torch.randn(size, 2)
        flag = 1

        for i in range(num_qubits):
            hx = self.rnn(data[i], hx)
            
            #transform hx to y
            y = torch.mm(self.W, hx) + self.alpha
            y = torch.nn.functional.softmax(input=y, dim=1, dtype=torch.float64)
            y = torch.clamp(y, 1e-15, 1)
            y = torch.log(hx)
            
            if flag:output = y
            else:torch.cat((output, y), 0)
            flag = 0
        
        output = torch.reshape(output, (self.num_qubits, size, 2))
        output = torch.transpose(output, 0, 1)

        return output

device = torch.device('cpu')
mitigator = QEM().to(device)
optimizer = torch.optim.Adam(mitigator.parameters(), lr, weight_decay=0)
criterion = torch.nn.KLDivLoss()


def train(epoch):
    mitigator.train()
    
    for data_ideal, data_noisy in zip(train_ideal, train_noisy):
        data_ideal = data_ideal.to(device)
        data_noisy = data_noisy.to(device)
        optimizer.zero_grad()
        loss = criterion(mitigator(data_noisy), data_ideal) #input change
        loss.backward(retain_graph=True)
        optimizer.step()
        
    if epoch % 1 == 0:
        torch.save(mitigator.state_dict(), './model/model_parameters_epoch%d.pkl' % (epoch))
        
    return loss

def test(model_a):
    model_a.eval()
    loss = 0
    for data_ideal, data_noisy in zip(test_ideal, test_noisy):
        data_ideal = data_ideal.to(device)
        data_noisy = data_noisy.to(device)
        pred = model_a(data_noisy)
        
        loss += criterion(pred, data_ideal).item()
        
    return loss

    
if __name__ == '__main__':
    training = 1

