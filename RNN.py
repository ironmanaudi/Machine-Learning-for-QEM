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
import data_loader as dl

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(precision=None, threshold=5000, edgeitems=None, linewidth=None, profile=None)


#define parameters
batch_num = 30
batch_num_test = 5
size = 128 #batch size
shots = 8192 #sampling shots
num_qubits = 4
depth = 10
max_operands = 2
prob_one = 6.5*1e-4
prob_two = 1.65*1e-2

#define hyperparameters
lr = 3e-4
input_size = 2
hidden_size = 2

#load training and testing data
train_ideal = torch.load("./train_set/train_ideal.pth")
train_noisy = torch.load("./train_set/train_noisy.pth")
test_ideal = torch.load("./test_set/test_ideal.pth")
test_noisy = torch.load("./test_set/test_noisy.pth")



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
        hx = torch.randn(size, 2).double()
        flag = 1

        for i in range(num_qubits):
            hx = self.rnn(data[i], hx)
            
            #transform hx to y
            y = torch.mm(hx, self.W) + self.alpha
            y = torch.nn.functional.log_softmax(input=y, dim=1, dtype=torch.float64)
            #y = torch.clamp(y, 1e-14, 1)

            #y = torch.log(hx)
            #print('2',y)

            if flag:output = y
            else:output = torch.cat((output, y), 0)
            flag = 0

        output = torch.reshape(output, (self.num_qubits, size, 2))
        output = torch.transpose(output, 0, 1)

        return output

device = torch.device('cpu')
mitigator = QEM(num_qubits).to(device)
optimizer = torch.optim.Adam(mitigator.parameters(), lr, weight_decay=0)
criterion = torch.nn.KLDivLoss(reduction='sum').double()

def train(epoch):
    mitigator.train()
    
    for data_ideal, data_noisy in zip(train_ideal, train_noisy):
        data_ideal = data_ideal.to(device)
        data_noisy = data_noisy.to(device)
        optimizer.zero_grad()
        data_ideal = torch.clamp(data_ideal, 1e-14, 1)
        output = mitigator(data_noisy)
        loss = criterion(output, data_ideal) #input change
        loss.backward(retain_graph=True)
        optimizer.step()
        
    if epoch % 1 == 0:
        torch.save(mitigator.state_dict(), './model/model_parameters_epoch%d.pkl' % (epoch))
        
    return loss

def test(model_a, training):
    model_a.eval()
    loss = 0
    loss2 = 0 

    for data_ideal, data_noisy in zip(test_ideal, test_noisy):
        data_ideal = data_ideal.to(device)
        data_noisy = data_noisy.to(device)
        pred = model_a(data_noisy)
        data_ideal = torch.clamp(data_ideal, 1e-14, 1)

        loss += criterion(pred, data_ideal).item()
        if training:return loss/ (size*batch_num_test)
        else:
            data_noisy = torch.clamp(data_noisy, 1e-14, 1)
            data_noisy = torch.log(data_noisy)
            loss2 += criterion(data_noisy, data_ideal).item()

            return (loss/(size*batch_num_test), loss2/(size*batch_num_test))

    
if __name__ == '__main__':
    training = 1

    if training:
        f = open('./mitigator_training_loss.txt','a')
        N = 4801
        for epoch in range(1, N):
            train(epoch)
            test_acc = test(mitigator, training)
            f.write('Epoch: {:03d}, Test Acc: {:.10f}'.format(epoch, test_acc))
            print('Epoch: {:03d}, Test Acc: {:.10f}'.format(epoch, test_acc))
        f.close()
    
    else:
        model_a = QEM(num_qubits).to(device)
        model_a.load_state_dict(torch.load('./trained/model_parameters_epoch400.pkl'))
        loss, loss2 = test(model_a, training)
        print(loss, loss2)


