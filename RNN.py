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
batch_num_test = 10
size = 75 #batch size
shots = 8192 #sampling shots
num_qubits = 6
depth = 10
max_operands = 2
prob_one = 6.5*1e-4
prob_two = 1.65*1e-2

#define hyperparameters
lr = 3e-4
input_size = 2
hidden_size = 2

#load training and testing data
#train_ideal = torch.load("./train_set/in{n}l{d}.pth".format(n=num_qubits, d=depth))
#train_noisy = torch.load("./train_set/nn{n}l{d}.pth".format(n=num_qubits, d=depth))
train_ideal = torch.load("./train_set/in5l10.pth")
train_noisy = torch.load("./train_set/nn5l10.pth")
train_sizes = torch.load("./train_set/sz5l10.pth")

test_ideal = torch.load("./test_set/in{n}l{d}.pth".format(n=num_qubits, d=depth))
test_noisy = torch.load("./test_set/nn{n}l{d}.pth".format(n=num_qubits, d=depth))
test_sizes = torch.load("./test_set/sz5l10.pth")


class QEM(torch.nn.Module):
    def __init__(self, num_qubits):
        super(QEM, self).__init__()
        self.num_qubits = num_qubits
        hidden_size = self.num_qubits*2
        # self.rnn = torch.nn.GRUCell(2, 2).double()
        # self.W = torch.nn.Parameter(Variable(torch.ones((2, 2)).double()))
        # self.alpha = torch.nn.Parameter(Variable(torch.Tensor([[0, 0]]).double()))
        self.rnn = torch.nn.LSTM(input_size=(self.num_qubits+1), hidden_size=hidden_size, bidirectional=True).double()
        self.linear = torch.nn.Linear(2*hidden_size, 1).double()
        self.sm = torch.nn.Softmax(dim=1).double()
        
    def forward(self, data, lengths):
        output_pack, (hn, cn) = self.rnn(data)
        
        #transform output
        output = self.linear(output_pack.data)
        output_pack = torch.nn.utils.rnn.PackedSequence(output, output_pack.batch_sizes, output_pack.sorted_indices, output_pack.unsorted_indices)
        
        #unpack output
        output, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(output_pack, batch_first=True, padding_value=-1e10)
        
        #sm & pack
        output = self.sm(output)
        output_pack = torch.nn.utils.rnn.pack_padded_sequence(output, lengths=lengths, batch_first=True, enforce_sorted=False)
        
        output = output_pack.data
        output = output.view(-1)
        output = torch.log(output)
        
        return output


device = torch.device('cpu')
mitigator = QEM(num_qubits).to(device)
optimizer = torch.optim.Adam(mitigator.parameters(), lr, weight_decay=0)
criterion = torch.nn.KLDivLoss(reduction='sum').double()


def train(epoch):
    mitigator.train()
    
    for data_ideal, data_noisy, sizes in zip(train_ideal, train_noisy, train_sizes):
        data_ideal = data_ideal.to(device)
        data_noisy = data_noisy.to(device)
        optimizer.zero_grad()
        data_ideal = torch.clamp(data_ideal, 1e-14, 1)
        output = mitigator(data_noisy, sizes)
        loss = criterion(output, data_ideal) #input change
        loss.backward(retain_graph=True)
        optimizer.step()
        
    if epoch % 10 == 0:
        torch.save(mitigator.state_dict(), './model7/model_parameters_epoch%d.pkl' % (epoch))
        
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
    training = 0

    if training:
        #f = open('./mitigator_training_loss.txt','a')
        N = 4801
        for epoch in range(1, N):
            train(epoch)
            test_acc = test(mitigator, training)
            #f.write('Epoch: {:03d}, Test Acc: {:.10f}'.format(epoch, test_acc))
            print('Epoch: {:03d}, Test Acc: {:.10f}'.format(epoch, test_acc))
        #f.close()
    
    else:
        model_a = QEM(num_qubits).to(device)
        model_a.load_state_dict(torch.load('./trained/model_parameters_7.pkl'))
        #model_a.load_state_dict(torch.load('./model7/model_parameters_epoch4800.pkl'))

        loss, loss2 = test(model_a, training)
        print(loss, loss2)


