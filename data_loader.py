import data_generation as dg
import numpy as np
import torch

def data_load(batch_num, size, shots, num_qubits, depth, max_operands, prob_one, prob_two):
    loader_ideal = []
    loader_noisy = []
    
    if batch_num == 1:
        noise_model, basis_gates = dg.noisy_model(prob_one, prob_two)
        data_ideal, data_noisy = dg.generate_data(size, shots, num_qubits, depth, max_operands, noise_model, basis_gates)
        return (data_ideal, data_noisy)
    else:
        for _ in range(batch_num):
            noise_model, basis_gates = dg.noisy_model(prob_one, prob_two)
            data_ideal, data_noisy = dg.generate_data(size, shots, num_qubits, depth, max_operands, noise_model, basis_gates)
            loader_ideal.append(data_ideal)
            loader_noisy.append(data_noisy)

        return (loader_ideal, loader_noisy)

if __name__ == "__main__":
    batch_num = 30
    batch_num_test = 10
    size = 128
    shots = 8192
    num_qubits = 7
    depth = 10
    max_operands = 2
    prob_one = 6.5*1e-4
    prob_two = 1.65*1e-2
    
    #generate training and testing data
    #train_ideal, train_noisy = data_load(batch_num, size, shots, num_qubits, depth, max_operands, prob_one, prob_two)
    test_ideal, test_noisy = data_load(batch_num_test, size, shots, num_qubits, depth, max_operands, prob_one, prob_two)
    
    #torch.save(train_ideal, "./train_set/in4l10.pth")
    #torch.save(train_noisy, "./train_set/nn5l10.pth")
    torch.save(test_ideal, "./test_set/in{n}l{d}.pth".format(n=num_qubits, d=depth))
    torch.save(test_noisy, "./test_set/nn{n}l{d}.pth".format(n=num_qubits, d=depth))

    print('done')


