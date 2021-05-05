import data_generation as dg
import clifford_generation as cg
import numpy as np
import torch


def data_load(batch_num, size, shots, num_qubits, depth, max_operands, prob_one, prob_two, clifford, device=0, train=1):
    loader_ideal = []
    loader_noisy = []
    noise_model, basis_gates = dg.noisy_model(prob_one, prob_two)
    cbasis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    length = []
    
    for group in range(batch_num):
        if clifford:
            if device:cg.generate_data(size, shots, num_qubits, cbasis_gates, device, group, batch_num, train)
            else:
                data_ideal, data_noisy, sizes = cg.generate_data(size, shots, num_qubits, cbasis_gates, device, group, batch_num, train)
                loader_ideal.append(data_ideal)
                loader_noisy.append(data_noisy)
                length.append(sizes)
        else:
            data_ideal, data_noisy, sizes = dg.generate_data(size, shots, num_qubits, depth, max_operands, noise_model, basis_gates)
            loader_ideal.append(data_ideal)
            loader_noisy.append(data_noisy)
            length.append(sizes)

    return (loader_ideal, loader_noisy, length)


if __name__ == "__main__":
    batch_num = 20
    batch_num_test = 5
    size = 75
    shots = 8192
    num_qubits = 5
    depth = 10
    max_operands = 2
    prob_one = 6.5*1e-4
    prob_two = 1.65*1e-2
    clifford = 1
    if clifford==1:device = 0
    else:device = 0
    train = 1
    
    if clifford:
        if device:
            if train:
                #generate training data on quantum device
                data_load(batch_num, size, shots, num_qubits, depth, max_operands, prob_one, prob_two, clifford, device, train)
            else:
                data_load(batch_num_test, size, shots, num_qubits, depth, max_operands, prob_one, prob_two, clifford, device, train)
        else:#not using quantum device, so separate to reading train & test phases
            if train:
                train_ideal, train_noisy, length = data_load(batch_num, size, shots, num_qubits, depth, max_operands, prob_one, prob_two, clifford, device, train)
                torch.save(train_ideal, "./ctrain_set/in{n}.pth".format(n=num_qubits))
                torch.save(train_noisy, "./ctrain_set/nn{n}.pth".format(n=num_qubits))
                torch.save(length, "./ctrain_set/sz{n}.pth".format(n=num_qubits))
            else:
                test_ideal, test_noisy, length = data_load(batch_num_test, size, shots, num_qubits, depth, max_operands, prob_one, prob_two, clifford, device, train)
                torch.save(test_ideal, "./ctest_set/in{n}.pth".format(n=num_qubits))
                torch.save(test_noisy, "./ctest_set/nn{n}.pth".format(n=num_qubits))
                torch.save(length, "./ctest_set/sz{n}.pth".format(n=num_qubits))
    else:
        #generate training and testing data
        train_ideal, train_noisy, train_length = data_load(batch_num, size, shots, num_qubits, depth, max_operands, prob_one, prob_two, clifford, device)
        test_ideal, test_noisy, test_length = data_load(batch_num_test, size, shots, num_qubits, depth, max_operands, prob_one, prob_two, clifford, device)
        torch.save(train_ideal, "./train_set/in{n}l{d}.pth".format(n=num_qubits, d=depth))
        torch.save(train_noisy, "./train_set/nn{n}l{d}.pth".format(n=num_qubits, d=depth))
        torch.save(train_length, "./train_set/sz{n}l{d}.pth".format(n=num_qubits, d=depth))
        torch.save(test_ideal, "./test_set/in{n}l{d}.pth".format(n=num_qubits, d=depth))
        torch.save(test_noisy, "./test_set/nn{n}l{d}.pth".format(n=num_qubits, d=depth))
        torch.save(test_length, "./test_set/sz{n}l{d}.pth".format(n=num_qubits, d=depth))
    
    print('done')


