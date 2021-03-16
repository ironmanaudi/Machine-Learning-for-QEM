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
    batch_num = 2
    size = 2
    shots = 2048
    num_qubits = 4
    depth = 3
    max_operands = 2
    prob_one =0.01
    prob_two = 0.01
    
    loader_ideal, loader_noisy = data_load(batch_num, size, shots, num_qubits, depth, max_operands, prob_one, prob_two)
    
    print('1', loader_ideal)
    print('2', loader_noisy)


