import data_generation as dg
import numpy as np
import torch

def data_load(batch_num, size, shots, num_qubits, depth, max_operands, prob_one, prob_tow):
    loader_ideal = []
    loader_noisy = []
    
    if batch_num == 1:
        noise_model, basis_gates = noisy_model(prob_one, prob_two)
        data_ideal, data_noisy = generate_data(size, shots, num_qubits, depth, max_operands, noise_model, basis_gates)
        return (data_ideal, data_noisy)
    else:
        for _ in range(batch_num):
            noise_model, basis_gates = noisy_model(prob_one, prob_two)
            data_ideal, data_noisy = generate_data(size, shots, num_qubits, depth, max_operands, noise_model, basis_gates)
            loader_ideal.append(data_ideal)
            loader_noisy.append(data_noisy)

        return (loader_ideal, loader_noisy)

