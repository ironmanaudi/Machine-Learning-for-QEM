from qiskit import *
import numpy as np
from qiskit import execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Kraus, SuperOp
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
import qiskit.providers.aer.noise as noise
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit import QuantumCircuit, execute
from qiskit import IBMQ, Aer
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel
from qiskit.circuit.random import random_circuit
import torch
import numpy as np

# display(plot_histogram(result_ideal.get_counts()))


def generate_data(size, shots, num_qubits, depth, max_operands, noise_model, basis_gates):
    circ = random_circuit(num_qubits, depth, max_operands, measure=True, seed=16)
    data_ideal = torch.zeros((size, num_qubits, 2), dtype=torch.float64)
    data_noisy = torch.zeros((size, num_qubits, 2), dtype=torch.float64)

    for i in range(size):
        for _ in range(int(shots/1024)):
            result_ideal = execute(circ, Aer.get_backend('qasm_simulator'),basis_gates=basis_gates).result()
            items_ideal = list(result_ideal.get_counts().items())
            result_noisy = execute(circ, Aer.get_backend('qasm_simulator'),basis_gates=basis_gates,noise_model=noise_model).result()
            items_noisy = list(result_noisy.get_counts().items())

            for k in range(len(items_ideal)):
                for j in range(num_qubits):
                    if items_ideal[k][0][j] == '0': data_ideal[i,j,0] += items_ideal[k][1]
                    else: data_ideal[i,j,1] += items_ideal[k][1]
            for l in range(len(items_noisy)):
                for m in range(num_qubits):
                    if items_noisy[l][0][m] == '0': data_noisy[i,m,0] += items_noisy[l][1]
                    else: data_noisy[i,m,1] += items_noisy[l][1]

    data_ideal = data_ideal / shots
    data_noisy = data_noisy/shots
    #data_noisy = torch.clamp(data_noisy, 1e-15, 1)
    #data_noisy = torch.log(data_noisy)

    return (data_ideal, data_noisy)


def noisy_model(prob_one, prob_two):
    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_one, 1)
    error_2 = noise.depolarizing_error(prob_two, 2)

    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    basis_gates = noise_model.basis_gates

    return (noise_model, basis_gates)


if __name__ == "__main__":
    size = 2
    shots = 2048
    num_qubits = 4
    depth = 3
    max_operands = 2
    prob_one =0.01
    prob_two = 0.01
    
    noise_model, basis_gates = noisy_model(prob_one, prob_two)
    data_ideal, data_noisy = generate_data(size, shots, num_qubits, depth, max_operands, noise_model, basis_gates)
    print('1', data_ideal)
    print('2', data_noisy)

