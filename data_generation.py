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
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit import QuantumCircuit, execute
from qiskit import IBMQ, Aer
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel
from qiskit.circuit.random import random_circuit
import torch
import numpy as np

# display(plot_histogram(result_ideal.get_counts()))


def generate_data(size, shots, num_qubits, depth, max_operands, noisy_simulator):
    circ = random_circuit(num_qubits, depth, max_operands, measure=True)
    ideal_simulator = QasmSimulator()
    data_ideal = torch.zeros(size, num_qubits, 2)
    data_noisy = torch.zeros(size, num_qubits, 2)

    for i in range(size):
        for k in range(int(shots/1024)):
            result_ideal = execute(circ, ideal_simulator).result()
            items_ideal = list(result_ideal.get_counts().items())
            result_noisy = execute(circ, noisy_simulator).result()
            items_noisy = list(result_noisy.get_counts().items())

            for l in range(len(items)):
                for j in range(num_qubits):
                    if items_ideal[l][0][j] == '0': data_ideal[i,j,0] += items_ideal[l][1]
                    else: data_ideal[i,j,1] += items_ideal[l][1]
                    if items_noisy[l][0][j] == '0': data_noisy[i,j,0] += items_noisy[l][1]
                    else: data_noisy[i,j,1] += items_noisy[l][1]

    data_ideal = data_ideal / shots
    data_noisy = torch.log(data_noisy / shots)

    return (data_ideal, data_noisy)


# Example error probabilities
p_reset = 0.03
p_meas = 0.1
p_gate1 = 0.05

# QuantumError objects
error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
error_gate2 = error_gate1.tensor(error_gate1)

# Add errors to noise model
noise_bit_flip = NoiseModel()
noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])

# print(noise_bit_flip)

# Run the noisy simulation
noisy_simulator = QasmSimulator(noise_model=noise_bit_flip)
job = execute(circ, noisy_simulator)
result_bit_flip = job.result()
counts_bit_flip = result_bit_flip.get_counts(0)
# print('2',counts_bit_flip)

# Plot noisy output
# display(plot_histogram(counts_bit_flip))

if __name__ == "__main__":
    size = 1
    shots = 2048
    num_qubits = 4
    depth = 3
    max_operands = 2
    
    
    data = generate_data(size, shots, num_qubits, depth, max_operands, noisy_simulator)
    data_ideal, data_noisy = data
    print('1', data_ideal)
    print('2', data_noisy)

