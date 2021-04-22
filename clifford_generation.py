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
from qiskit.quantum_info import random_clifford
import pickle
# display(plot_histogram(result_ideal.get_counts()))


def generate_data(size, shots, num_qubits, basis_gates, device, group=-1, limit=0, train=1):
    data_ideal = torch.zeros((size, num_qubits, 2), dtype=torch.float64)
    data_noisy = torch.zeros((size, num_qubits, 2), dtype=torch.float64)
    circ_set = []

    backend_1 = Aer.get_backend('qasm_simulator')
    #IBMQ.save_account('83a95fc7efba05f250ed50ff6bdf1638541ebd4f9e46396f26c8dc12f15b2331ea24d9db4f59c30b6e397c2268e40ddc1dae4772091baa73d7e99c91e8d8c56b')
    provider = IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    backend_2 = provider.get_backend('ibmq_16_melbourne')
    
    if device:
        for i in range(size):
            qr = QuantumRegister(num_qubits, 'q')
            cr = ClassicalRegister(num_qubits, 'c')
            circ = QuantumCircuit(qr,cr)
            cliff = random_clifford(num_qubits)
            cliff = cliff.to_circuit()
            circ = circ.compose(cliff,inplace=False)
            circ.measure(qr[:],cr[:])
            circ_set.append(circ)
        execute(circ_set,backend_2,basis_gates=basis_gates,shots=shots)
        output = open('./circs/circ{n}tr{tr}bn{bn}.pkl'.format(n=num_qubits, tr=train, bn=group), 'wb')
        pickle.dump(circ_set, output)
        output.close()
    else:
        pkl_file = open('./circs/circ{n}tr{tr}bn{bn}.pkl'.format(n=num_qubits, tr=train, bn=group), 'rb')
        circ_set = pickle.load(pkl_file)
        pkl_file.close()
        
        data = (list(reversed(backend_2.jobs(limit)))[group]).result()
        
        for circ, result_noisy in zip(circ_set, data):
            result_ideal = execute(circ, backend_1, basis_gates=basis_gates, shots=shots).result()
            items_ideal = list(result_ideal.get_counts().items())
            items_noisy = list(result_noisy.get_counts().items())
    
            for k in range(len(items_ideal)):
                for j in range(num_qubits):
                    if items_ideal[k][0][j] == '0': data_ideal[i,j,0] += items_ideal[k][1]
                    else: data_ideal[i,j,1] += items_ideal[k][1]
            for l in range(len(items_noisy)):
                for m in range(num_qubits):
                    if items_noisy[l][0][m] == '0': data_noisy[i,m,0] += items_noisy[l][1]
                    else: data_noisy[i,m,1] += items_noisy[l][1]

    data_ideal = data_ideal/shots
    data_noisy = data_noisy/shots
    return (data_ideal, data_noisy)


if __name__ == "__main__":
    size = 2
    shots = 2048
    num_qubits = 4
    depth = 3
    max_operands = 2
    prob_one =0.01
    prob_two = 0.01
    
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    data_ideal, data_noisy = generate_data(size, shots, num_qubits, depth, max_operands, basis_gates)
    print('1', data_ideal)
    print('2', data_noisy)

