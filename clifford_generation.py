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
from operator import itemgetter
# display(plot_histogram(result_ideal.get_counts()))


def generate_data(size, shots, num_qubits, basis_gates, device, group=-1, limit=0, train=1):
    circ_set = []
    val_ideals = []
    val_noisys = []
    idxs = []
    sizes = []
    
    backend_1 = Aer.get_backend('qasm_simulator')
    #IBMQ.save_account('83a95fc7efba05f250ed50ff6bdf1638541ebd4f9e46396f26c8dc12f15b2331ea24d9db4f59c30b6e397c2268e40ddc1dae4772091baa73d7e99c91e8d8c56b')
    provider = IBMQ.load_account()
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
        limit = 66
        pkl_file = open('./circs/circ{n}tr{tr}bn{bn}.pkl'.format(n=num_qubits, tr=train, bn=group), 'rb')
        circ_set = pickle.load(pkl_file)
        pkl_file.close()
        
        data = ((list(reversed(backend_2.jobs(limit)))[group]).result()).get_counts()
        data = list(data) #make sure it is a list of dicts

        for circ, result_noisy, i in zip(circ_set, data, range(size)):
            # print(result_noisy)
            result_ideal = execute(circ, backend_1, basis_gates=basis_gates, shots=shots).result()
            result_ideal = result_ideal.get_counts()
            
            #update dict of ideal and noisy
            temp_ideal = dict.fromkeys(result_ideal, 0)
            temp_noisy = dict.fromkeys(result_noisy, 0)
            result_ideal.update(temp_noisy)
            result_noisy.update(temp_ideal)
            idx = sorted(result_ideal)
            val_ideal = list(itemgetter(*idx)(result_ideal))
            val_noisy = list(itemgetter(*idx)(result_noisy))
            
            #append noisy prob for each index
            for k in range(len(idx)):
                idx[k] = list(idx[k])
                idx[k].append(val_noisy[k]/shots)
                
            sizes.append(len(idx))
                
            #convert idx from str to float
            idx = np.array(idx)
            idx = idx.astype(np.float64)
            idx = torch.from_numpy(idx)
            
            #list of tensors
            val_ideals.append(torch.Tensor(val_ideal, dtype=torch.float64)/shots)
            val_noisys.append(torch.Tensor(val_noisy, dtype=torch.float64)/shots)
            idxs.append(idx)
        
        #pad noisy and ideal
        idxs = torch.nn.utils.rnn.pad_sequence(idxs, batch_first=True)
        val_ideals = torch.nn.utils.rnn.pad_sequence(val_ideals, batch_first=True)
        val_noisys = torch.nn.utils.rnn.pad_sequence(val_noisys, batch_first=True)
        
        #pack padded sequences
        pack_idxs = torch.nn.utils.rnn.pack_padded_sequence(idxs, lengths=sizes, batch_first=True, enforce_sorted=False)
        pack_ideals = torch.nn.utils.rnn.pack_padded_sequence(val_ideals, lengths=sizes, batch_first=True, enforce_sorted=False)
        pack_noisys = torch.nn.utils.rnn.pack_padded_sequence(val_noisys, lengths=sizes, batch_first=True, enforce_sorted=False)
        
        return (val_ideals, pack_idxs, sizes)


if __name__ == "__main__":
    size = 2
    shots = 2048
    num_qubits = 4
    depth = 3
    max_operands = 2
    prob_one =0.01
    prob_two = 0.01
    
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    val_ideals, pack_idxs, sizes = generate_data(size, shots, num_qubits, depth, max_operands, basis_gates)
    print('1', data_ideal)
    print('2', data_noisy)

