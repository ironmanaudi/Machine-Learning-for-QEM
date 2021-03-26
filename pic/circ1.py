import qiskit

from qiskit.tools import visualization
from qiskit.tools.visualization import circuit_drawer
from qiskit import QuantumCircuit

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
circuit_drawer(qc, filename='circuit.png')
