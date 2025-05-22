from qiskit import QuantumCircuit

def create_basic_circuit(num_qubits):
    qc = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        qc.h(qubit)
    return qc

def create_qaoa_circuit(num_qubits, edges, gamma, beta):
    qc = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        qc.h(qubit)
    for (i, j) in edges:
        qc.cx(i, j)
        qc.rz(2 * gamma, j)
        qc.cx(i, j)
    for qubit in range(num_qubits):
        qc.rx(2 * beta, qubit)
    return qc

def create_quantum_annealing_circuit(num_qubits, edges, tunneling, problem_strength=1.0):
    qc = QuantumCircuit(num_qubits)
    # 初始态：全体Hadamard，形成均匀叠加
    for i in range(num_qubits):
        qc.h(i)
    # 量子退火主循环：模拟哈密顿量演化
    # 1. 施加横场（X方向）
    for i in range(num_qubits):
        qc.rx(2 * tunneling, i)
    # 2. 施加问题哈密顿量（Z-Z耦合）
    for (i, j) in edges:
        qc.rzz(2 * problem_strength, i, j)
    return qc