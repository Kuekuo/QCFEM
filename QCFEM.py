import numpy as np
import torch
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RYGate, RZGate
from qiskit.opflow import I, X, Y, Z
from qiskit.opflow.state_fns import StateFn
from qiskit.utils import QuantumInstance

# 定义问题规模
num_qubits = 5
num_layers = 3

# 定义最大割问题的成本哈密顿量
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
cost_hamiltonian = sum((1 - Z ^ Z).eval([i, j]) for i, j in edges)

# 初始化量子线路
def build_circuit(params, beta):
    qc = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        qc.h(qubit)
    for layer in range(num_layers):
        # 成本层
        for i, j in edges:
            qc.rzz(2 * beta * params[layer, 2 * i], i, j)
        # 混合层
        for qubit in range(num_qubits):
            qc.rx(2 * params[layer, 2 * qubit + 1], qubit)
    return qc

# 计算期望值
def compute_expectation(params, beta):
    qc = build_circuit(params, beta)
    qc.measure_all()
    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=1024).result()
    counts = result.get_counts(qc)
    expectation_value = 0
    for bitstring, count in counts.items():
        cut_value = sum(1 for (i, j) in edges if bitstring[i] != bitstring[j])
        expectation_value += count * cut_value
    expectation_value /= 1024
    return expectation_value

# 变分优化
def compute_gradient(params, beta):
    gradient = np.zeros_like(params)
    for layer in range(num_layers):
        for param_index in range(2 * num_qubits):
            # 计算正向和反向的期望值
            params[layer, param_index] += np.pi / 2
            pos_expectation = compute_expectation(params, beta)
            params[layer, param_index] -= np.pi
            neg_expectation = compute_expectation(params, beta)
            params[layer, param_index] += np.pi / 2
            gradient[layer, param_index] = (pos_expectation - neg_expectation) / 2
    return gradient

# 使用策略梯度方法更新参数
learning_rate = 0.1
betas = np.arange(0.3, 1.5 + 0.3, 0.3)  # 从 0.3 到 1.5，间隔 0.3
params = np.random.rand(num_layers, 2 * num_qubits)

for iteration, beta in enumerate(betas):
    gradient = compute_gradient(params, beta)
    params -= learning_rate * gradient
    expectation_value = compute_expectation(params, beta)
    print(f"迭代 {iteration}: 逆温度 β: {beta}, 期望值: {expectation_value}")

# 最终结果
final_qc = build_circuit(params, betas[-1])
final_qc.measure_all()
backend = Aer.get_backend('qasm_simulator')
result = execute(final_qc, backend, shots=1024).result()
counts = result.get_counts(final_qc)

print("量子线路的输出分布:")
print(counts)