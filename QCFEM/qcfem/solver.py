import torch
from qiskit import transpile
from qiskit_aer import Aer
from .problems import MaxCutProblem, MockProblem
from .circuits import create_basic_circuit, create_qaoa_circuit, create_quantum_annealing_circuit
from .utils import entropy_binary
import math

class QuantumSolver:
    def __init__(self, problem, num_trials=100, num_steps=100, tunneling_strength=0.1, seed=1, h_factor=0.01):
        self.num_trials = num_trials
        self.num_steps = num_steps
        self.tunneling_strength = tunneling_strength
        self.seed = seed
        self.h_factor = h_factor
        self.problem = problem
        self.problem.set_up_couplings_status('cuda', torch.float32)

    def measure_circuit(self, circuit):
        simulator = Aer.get_backend('qasm_simulator')
        circuit.measure_all()
        new_circuit = transpile(circuit, simulator)
        job = simulator.run(new_circuit, shots=self.num_trials)
        result = job.result()
        counts = result.get_counts()
        return counts

    def iterate(self):
        num_nodes = self.problem.num_nodes
        gamma = torch.randn(1)
        beta = torch.randn(1)
        initial_tunneling = self.tunneling_strength
        final_tunneling = 0.01

        for step in range(self.num_steps):
            t = step / (self.num_steps - 1)
            current_tunneling = initial_tunneling * (1 - t) + final_tunneling * t
            gamma += current_tunneling * torch.randn(1)
            beta += current_tunneling * torch.randn(1)
            if hasattr(self.problem, 'edges'):
                circuit = create_qaoa_circuit(num_nodes, self.problem.edges, gamma.item(), beta.item())
            else:
                circuit = create_basic_circuit(num_nodes)
            counts = self.measure_circuit(circuit)
            # 可扩展反馈机制
            max_cut = -1
            best_bits = None
            for bitstring, count in counts.items():
                bits = [int(x) for x in bitstring[::-1]]
                cut = 0
                if hasattr(self.problem, 'edges'):
                    for i, j in self.problem.edges:
                        if bits[i] != bits[j]:
                            cut += 1
                else:
                    cut = sum(bits)
                if cut > max_cut:
                    max_cut = cut
                    best_bits = bits
        return torch.tensor(best_bits, dtype=torch.float32)

    def solve(self):
        marginal = self.iterate()
        configs, results = self.problem.inference_value(marginal)
        return configs, results

class SimulatedAnnealingSolver:
    def __init__(self, problem, num_steps=1000, initial_temp=5.0, final_temp=0.01, seed=1):
        self.problem = problem
        self.num_steps = num_steps
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.seed = seed
        torch.manual_seed(seed)
        self.num_nodes = problem.num_nodes

    def energy(self, state):
        # 这里假设问题的 inference_value 返回 (configs, value)，value 越大越好
        # 退火时能量应为负的目标函数
        _, value = self.problem.inference_value(torch.tensor(state, dtype=torch.float32))
        return -value

    def solve(self):
        # 随机初始化状态
        state = torch.randint(0, 2, (self.num_nodes,)).tolist()
        best_state = list(state)
        best_energy = self.energy(state)
        for step in range(self.num_steps):
            t = step / (self.num_steps - 1)
            temp = self.initial_temp * (1 - t) + self.final_temp * t
            # 随机选择一个比特翻转
            i = torch.randint(0, self.num_nodes, (1,)).item()
            new_state = list(state)
            new_state[i] = 1 - new_state[i]
            e_old = self.energy(state)
            e_new = self.energy(new_state)
            # Metropolis准则
            if e_new < e_old or math.exp((e_old - e_new) / temp) > torch.rand(1).item():
                state = new_state
                if e_new < best_energy:
                    best_energy = e_new
                    best_state = list(new_state)
        configs, results = self.problem.inference_value(torch.tensor(best_state, dtype=torch.float32))
        return configs, results

class QuantumAnnealingSolver:
    def __init__(self, problem, num_steps=20, initial_tunneling=2.0, final_tunneling=0.01, seed=1):
        self.problem = problem
        self.num_steps = num_steps
        self.initial_tunneling = initial_tunneling
        self.final_tunneling = final_tunneling
        self.seed = seed
        torch.manual_seed(seed)
        self.num_nodes = problem.num_nodes

    def measure_circuit(self, circuit, num_shots=100):
        from qiskit_aer import Aer
        from qiskit import transpile
        simulator = Aer.get_backend('qasm_simulator')
        circuit.measure_all()
        new_circuit = transpile(circuit, simulator)
        job = simulator.run(new_circuit, shots=num_shots)
        result = job.result()
        counts = result.get_counts()
        return counts

    def solve(self):
        num_nodes = self.num_nodes
        best_bits = None
        max_value = float('-inf')
        for step in range(self.num_steps):
            t = step / (self.num_steps - 1)
            tunneling = self.initial_tunneling * (1 - t) + self.final_tunneling * t
            circuit = create_quantum_annealing_circuit(num_nodes, self.problem.edges, tunneling)
            counts = self.measure_circuit(circuit)
            for bitstring, count in counts.items():
                bits = [int(x) for x in bitstring[::-1]]
                _, value = self.problem.inference_value(torch.tensor(bits, dtype=torch.float32))
                if value > max_value:
                    max_value = value
                    best_bits = bits
        configs, results = self.problem.inference_value(torch.tensor(best_bits, dtype=torch.float32))
        return configs, results