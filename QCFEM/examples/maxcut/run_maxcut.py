import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from qcfem.problems import MaxCutProblem
from qcfem.solver import SimulatedAnnealingSolver, QuantumSolver, QuantumAnnealingSolver

if __name__ == "__main__":
    # 4节点环形图
    edges = [(0,1), (1,2), (2,3), (3,0)]
    problem = MaxCutProblem(num_nodes=4, edges=edges)
    print("---模拟退火---")
    solver1 = SimulatedAnnealingSolver(problem, num_steps=1000, initial_temp=5.0, final_temp=0.01)
    configs, results = solver1.solve()
    print("最优分组:", configs)
    print("最大割边数:", results)
    print("---QAOA---")
    solver2 = QuantumSolver(problem, num_trials=100, num_steps=50, tunneling_strength=0.1)
    configs, results = solver2.solve()
    print("最优分组:", configs)
    print("最大割边数:", results)
    print("---量子退火---")
    solver3 = QuantumAnnealingSolver(problem, num_steps=20, initial_tunneling=2.0, final_tunneling=0.01)
    configs, results = solver3.solve()
    print("最优分组:", configs)
    print("最大割边数:", results) 