import sys
import os
import json
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from qcfem.problems import IndependentSetProblem
from qcfem.solver import SimulatedAnnealingSolver, QuantumSolver, QuantumAnnealingSolver

def main():
    parser = argparse.ArgumentParser(description='批量求解最大独立集问题')
    parser.add_argument('--method', type=str, default='anneal', choices=['anneal', 'qaoa', 'qanneal'],
                        help='选择求解方法: anneal(退火) 或 qaoa(量子近似优化) 或 qanneal(量子退火)')
    parser.add_argument('--json', type=str, default='independent_set_list.json',
                        help='包含图结构和标准答案的json文件')
    args = parser.parse_args()

    with open(args.json, 'r', encoding='utf-8') as f:
        graphs = json.load(f)

    for graph in graphs:
        print(f"图名: {graph['name']}")
        problem = IndependentSetProblem(num_nodes=graph['num_nodes'], edges=graph['edges'])
        if args.method == 'anneal':
            solver = SimulatedAnnealingSolver(problem, num_steps=1000, initial_temp=5.0, final_temp=0.01)
        elif args.method == 'qaoa':
            solver = QuantumSolver(problem, num_trials=100, num_steps=50, tunneling_strength=0.1)
        else:
            solver = QuantumAnnealingSolver(problem, num_steps=20, initial_tunneling=2.0, final_tunneling=0.01)
        configs, results = solver.solve()
        print("  计算分组:", configs)
        print("  计算最大独立集大小:", results)
        print("  标准最大独立集大小:", graph['max_independent_set'])
        if results == graph['max_independent_set']:
            print("  结果正确！")
        else:
            print("  结果错误！")
        print("-" * 40)

if __name__ == "__main__":
    main() 