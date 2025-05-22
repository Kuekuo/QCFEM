import torch

class MockProblem:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def set_up_couplings_status(self, device, dtype):
        pass

    def inference_value(self, marginal):
        configs = (marginal > 0).int().tolist()
        results = sum(configs)
        return configs, results

class MaxCutProblem:
    def __init__(self, num_nodes, edges):
        self.num_nodes = num_nodes
        self.edges = edges

    def set_up_couplings_status(self, device, dtype):
        pass

    def inference_value(self, marginal):
        configs = (marginal > 0).int().tolist()
        cut = 0
        for i, j in self.edges:
            if configs[i] != configs[j]:
                cut += 1
        return configs, cut 

class IndependentSetProblem:
    def __init__(self, num_nodes, edges):
        self.num_nodes = num_nodes
        self.edges = edges

    def set_up_couplings_status(self, device, dtype):
        pass

    def inference_value(self, marginal):
        configs = (marginal > 0).int().tolist()
        # 检查是否为独立集
        is_independent = True
        for i, j in self.edges:
            if configs[i] == 1 and configs[j] == 1:
                is_independent = False
                break
        size = sum(configs)
        # 如果不是独立集，返回-1作为惩罚
        return configs, size if is_independent else -1 