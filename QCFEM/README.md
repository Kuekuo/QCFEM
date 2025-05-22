# QCFEM

QCFEM 是一个基于 Qiskit 和 PyTorch 的量子组合优化与变分量子算法实验平台，支持如最大割（MaxCut）等问题的量子/混合求解。

## 目录结构
```
QCFEM/
├── qcfem/                # 主包目录
│   ├── __init__.py
│   ├── solver.py         # 量子求解器主类
│   ├── problems.py       # 问题定义（如MaxCut、MockProblem等）
│   ├── circuits.py       # 量子线路构建相关
│   └── utils.py          # 工具函数（如熵、概率等）
├── examples/             # 示例脚本
│   ├── maxcut/           # 最大割问题相关示例
│   │   ├── run_maxcut.py
│   │   ├── batch_run_matxcut.py
│   │   └── edges_list.json
│   └── independent_set/  # 独立集问题相关示例
│       ├── run_independent_set.py
│       ├── batch_run_independent_set.py
│       └── independent_set_list.json
├── README.md
├── requirements.txt
└── setup.py              # 可选，便于安装
```

## 安装
建议使用 Python 3.8+，并提前安装好 Qiskit 相关依赖。
```bash
pip install -r requirements.txt
```

## 快速开始
以最大割问题为例：
```bash
python examples/maxcut/run_maxcut.py
```

以独立集问题为例：
```bash
python examples/independent_set/run_independent_set.py
```

## 主要功能
- 量子退火、QAOA等量子优化算法
- 支持自定义问题建模
- 结果可视化与统计
- 易于扩展的模块化设计

## 依赖
- qiskit
- qiskit-aer
- torch
- numpy

## 引用
如果本项目对您的研究有帮助，请引用 Qiskit 和 PyTorch 官方文档。

## 许可证
MIT License