# 样例运行验证

开发者完成 XLA-NPU 安装后,可通过本目录下的样例验证 XLA-NPU 功能。本代码仓提供基础算子测试和模型推理样例,具体如下。

## 样例列表

| 目录名称 | 功能描述 |
| -------- | -------- |
| [ops](./ops) | JAX 基础算子测试样例,支持 CPU 与 NPU 执行结果对比,包含 Add、Tanh、Multiply、Dot、Divide 等多种算子 |
| [mmoe](./mmoe) | MMOE (Multi-gate Mixture-of-Experts) 模型推理样例,展示多任务学习模型在 NPU 上的推理性能 |
| [common](./common) | 公共工具模块,包含基础运行器、配置管理和结果对比工具 |
| [output](./output) | 样例运行输出目录,存放 CPU 和 NPU 的执行结果文件 |

## 目录结构说明

```
examples/
├── common/          # 公共工具模块
│   ├── base_runner.py   # 基础运行器类
│   ├── config.py        # 公共配置
│   └── utils.py         # 工具函数
├── ops/             # 基础算子测试样例
│   ├── main.py          # 主入口
│   ├── config.py        # 算子配置
│   ├── run_cpu.py       # CPU 运行脚本
│   ├── run_npu.py       # NPU 运行脚本
│   └── utils.py         # 工具函数
├── mmoe/            # MMOE 模型样例
│   ├── main.py          # 主入口
│   ├── model.py         # 模型定义
│   ├── config.py        # 模型配置
│   ├── init_params.py   # 参数初始化
│   ├── run_cpu.py       # CPU 运行脚本
│   ├── run_npu.py       # NPU 运行脚本
│   ├── utils.py         # 工具函数
│   └── show_result.py   # 结果展示
└── output/          # 输出目录
    ├── ops/             # ops 样例输出
    └── mmoe/            # mmoe 样例输出
```

## 快速开始

### 环境准备

1. 确保已完成 XLA-NPU 的安装, 参考项目根目录 [INSTALL_GUIDE.md](../INSTALL_GUIDE.md)
2. 设置运行所需环境变量, 参考项目根目录 [README.md](../README.md)

### 运行算子测试样例

```bash
cd ops
python main.py
```

可选参数:
- `--num-runs N`: 设置运行次数,默认为 1
- `--skip-cpu`: 跳过 CPU 执行,仅运行 NPU
- `--skip-npu`: 跳过 NPU 执行,仅运行 CPU
- `--cleanup`: 测试完成后清理临时文件

### 运行 MMOE 模型样例

```bash
cd mmoe

# 首次运行时, 如果examples/mmoe/目录下没有model_params.npz文件, 则需要初始化模型参数
python main.py --init

# 后续运行
python main.py
```

可选参数:
- `--num-runs N`: 设置运行次数,默认为 1
- `--skip-cpu`: 跳过 CPU 执行,仅运行 NPU
- `--skip-npu`: 跳过 NPU 执行,仅运行 CPU
- `--init`: 初始化模型参数
- `--cleanup`: 测试完成后清理临时文件

## 结果对比

样例运行后会自动对比 CPU 和 NPU 的执行结果,输出:
- 执行时间统计
- 结果差异分析(最大绝对误差、最大相对误差等)
- 测试结果状态(通过/失败)

## 注意事项

1. 运行 NPU 样例前请确保 NPU 设备可用
2. 模型执行的输入和输出数据默认保存在 `examples/output/` 目录下, 可通过--cleanup清理
