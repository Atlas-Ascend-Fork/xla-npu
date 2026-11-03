# Claude Code 项目说明

本目录用于存放 Claude Code 的项目级约定与辅助资源，使 Agent 在理解仓库结构、编码规范与验证方式的基础上参与开发。

## 目录结构

```text
.claude/
├── README.md        # 项目级使用说明
└── skills/
    └── README.md    # 仓内默认 skills 说明
```

仓内共享 skills 的目录约定与列表见 [skills/README.md](skills/README.md)。如需在本地扩展 Agent 能力，可在本目录下自行添加 `settings.json` 等 Claude Code 支持的内容；提交到仓库前请确认不包含个人路径、令牌或其他敏感信息。

## 项目上下文

- 项目定位：OpenXLA/XLA 的昇腾 NPU 后端，通过 PJRT 插件对接 JAX。
- 核心实现：`xla_npu/`，包括 PJRT、Service、HLO pass、融合后端、Thunk 与 StreamExecutor。
- 构建方式：`build/build.sh`，详细说明见 [构建与验证](../docs/zh/build.md)。
- 架构设计：见 [架构说明](../docs/zh/design/architecture.md)。
- 样例：`examples/ops`、`examples/mmoe`、`examples/qwen`。

## 默认 Skills

仓内默认约定、推荐随 Agent 启用的 skills 包括：`gitcode-issue`、`gitcode-pr`、`gitcode-pipeline`、`code-review`、`docs-writer`、`build-verify`、`example-runner`。各 skill 的用途、典型指令与使用约定见 [AI Agent 支持](../.opencode/README.md#默认-skills)。

本地扩展的 skill 建议放在 `.claude/skills/` 下，并在提交前确认不包含个人路径或敏感信息。

## 协作约定

1. 修改前先阅读相关目录与现有实现，遵循既有代码风格。
2. 仅修改与需求直接相关的代码，避免顺手重构或格式化无关文件。
3. C/C++ 改动需通过 `clang-format`，提交前执行 `pre-commit run`。
4. 新增源文件需包含 Apache License 2.0 许可证头。
5. 用户可见改动同步更新文档，并在 [docs/README.md](../docs/README.md) 中登记新增文档。
6. 提交信息遵循 `<type>: <subject>` 格式，类型定义见 [CONTRIBUTING.md](../CONTRIBUTING.md)。

## 验证方式

```bash
# 构建
cd build
./build.sh install

# 代码格式与合规检查
pre-commit run

# 算子样例
cd ../examples/ops
python main.py
```

## 关联说明

- OpenCode 配置与 Agent 使用说明见 [.opencode/README.md](../.opencode/README.md)。
- 贡献流程与测试要求见 [CONTRIBUTING.md](../CONTRIBUTING.md)。
