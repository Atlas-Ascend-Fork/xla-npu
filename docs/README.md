# 文档总览

简体中文 | [English](./README_en.md)

本目录汇总 XLA-NPU 的用户文档、设计文档与开发规范。如需了解项目整体定位，请先阅读根目录 [README.md](../README.md)。

## 目录结构

```text
docs/
├── README.md                        # 文档总览（中文）
├── README_en.md                     # 文档总览（英文）
└── zh/
    ├── README.md                    # 特性与支持清单、使用约束、常见问题
    ├── build.md                     # 构建与验证指南
    ├── precommit_guide.md           # pre-commit 与开源合规检查
    ├── overview.png                 # 架构示意图
    └── design/
        └── architecture.md          # 架构设计说明
```

## 用户文档

- [特性与支持清单](./zh/README.md)：整体约束、支持的 JAX API 清单与常见问题。
- [构建与验证](./zh/build.md)：依赖准备、一键构建、运行期环境变量与样例验证。
- [安装与卸载指南](../INSTALL_GUIDE.md)：完整安装、卸载流程与故障处理。
- [样例运行验证](../examples/README.md)：`ops`、`mmoe`、`qwen` 样例的运行方式与结果对比。

## 设计文档

- [架构说明](./zh/design/architecture.md)：核心组件、编译执行流程、后端选择与扩展机制。
- [架构示意图](./zh/overview.png)：XLA-NPU 与 JAX、XLA、CANN 的层次关系。

## 开发规范

- [贡献指南](../CONTRIBUTING.md)：Issue / Pull Request 流程、代码规范、测试与文档要求。
- [提交规范](./zh/precommit_guide.md)：clang-format、pre-commit 与 OAT 开源合规检查。
- [AI Agent 支持](../.opencode/README.md)：OpenCode / Claude Code 的仓内上下文与辅助开发流程。

## 相关信息

- [安全声明](../SECURITY.md)：漏洞报告渠道与处理流程。
- [许可证](../LICENSE)：Apache License 2.0。
- [开源合规配置](../OAT.xml)：开源合规检查配置。
