# AI Agent 支持

本目录用于存放 OpenCode 的项目级配置与 AI Agent 使用说明，帮助开发者使用 Agent 辅助完成需求分析、代码修改、文档编写与验证。

## 目录结构

```text
.opencode/
└── opencode.jsonc    # OpenCode 项目配置
```

## 配置说明

`opencode.jsonc` 定义了本项目允许 Agent 使用的工具权限：

| 配置项 | 说明 |
| --- | --- |
| `bash` | 允许执行终端命令，用于构建、测试与 Git 操作 |
| `external_directory` | 允许访问工作区之外的目录 |
| `read` | 允许读取仓库文件 |
| `edit` | 允许修改仓库文件 |

如需收紧权限，可在个人环境中覆盖该配置，避免 Agent 执行未预期的命令或修改。

## 默认 Skills

下表为仓内默认约定、推荐随 Agent 启用的 skills，覆盖从问题接收、代码修改到提交合入的完整流程。仓内共享 skill 的目录约定见 [.claude/skills/README.md](../.claude/skills/README.md)，通用 skill 可来源于 [CANN Agent Skills](https://gitcode.com/cann-agent/skills)。

| Skill | 用途 | 典型指令 |
| --- | --- | --- |
| `gitcode-issue` | 读取、创建与回复 Issue | `读取 issue 12 并给出修复建议` |
| `gitcode-pr` | 创建 PR 并按模板填写描述 | `为当前改动创建 PR` |
| `gitcode-pipeline` | 触发并跟踪 CI 流水线，定位失败项 | `触发 CI 并跟踪结果` |
| `code-review` | 按仓库规范检视 diff，检查格式、接口与测试 | `检视当前分支的改动` |
| `docs-writer` | 编写或更新 README、docs 等文档 | `补充构建章节的常见问题` |
| `build-verify` | 执行 `build.sh` 并检查 PJRT 插件等产物 | `构建并校验产物是否生成` |
| `example-runner` | 运行 `examples` 下的算子与模型样例 | `运行 ops 样例并对比 CPU/NPU 结果` |

Skills 使用约定：

- 优先使用仓内已有的 skill 定义；本地扩展的 skill 建议放在 `.claude/skills/` 或 `.opencode/skills/`，并按需评估是否提交到仓库。
- 执行构建、样例或 CI 相关 skill 前，先确认 CANN 环境、`SOC_VERSION` 与 NPU 设备可用。
- skill 的输出仅作为辅助结果，合入前仍需人工确认代码、文档与验证结论。

## 推荐工作流

1. **需求澄清**：说明问题背景、目标、影响范围与约束，避免只给出模糊指令。
2. **方案设计**：让 Agent 先阅读相关目录与文档，输出改动计划，确认后再修改代码。
3. **代码实现**：按 [贡献指南](../CONTRIBUTING.md) 的规范实施改动，保持改动最小化。
4. **验证**：执行构建或样例命令，确认结果符合预期。
5. **文档同步**：用户可见改动同步更新 `README.md`、`docs/` 或 `examples/README.md`。

## 常用指令示例

```text
说明 xla_npu/service/transform 下各 pass 的作用，并给出执行顺序。
分析 examples/ops 中某个算子的报错原因，给出修复方案。
参照 docs/zh/build.md 的结构，补充某一章节。
新增一个 HLO pass，需要修改哪些文件？
```

## 上下文约定

与 Agent 协作时，建议明确以下信息：

- **背景**：要解决什么问题，当前表现是什么。
- **目标**：期望达到的效果与验收标准。
- **范围**：涉及哪些目录、文件或接口。
- **约束**：是否允许修改对外接口、构建流程或性能敏感代码。
- **验证**：如何判断改动完成，例如构建命令、样例或测试。

## 关联说明

- Claude Code 的项目级说明见 [.claude/README.md](../.claude/README.md)。
- 开发规范与提交要求见 [CONTRIBUTING.md](../CONTRIBUTING.md)。
- 文档索引见 [docs/README.md](../docs/README.md)。
