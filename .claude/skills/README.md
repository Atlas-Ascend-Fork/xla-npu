# 仓内 Skill 约定

本目录用于存放 XLA-NPU 仓库共享的 Agent skills 说明，供 Claude Code 与 OpenCode 等工具参考。每个 skill 建议使用独立子目录，入口文件为 `SKILL.md`。

## 目录约定

```text
.claude/skills/
├── README.md            # 本说明
└── <skill-name>/
    ├── SKILL.md         # skill 定义：适用场景、步骤、命令与约束
    ├── references/      # 可选：参考资料
    └── scripts/         # 可选：辅助脚本
```

- 仓库共享 skill 放在本目录并随代码提交。
- 个人临时 skill 建议放在本地环境，不提交到仓库。
- 通用 skill 可从 [CANN Agent Skills](https://gitcode.com/cann-agent/skills) 获取。

## 推荐 Skill 列表

| Skill | 用途 | 典型指令 |
| --- | --- | --- |
| `gitcode-issue` | 读取、创建与回复 GitCode Issue | `读取 issue 12 并给出修复建议` |
| `gitcode-pr` | 创建 Pull Request 并按模板填写描述 | `为当前改动创建 PR` |
| `gitcode-pipeline` | 触发并跟踪 CI 流水线，定位失败项 | `触发 CI 并跟踪结果` |
| `code-review` | 按仓库规范检视 diff，检查格式、接口与测试 | `检视当前分支的改动` |
| `docs-writer` | 编写或更新 README、docs 等文档 | `补充构建章节的常见问题` |
| `build-verify` | 执行 `build.sh` 并检查 PJRT 插件等产物 | `构建并校验产物是否生成` |
| `example-runner` | 运行 `examples` 下的算子与模型样例 | `运行 ops 样例并对比 CPU/NPU 结果` |

## 使用约定

1. 执行构建、样例或 CI 相关 skill 前，先确认 CANN 环境、`SOC_VERSION` 与 NPU 设备可用。
2. skill 输出仅作为辅助结果，合入前仍需人工确认代码、文档与验证结论。
3. 新增 skill 时，在 `SKILL.md` 中写明适用场景、前置条件、执行步骤与失败处理方式。
4. 提交前确认 skill 中不包含个人路径、令牌或其他敏感信息。

## 关联说明

- AI Agent 使用流程见 [.opencode/README.md](../../.opencode/README.md)。
- 开发规范见 [CONTRIBUTING.md](../../CONTRIBUTING.md)。
