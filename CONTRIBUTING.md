# 贡献指南

欢迎参与 XLA-NPU 的开发与贡献。在提交 Issue 或 Pull Request 之前，请先阅读本指南以及 [CANN 社区贡献流程](https://gitcode.com/cann/community)，了解行为准则、CLA 协议签署与开源贡献流程。

## 1. 提交 Issue

请根据问题类型选择合适的 [Issue 模板](.gitcode/ISSUE_TEMPLATE)：

| 模板 | 适用场景 |
| --- | --- |
| `bug-report.yml` | 缺陷反馈：构建失败、运行异常、精度问题等 |
| `feature-request.yml` | 新功能或能力增强建议 |
| `documentation.yml` | 文档错误、缺失或表述不清 |
| `question.yml` | 使用咨询与答疑 |
| `request-for-comments.yml` | 设计方案或较大改动的意见征集 |

Issue 中请尽量提供复现步骤、环境信息（CANN 版本、JAX 版本、芯片型号）、日志与期望结果。

## 2. 贡献流程

1. Fork 代码仓并创建个人分支。
2. 完成代码或文档修改，并在本地验证。
3. 执行 `pre-commit run` 确保格式与合规检查通过。
4. 提交 Pull Request，并按 [PR 模板](.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md) 填写背景、方案与验证结果。
5. 等待 CI 流水线与代码检视通过。
6. Committer 检视通过并合入。

提交 PR 时请关注以下事项：

1. PR 描述需要说明业务背景、改动目的与验证方式。
2. 提交前使用 pre-commit 完成代码格式化与 OAT 合规检查，参见 [pre-commit 使用指导](docs/zh/precommit_guide.md)。
3. 涉及新增能力、修改对外接口、修改编译流程或较大重构的改动，建议先通过 Issue 讨论方案，避免无效开发。
4. 遵循项目现有代码风格：C/C++ 使用 `clang-format`，Python 使用现有文件风格，Shell 脚本遵循 `shellcheck` 常见约束。
5. 一个 PR 尽量只做一件事，避免混合无关改动；提交历史保持清晰可读。

## 3. 提交前自检清单

### 3.1 代码格式

```bash
pip install pre-commit
pre-commit run
```

`pre-commit` 会执行 `clang-format` 与 OAT 开源合规检查。

### 3.2 许可证头

新增源文件需包含与仓库一致的 Apache License 2.0 许可证头，格式参考同目录已有文件。开源合规配置见 [OAT.xml](OAT.xml)。

### 3.3 Shell 脚本

- 使用 `set -e`，避免错误被静默忽略。
- `cd` 后检查返回值：`cd path || { echo "error"; exit 1; }`。
- 单个函数不宜过长，保持职责单一。
- 循环处理文件时，失败场景需明确输出 `FAILED` 等信息。

### 3.4 文档

- 用户可见的改动需同步更新相关文档（`README.md`、`README_en.md`、`docs/`、`examples/README.md` 等）。
- 文档中的命令、路径、环境变量必须与代码保持一致。
- 新增文档请在 [docs/README.md](docs/README.md) 中登记。

## 4. 提交规范

Commit message 采用 `<type>: <subject>` 格式：

| 类型 | 说明 | 示例 |
| --- | --- | --- |
| `feat` | 新功能 | `feat: add gather lowering for afir backend` |
| `fix` | 修复缺陷 | `fix: correct aclnn permute result handoff` |
| `docs` | 文档修改 | `docs: restructure README and add build guide` |
| `style` | 格式调整，不影响逻辑 | `style: apply clang-format` |
| `refactor` | 重构，不新增功能或修复缺陷 | `refactor: split fusion pipeline` |
| `perf` | 性能优化 | `perf: reuse compiled fusion artifacts` |
| `test` | 测试相关 | `test: expand ops coverage` |
| `chore` | 构建、依赖或工具调整 | `chore: update pre-commit hooks` |
| `ci` | CI 配置调整 | `ci: add OAT check stage` |

合并前建议通过 `git rebase` 压缩无效 commit，保持提交历史整洁。

## 5. CI 检查项

| 检查项 | 说明 | 本地验证方式 |
| --- | --- | --- |
| `clang-format` | C/C++ 代码格式 | `clang-format -i <file>` |
| `OAT` | 开源合规：二进制文件、许可证头 | `pre-commit run oat-check` |
| 构建 | PJRT 插件可正常编译 | `./build/build.sh install` |
| 样例 | 算子与模型样例运行通过 | 参见 [examples/README.md](examples/README.md) |

## 6. 目录职责

| 目录 | 职责 |
| --- | --- |
| `xla_npu/pjrt/` | PJRT 插件、客户端、设备描述与 GE/AIR 路径 |
| `xla_npu/service/` | 编译入口、HLO 优化、IR 发射、buffer 与执行流分配 |
| `xla_npu/service/transform/` | NPU 专属 HLO pass |
| `xla_npu/service/convert_to_afir/` | AFIR/AscIR 自动融合后端 |
| `xla_npu/backend/` | 融合算子发射与运行期 thunk |
| `xla_npu/stream_executor/` | NPU StreamExecutor 与自动融合 kernel 加载 |
| `xla_npu/dependency/` | CANN 接口封装、头文件与外部依赖 |
| `examples/` | 算子与模型样例 |
| `docs/` | 用户文档、设计文档与开发规范 |

## 7. 贡献分级

| 类型 | 范围 | 建议流程 |
| --- | --- | --- |
| 一般贡献 | Bug 修复、文档修正、小范围格式调整 | 直接提交 PR，CI 与检视通过后合入 |
| 功能贡献 | 新增算子支持、后端能力、性能优化 | 先提交 Issue 讨论方案，再提交 PR |
| 架构贡献 | 模块边界、编译流程、对外接口调整 | 先通过 Issue 或 SIG 讨论形成设计文档，再提交 PR |

架构类改动建议在 PR 中附上设计说明，覆盖改动背景、方案对比、影响范围与验证结果。

## 8. 测试要求

- **Bug 修复**：需说明复现方式与修复后的验证结果。
- **算子或融合改动**：运行 `examples/ops`，确认相关算子结果与 CPU 一致。
- **模型相关改动**：运行 `examples/mmoe` 或 `examples/qwen`，确认端到端结果与性能无明显回退。
- **构建相关改动**：至少验证一种 `--npu-auto-fuse-backend` 取值下的完整构建与样例运行。

感谢您的贡献！
