# Documentation

English | [简体中文](./README.md)

This directory contains the user guides, design documents and development conventions for XLA-NPU. For a project overview, see the root [README_en.md](../README_en.md).

## Directory Layout

```text
docs/
├── README.md                        # Documentation index (Chinese)
├── README_en.md                     # Documentation index (English)
└── zh/
    ├── README.md                    # Feature/support list, constraints and FAQ
    ├── build.md                     # Build and verification guide
    ├── precommit_guide.md           # pre-commit and open-source compliance checks
    ├── overview.png                 # Architecture figure
    └── design/
        └── architecture.md          # Architecture design
```

## User Documentation

- [Features and supported JAX APIs](./zh/README.md) (Chinese): constraints, supported JAX APIs and FAQ.
- [Build and verification](./zh/build.md) (Chinese): dependencies, one-click build, runtime environment variables and example validation.
- [Install and uninstall guide](../INSTALL_GUIDE.md) (Chinese): full installation/uninstallation flow and troubleshooting.
- [Examples](../examples/README.md) (Chinese): run `ops`, `mmoe` and `qwen`, and compare CPU/NPU results.

## Design Documentation

- [Architecture](./zh/design/architecture.md) (Chinese): core components, compilation/execution flow, backend selection and extension mechanisms.
- [Architecture figure](./zh/overview.png): relationship between XLA-NPU, JAX, XLA and CANN.

## Development Conventions

- [Contributing guide](../CONTRIBUTING.md): issue/pull request flow, code style, testing and documentation requirements.
- [Commit conventions](./zh/precommit_guide.md) (Chinese): clang-format, pre-commit and OAT compliance checks.
- [AI Agent support](../.opencode/README.md) (Chinese): in-repository context and assisted development workflow for OpenCode / Claude Code.

## Related Information

- [Security statement](../SECURITY.md): vulnerability reporting channels and process.
- [License](../LICENSE): Apache License 2.0.
- [Open-source compliance configuration](../OAT.xml): open-source compliance check configuration.
