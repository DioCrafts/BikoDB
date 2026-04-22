# Installation

## What “installation” means today

BikoDB currently has two realistic entry paths:

1. **Build the Rust workspace and use it as an embedded library**.
2. **Wire the experimental HTTP router into your own `axum` application**.

The repository does **not** currently ship a polished standalone server binary, Docker image, or one-command deployment path.

## Requirements

- Rust **1.93+**
- A recent C toolchain suitable for building Rust dependencies
- Linux, macOS, or Windows with standard Rust support

## Build from source

```bash
git clone https://github.com/unnamedlab/BikoDB.git
cd BikoDB
cargo test --workspace
cargo build --workspace
```

Why start with `cargo test --workspace`?

- it validates that the workspace compiles on your machine,
- it exercises the crates the docs point at,
- it is currently the closest thing to a full-project health check.

## Use BikoDB as an embedded dependency

The cleanest current adoption path is to depend on the crates you need.

For a graph-centric embedded integration, the most relevant crates are:

- `bikodb-core`
- `bikodb-graph`
- `bikodb-query`
- `bikodb-server` (for the in-process `Database` facade)

## HTTP usage today

An HTTP API router exists in `bikodb-server`, but the repo currently exposes it as a library component rather than a finished packaged service.

That means:

- you can build an `axum` app around `bikodb_server::build_router`,
- the API surface is testable,
- but you should treat it as **experimental integration surface**, not a documented production server distribution.

## Not included yet

These items are intentionally **not** presented as ready-to-use installation targets:

- official Docker image,
- Docker Compose deployment,
- Kubernetes packaging,
- packaged cluster deployment,
- hosted docs describing production operations end-to-end.

## Next steps

- [Quickstart →](/guide/quickstart)
- [Supported features & limitations →](/guide/supported-features)
- [Transactions & concurrency →](/architecture/transactions)
