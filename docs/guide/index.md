# What is BikoDB?

BikoDB is a Rust codebase that combines a concurrent graph engine, a document store, and vector search primitives behind one workspace.

The current project is most credible as an **embedded graph runtime with multi-model building blocks**. It is not yet a finished packaged database distribution with a polished operator experience.

## Who this is for

BikoDB is currently a better fit for people who want to:

- explore or extend a Rust graph/database engine,
- embed graph capabilities into another Rust application,
- evaluate a graph + document + vector architecture at code level,
- test query/parser, algorithm, and storage ideas inside one workspace.

It is a weaker fit today for teams looking for:

- a drop-in server binary or Docker image,
- a production-ready cluster story,
- broad SQL/openCypher/TinkerPop compatibility,
- a complete operational surface with timeouts, profiling, and slow-query tooling.

## Maturity snapshot

| Area | Status | What that means today |
| --- | --- | --- |
| Core graph data structures and algorithms | Stable | This is the strongest, most exercised part of the repo. |
| Embedded `Database` API | Stable | Good for in-process Rust usage and tests. |
| SQL/Cypher/Gremlin read subset | Stable | Useful documented subset, not full language coverage. |
| Document store | Stable | CRUD and filtered reads are implemented in-process. |
| HNSW vector search | Experimental | Works in code, but broader packaging and operational docs are still thin. |
| HTTP router | Experimental | An `axum` router exists, but the repo does not ship a standalone daemon. |
| Incremental embeddings / inference | Experimental | Real code exists, but parts of the AI stack are still clearly evolving. |
| Cluster / HA | Experimental | Present in the workspace, but not yet documented as a hardened deployment story. |
| Python / Node bindings | Experimental | Bindings exist, but they are not yet the main documented entry path. |
| Packaged server / Docker-first UX | Planned | Not yet the primary way to start BikoDB. |

## Project shape

BikoDB is organised as a Rust workspace with crates for core types, storage, graph execution, query parsing, AI/vector features, clustering, bindings, and server integration.

The most important thing for external users is that **not every crate is equally mature**. These docs focus first on the parts that can be demonstrated and explained precisely.

## Start here

- [Installation →](/guide/installation)
- [Quickstart →](/guide/quickstart)
- [Supported features & limitations →](/guide/supported-features)
- [Transactions & concurrency →](/architecture/transactions)
- [Supported guarantees →](/architecture/supported-guarantees)
- [Operations & observability →](/architecture/operations-observability)
- [Benchmark methodology →](/benchmarks/methodology)
