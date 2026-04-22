# Architecture Overview

BikoDB is organised as a multi-crate Rust workspace.

## Workspace layers

| Layer | What it contains |
| --- | --- |
| `bikodb-core` | IDs, values, schema-related types, error types, plugin traits |
| `bikodb-storage` | Pages, WAL, mmap, compression, storage primitives |
| `bikodb-graph` | Concurrent graph structures, traversals, algorithms, transaction layer |
| `bikodb-execution` | Logical plans, optimizer, execution pipeline, internal metrics helpers |
| `bikodb-query` | SQL, Cypher, and Gremlin subset parsers |
| `bikodb-ai` | HNSW, embeddings, inference, incremental/event-driven AI helpers |
| `bikodb-resource` | Resource monitoring primitives |
| `bikodb-cluster` | Cluster and routing building blocks |
| `bikodb-server` | Embedded `Database` facade plus HTTP router library |
| `bikodb-python` / `bikodb-node` | FFI entry points |
| `bikodb-bench` | Benchmarks and comparison report tooling |

## What matters most to external users

The important architectural fact is not just that these crates exist, but that they do **not all represent the same maturity level**.

Today the clearest story is:

- graph core + embedded Rust API are the strongest path,
- query languages should be read as documented subsets,
- multi-model helpers exist but are not yet described as one hardened transactional surface,
- cluster and AI surface area are broader than the public operational story.

## Questions evaluators usually ask

The most important architecture questions are usually:

- what is actually guaranteed today,
- how far the transaction layer really goes,
- how multi-model consistency should be interpreted,
- what can be observed and operated confidently under real workloads.

These docs now answer those questions explicitly instead of leaving them implicit in the code.

## Read next

- [Transactions & concurrency →](/architecture/transactions)
- [Multi-model consistency →](/architecture/multi-model-consistency)
- [Supported guarantees →](/architecture/supported-guarantees)
- [Operations & observability →](/architecture/operations-observability)
- [Supported features & limitations →](/guide/supported-features)
