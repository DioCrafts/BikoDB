<p align="center">
  <img src="image/logo.png" alt="BikoDB" width="320">
</p>

<p align="center">
  Multi-model graph runtime in Rust for graph, document, and vector workloads.
</p>

<p align="center">
  <a href="docs/index.md">Documentation</a>
  ·
  <a href="docs/guide/quickstart.md">Quickstart</a>
  ·
  <a href="docs/guide/supported-features.md">Supported Features</a>
  ·
  <a href="docs/architecture/supported-guarantees.md">Supported Guarantees</a>
  ·
  <a href="docs/architecture/operations-observability.md">Operations & Observability</a>
  ·
  <a href="docs/architecture/multi-model-consistency.md">Multi-model Consistency</a>
  ·
  <a href="docs/architecture/transactions.md">Transactions</a>
  ·
  <a href="docs/faq.md">FAQ</a>
  ·
  <a href="BENCHMARK_COMPARISON.md">Benchmark Report</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Rust-orange?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/docs-VitePress-blue" alt="Docs">
</p>

## What BikoDB is today

BikoDB is a Rust workspace that combines:

- a concurrent graph engine,
- a document store,
- HNSW-based vector search,
- query parsers for SQL, Cypher, and Gremlin subsets,
- an embedded `Database` facade,
- an experimental HTTP router and bindings layer.

The project is most credible today as an **embedded graph runtime with multi-model building blocks**.

## Project status by module

| Area | Status | Notes |
| --- | --- | --- |
| Graph core and algorithms | Stable | Strongest and most exercised part of the repository |
| Embedded Rust `Database` API | Stable | Best current entry point for external evaluation |
| SQL/Cypher/Gremlin read subset | Stable | Supported as documented subsets, not full language compatibility |
| Document store | Stable | In-process CRUD and filtered queries |
| HNSW vector search | Experimental | Real and usable, but part of a broader evolving AI surface |
| HTTP router | Experimental | Router exists as a library component, not a packaged server product |
| Incremental embeddings / inference | Experimental | Implemented, but parts of the AI stack are still evolving |
| Cluster / HA modules | Experimental | Present in the workspace, not yet documented as hardened operations |
| Python / Node bindings | Experimental | Available, but not yet the main documented adoption path |
| Docker-first packaged deployment | Planned | Not the primary way to start BikoDB today |

This repo will look more serious by being precise about maturity than by claiming every module is production-ready.

## Quick start

### Build and validate the workspace

```bash
git clone https://github.com/unnamedlab/BikoDB.git
cd BikoDB
cargo test --workspace
cargo build --workspace
```

### Use the embedded API

```rust
use bikodb_core::types::TypeId;
use bikodb_core::value::Value;
use bikodb_server::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Database::new();

    db.register_type("Person", TypeId(1));
    db.register_property("name", 0);
    db.register_property("age", 1);
    db.register_relationship("KNOWS", TypeId(10));

    let alice = db.create_node(TypeId(1), vec![
        ("name", Value::from("Alice")),
        ("age", Value::Int(30)),
    ]);
    let bob = db.create_node(TypeId(1), vec![
        ("name", Value::from("Bob")),
        ("age", Value::Int(25)),
    ]);

    db.create_edge(alice, bob, TypeId(10))?;

    let rows = db.query_sql("SELECT * FROM Person WHERE age > 20")?;
    println!("rows = {}", rows.len());

    Ok(())
}
```

More onboarding:

- [Installation](docs/guide/installation.md)
- [Quickstart](docs/guide/quickstart.md)
- [Supported features & limitations](docs/guide/supported-features.md)

## Supported query surface

BikoDB currently supports **documented subsets** of each language.

| Language | Publicly supported today |
| --- | --- |
| SQL | `SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, `COUNT`, `GROUP BY`, and a documented predicate subset |
| Cypher | `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `ORDER BY`, `LIMIT`, `count(...)`, and supported path traversal subset |
| Gremlin | `g.V()`, `has`, `hasLabel`, `out/in/both`, `values`, `count`, `limit`, `order().by(...)` |

Important boundary:

- parser work exists beyond this in some places,
- but the main `Database` query dispatcher is currently **read-oriented**,
- so SQL/Cypher writes should not yet be marketed as a finished public query surface.

## Multi-model scope

BikoDB already combines graph, document, and vector components inside one codebase.

What exists now:

- graph nodes and edges,
- document collections with nested-field filtering,
- HNSW vector indexing,
- helper methods that bridge graph, documents, and vectors.

What is **not** yet a finished public contract:

- documented cross-model conflict detection,
- cross-model atomicity guarantees,
- a full public transaction API that spans graph + document + vector state.

See [transactions & concurrency](docs/architecture/transactions.md) and [multi-model consistency](docs/architecture/multi-model-consistency.md).

## Questions evaluators usually ask

### How are cross-model write conflicts handled?

The honest answer today is: **there is not yet a hardened public cross-model conflict-resolution contract**.

BikoDB does support multi-model composition behind one embedded `Database` facade, but that should not currently be marketed as shared versioning, entity-level conflict detection, or atomic commit across graph + document + vector state.

Read:

- [Multi-model consistency](docs/architecture/multi-model-consistency.md)
- [Supported guarantees](docs/architecture/supported-guarantees.md)

### What can you safely claim about transactions and guarantees?

The strongest guarantee story in the repo today is the **graph transaction layer**:

- buffered graph writes,
- commit/rollback support,
- optional WAL-aware commit flow.

That is real, but it is not the same thing as a full product-wide transactional contract across every public API and every model combination.

Read:

- [Transactions & concurrency](docs/architecture/transactions.md)
- [Supported guarantees](docs/architecture/supported-guarantees.md)

### What observability and workload controls exist today?

The repo already includes:

- `ResourceMonitor` counters and snapshots,
- `AccessTracker` execution metrics,
- `EXPLAIN` plan formatting,
- live-query and async event hooks.

What it does **not** yet expose as a finished operator-facing surface:

- slow-query reporting,
- query profiling workflows,
- timeout/cancel control,
- admission control for spikes and overload.

Read:

- [Operations & observability](docs/architecture/operations-observability.md)

### How should people interpret benchmark claims?

The safest public framing is:

- BikoDB benchmark commands are reproducible from the repo,
- the current cross-database markdown comparison is informative,
- competitor values in the report should still be treated as reference/approximate rather than a fully audited neutral harness.

Read:

- [Benchmark methodology](docs/benchmarks/methodology.md)

## Operational reality today

Useful internal primitives already exist:

- `ResourceMonitor` for counters and snapshots,
- `AccessTracker` for per-operator/query metrics,
- `EXPLAIN` plan formatting inside the optimizer,
- live query and async event hooks in the embedded API.

But the repo does not yet expose a complete operator-facing surface for:

- slow-query reporting,
- profiling,
- timeout/cancel control,
- admission control.

## Benchmarks

The repo contains real BikoDB benchmark code and a generated comparison report.

However, the serious reading is:

- BikoDB measurements are reproducible from the repo,
- the comparison report still includes approximate/reference competitor values,
- the current benchmark story is informative, but not yet a fully audited neutral harness.

Read before repeating performance claims:

- [Benchmark methodology](docs/benchmarks/methodology.md)
- [Current comparison report](BENCHMARK_COMPARISON.md)

## Documentation

BikoDB now uses a VitePress-style docs structure under [`docs/`](docs/).

Main entry points:

- [Docs home](docs/index.md)
- [Installation](docs/guide/installation.md)
- [Quickstart](docs/guide/quickstart.md)
- [Supported features & limitations](docs/guide/supported-features.md)
- [Supported guarantees](docs/architecture/supported-guarantees.md)
- [Operations & observability](docs/architecture/operations-observability.md)
- [Transactions & concurrency](docs/architecture/transactions.md)
- [Multi-model consistency](docs/architecture/multi-model-consistency.md)
- [Benchmark methodology](docs/benchmarks/methodology.md)
- [FAQ](docs/faq.md)

## Workspace layout

| Crate | Role |
| --- | --- |
| `bikodb-core` | Core IDs, values, errors, plugin traits |
| `bikodb-storage` | Pages, WAL, mmap, compression |
| `bikodb-graph` | Concurrent graph engine, algorithms, transactions |
| `bikodb-execution` | Logical plans, optimizer, execution pipeline |
| `bikodb-query` | SQL, Cypher, and Gremlin subset parsers |
| `bikodb-ai` | HNSW, embeddings, inference helpers |
| `bikodb-resource` | Resource monitoring primitives |
| `bikodb-cluster` | Cluster building blocks |
| `bikodb-server` | Embedded `Database` facade and HTTP router library |
| `bikodb-bench` | Benchmarks and report generation |
| `bikodb-python` | Python bindings |
| `bikodb-node` | Node bindings |

## Development

```bash
cargo test --workspace
cargo build --workspace
```

Docs:

```bash
cd docs
npm install
npm run docs:build
```

## License

MIT

