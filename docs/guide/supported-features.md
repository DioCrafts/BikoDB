# Supported Features & Limitations

This page is the contract for external users. It intentionally distinguishes between what is ready to evaluate now and what is still experimental or incomplete.

## Maturity matrix

| Capability | Status | Notes |
| --- | --- | --- |
| Concurrent graph storage and traversal APIs | Stable | Core graph engine is the strongest part of the repo. |
| Embedded `Database` facade for Rust | Stable | Best current entry point. |
| Graph algorithms in the workspace | Stable | Multiple algorithms are implemented and tested. |
| SQL read subset | Stable | `SELECT`, filtering, ordering, limit, `COUNT`, and `GROUP BY` subset are implemented. |
| Cypher read subset | Stable | `MATCH`-style reads, filters, ordering, limits, counts, and path traversal subset are implemented. |
| Gremlin traversal subset | Stable | `g.V()`, `has`, `out/in/both`, `values`, `count`, `limit`, and ordering subset are implemented. |
| Document store | Stable | Collection-based CRUD plus nested-field filtering in-process. |
| Resource monitor and internal query metrics primitives | Stable | Useful internally, but not yet a full operator-facing observability surface. |
| HNSW vector index | Experimental | Implemented and integrated, but still part of a broader evolving AI/vector story. |
| Live queries / async graph events | Experimental | Present in the embedded API, not yet part of a hardened ops surface. |
| HTTP router | Experimental | Router exists and is tested, but the repo does not ship a standalone server runtime. |
| Python and Node bindings | Experimental | Available in the workspace, but not yet the primary documented adoption path. |
| Incremental embeddings / inference | Experimental | Real code exists, but parts of the AI path still contain placeholder/evolving pieces. |
| Cluster / sharding / HA | Experimental | Present in the workspace, but not yet documented as production-ready operations. |
| Docker-first deployment story | Planned | Not the current entry path. |
| Production-grade operator controls (timeouts, admission control, slow-query tooling) | Planned | Internal primitives exist, external surface is still limited. |

## Query language coverage

BikoDB should currently be understood as supporting **documented subsets**, not full language compatibility.

### SQL

Supported subset today includes:

- `SELECT ... FROM ...`
- `WHERE`
- `ORDER BY`
- `LIMIT`
- `COUNT(*)`
- `GROUP BY`
- predicates such as `=`, `!=`, `>`, `<`, `>=`, `<=`, `LIKE`, `IN`, `BETWEEN`

Important limitation:

- the parser contains broader work such as `INSERT` and `DELETE`,
- but the current `Database::query_sql()` execution path is a **read path**,
- so you should not treat SQL writes as a public, end-to-end supported query surface yet.

### Cypher

Supported subset today includes:

- `MATCH`
- `OPTIONAL MATCH`
- `WHERE`
- `RETURN`
- `ORDER BY`
- `LIMIT`
- `count(...)`
- variable-length path syntax for supported traversal cases

Important limitation:

- parser work exists for `CREATE`, `DELETE`, and `SET`,
- but the public `Database::query_cypher()` path is currently a **read-oriented execution path**.

### Gremlin

Supported subset today includes:

- `g.V()`
- `hasLabel(...)`
- `has(...)`
- `out(...)`, `in(...)`, `both(...)`
- `values(...)`
- `count()`
- `limit(...)`
- `order().by(...)`

Important limitation:

- this is a traversal subset, not full Apache TinkerPop compatibility.

## Multi-model scope

What exists today:

- graph nodes and edges,
- document collections,
- HNSW vector search,
- helper methods that bridge graph, documents, and vectors.

What you should **not** assume today:

- a single public transaction API spanning graph + document + vector state,
- documented conflict semantics across all models,
- versioned cross-model isolation guarantees.

See [multi-model consistency](/architecture/multi-model-consistency) for the current boundaries.
See [supported guarantees](/architecture/supported-guarantees) for the shortest summary of what is safe to claim publicly.

## Operational limitations

Useful internal pieces already exist:

- `ResourceMonitor` for counters and snapshots,
- `AccessTracker` for per-operator/query metrics,
- `EXPLAIN` formatter in the optimizer.

But the repo does not yet present a complete operator-facing layer for:

- slow query reporting,
- query profiling,
- timeout/cancel controls,
- admission control and concurrency governance.

See [operations & observability](/architecture/operations-observability) for the current operator-facing picture.

## Packaging limitations

External users should know this up front:

- there is no official standalone BikoDB server binary at the repo root,
- there is no first-class Docker deployment flow in the repo,
- the best current starting point is still the embedded Rust API.
