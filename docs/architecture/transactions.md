# Transactions & Concurrency

This page documents the current semantics that are visible in the codebase today.

## Scope of the transaction layer

The clearest transaction implementation currently lives in `bikodb-graph::transaction`.

That layer provides:

- a `Transaction` type with a local write-set,
- `commit()` and `rollback()`,
- optional WAL integration through `TxManager`,
- commit-time application of buffered graph mutations.

## What it guarantees today

For graph mutations executed through this transaction API, the current model is:

1. operations accumulate in a local write-set,
2. writes are not visible until commit,
3. commit applies buffered graph work as an all-or-nothing sequence,
4. WAL entries can be written before graph application when `TxManager` is configured with a WAL,
5. rollback discards buffered work, and active transactions roll back on drop.

In practical terms, this is a **graph-scoped atomic write buffer with WAL-aware commit flow**.

## Read behaviour

The current transaction API uses read-through access to the underlying graph.

That means:

- reads can see committed graph state,
- reads do **not** expose uncommitted writes from the current transaction as a separate snapshot view,
- there is no public MVCC-style snapshot model described in the current implementation.

## What is not documented as guaranteed today

These are the main limits external users should understand:

- no documented MVCC implementation,
- no public entity-version conflict detection model,
- no explicit optimistic concurrency protocol based on row/entity versions,
- no user-facing isolation-level matrix,
- no public cross-model transaction contract spanning graph + document + vector state.

`TxManager` currently focuses on:

- transaction IDs,
- active transaction bookkeeping,
- WAL serialization,
- commit/rollback orchestration.

That is useful and real, but it is not the same thing as a published OCC/MVCC conflict story.

## Relationship to the top-level `Database` facade

This is an important boundary:

- the graph transaction API exists in `bikodb-graph`,
- the top-level `bikodb-server::Database` facade mainly exposes immediate mutation helpers such as `create_node`, `create_edge`, and `set_node_property`,
- those helpers are not currently presented as one public transaction API spanning every model.

So if you evaluate BikoDB today, you should distinguish between:

- **graph transaction building blocks that exist**, and
- **a full public database transaction story across the whole product surface**, which is not yet fully defined.

## Concurrency model in practice

The broader workspace also uses concurrent Rust primitives such as:

- `DashMap`-based structures,
- atomic counters,
- `rayon` for parallel graph algorithms,
- internal resource and execution tracking helpers.

This supports good in-process concurrency, but external users should still avoid assuming enterprise-grade transactional semantics that have not yet been documented explicitly.
