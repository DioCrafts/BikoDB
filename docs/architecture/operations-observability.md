# Operations & Observability

This page explains what BikoDB exposes today for operators and evaluators who want to understand runtime behaviour beyond benchmarks.

## What exists today

The repository already contains useful internal primitives:

- `ResourceMonitor` for engine-level counters and snapshots,
- `AccessTracker` for per-query and per-operator execution metrics,
- `EXPLAIN` plan formatting in the optimizer,
- live-query subscriptions and async graph event hooks in the embedded `Database` API,
- storage checkpoint and snapshot helpers in the storage layer.

These are real building blocks, but they do **not** yet add up to a polished operator-facing platform.

## Current observability surface

| Area | What exists today | Current limitation |
| --- | --- | --- |
| Basic counters | `ResourceMonitor` records query count, bytes read/written, graph memory estimate, uptime, node/edge counts | Not yet exposed as a first-class metrics endpoint or dashboard surface |
| Query execution metrics | `AccessTracker` records query frequency, operator timings, rows produced, and filter patterns | Internal helper, not yet a stable public profiling API |
| Plan inspection | Optimizer can render logical plans through `EXPLAIN` formatting | Plan introspection is available as formatting logic, not yet a complete end-user explain/analyze workflow |
| Event visibility | Live queries and async graph events exist in the embedded API | Useful for integrations, but not a full auditing or alerting surface |
| Storage durability tooling | Checkpoint, WAL, and snapshot helpers exist in storage | These are storage primitives, not yet a complete operational runbook |

## What this means in practice

For an evaluator today, the honest summary is:

- you can inspect internal behaviour to some degree,
- you can reason about execution paths and resource counters,
- you cannot yet expect a mature production operator experience comparable to established database products.

## Real workload behaviour: what to expect today

The repo is strongest under:

- in-process evaluation,
- graph-centric workloads,
- controlled benchmarks,
- API-level experimentation with graph + document + vector building blocks.

The repo is less mature today for:

- messy production workloads with many concurrent tenants,
- long-running queries that need robust timeout or cancel semantics,
- traffic spikes that need explicit admission control,
- partial-failure handling across multiple subsystems under one documented contract,
- full operational debugging from logs, traces, dashboards, and slow-query tooling alone.

## Specific questions external users tend to ask

### Slow queries

There are useful internal metrics and plan formatting primitives, but the repo does not yet present a finished slow-query reporting or profiling product surface.

### Timeouts and cancellation

There is no documented general-purpose timeout/cancel contract across the public query surface today.

### Admission control and spike handling

The workspace contains concurrency and resource-management primitives, but not a full documented admission-control layer for overloaded systems.

### Partial failures

The storage and transaction layers contain real durability and rollback-related building blocks, but the repository does not yet document a full product-level failure matrix for graph + document + vector combinations.

### Memory pressure

The repo includes monitoring and cache-budget enforcement primitives, but external users should still treat memory-governance behaviour as an area that needs more explicit public hardening and documentation.

## Recommended public framing

The safest and most credible way to describe BikoDB today is:

- strong internal building blocks for observability exist,
- graph-centric execution and instrumentation are real,
- the operator-facing surface is still emerging,
- production-grade observability and workload-management ergonomics are a roadmap area, not a finished promise.
