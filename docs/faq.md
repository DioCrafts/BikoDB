# FAQ

## Is BikoDB production-ready?

Not as a general-purpose packaged database product yet.

The repo is strongest today as an embedded Rust graph runtime with additional document and vector capabilities. Several broader areas are still better described as experimental.

## What is the recommended way to try it today?

Use the Rust workspace directly:

```bash
cargo test --workspace
cargo build --workspace
```

Then start with the embedded `Database` API shown in the [quickstart](/guide/quickstart).

## Does BikoDB support full SQL, full Cypher, or full Gremlin?

No.

It supports documented subsets of each language. The goal of these docs is to make that boundary explicit instead of implying full compatibility.

## Are SQL and Cypher writes supported through the public query endpoint?

Not as a documented end-to-end public surface today.

The parsers contain mutation-related work, but the primary `Database` query dispatch path is currently read-oriented.

## Is there a standalone server binary?

Not as the main documented entry point today.

The repo includes an HTTP router library, but not a polished packaged server distribution with first-class deployment docs.

## Are graph transactions the same as cross-model transactions?

No.

A graph transaction implementation exists in `bikodb-graph`, but that should not be read as a full public transactional contract across graph, document, and vector subsystems.

## How do conflicting updates across models on the same node work today?

The safe answer is: BikoDB does **not** yet document a hardened public cross-model conflict-resolution contract.

The repository does support multi-model composition behind one `Database` facade, but that should not be interpreted as shared versioning, OCC/MVCC conflict detection, or atomic cross-model commits across graph, document, and vector state.

## Is there cross-model atomicity or rollback?

Not as a published public guarantee today.

The strongest transaction story in the repo is graph-scoped. Cross-model helpers should currently be understood as composition helpers rather than one documented all-model transaction protocol.

## Is there version-based conflict detection?

Not as a documented public feature today.

If someone is evaluating BikoDB for enterprise-style conflict semantics, they should currently treat that as future work rather than an existing external contract.

## What observability exists today?

Useful internal primitives already exist:

- resource counters and snapshots,
- execution/access tracking,
- `EXPLAIN` plan formatting,
- event and live-query hooks.

What does **not** yet exist is a fully polished operator-facing suite for slow-query tooling, profiling, timeouts, cancellation, and admission control.

## How does BikoDB behave under messy real workloads?

The honest answer today is that the repo is strongest for embedded evaluation, graph-centric workloads, and controlled benchmarking.

It is less mature as a documented product story for partial failures, overloaded systems, long-running query control, and other operator-heavy scenarios that only show up under messy production workloads.

## Are the benchmark comparisons fully audited?

Not yet.

BikoDB measurements are reproducible from the repo, but the current cross-database comparison still includes reference/approximate competitor values and should be treated accordingly.
