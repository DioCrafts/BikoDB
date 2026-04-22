# Supported Guarantees

This page summarises the guarantees that are most reasonable to claim from the repository as it exists today.

## Guarantee matrix

| Concern | What is reasonable to claim today | What you should not claim yet |
| --- | --- | --- |
| Graph write atomicity | The graph transaction layer provides a buffered write-set with commit/rollback semantics | A product-wide ACID story across every public entry point |
| Graph durability with WAL | WAL-aware commit flow exists when `TxManager` is configured with a WAL | That every mutation path in the full product surface shares the same durability contract |
| Rollback | Graph transaction rollback exists and active graph transactions roll back on drop | Cross-model rollback across graph + document + vector state |
| Read isolation | Reads can observe committed graph state | Public MVCC snapshots or a published isolation-level matrix |
| Conflict detection | Some transactional structure exists for graph writes | Public entity-version OCC/MVCC conflict semantics across all models |
| Cross-model consistency | One `Database` facade coordinates graph, document, vector, and event-related components | Atomic commit or shared versioning across every model combination |
| Query language compatibility | Documented subsets of SQL, Cypher, and Gremlin are supported | Full SQL, full openCypher, or full TinkerPop compatibility |
| Operational introspection | Internal counters, execution metrics, and plan formatting exist | A complete operator-facing observability suite |
| Storage snapshots | Storage snapshot and restore helpers exist in the storage layer | A finished backup/restore operations product for the whole public surface |

## Atomicity

The strongest atomicity story in the repo today is the graph transaction layer in `bikodb-graph::transaction`.

That supports:

- local buffering of graph writes,
- commit/rollback flow,
- optional WAL-aware commit orchestration.

That does **not** currently imply:

- atomic graph + document + vector commits,
- one public transactional contract spanning all helper methods in the `Database` facade.

## Isolation

The current transaction docs support a conservative reading:

- writes are buffered until commit,
- reads can see committed graph state,
- there is no published MVCC snapshot contract for external users.

So the credible public claim is **graph-scoped buffered writes**, not a full isolation-level story.

## Ordering

Within graph transaction handling, commit-time application order and WAL-aware sequencing exist as real building blocks.

However, you should not currently assume:

- strict documented ordering between graph mutation and downstream event/inference/vector update paths,
- one ordering contract across all cross-model helper flows.

## Durability

Durability-related pieces clearly exist in the workspace:

- WAL in storage,
- WAL-aware graph transaction management,
- checkpoints,
- snapshot and restore helpers.

What remains important is scope:

- these are meaningful durability primitives,
- they are not yet the same thing as a polished product-wide durability guarantee across all public usage patterns.

## Cross-model consistency

The accurate public posture today is:

- BikoDB supports multi-model composition,
- some helpers combine graph, document, and vector workflows,
- those helpers should currently be understood as sequential composition helpers rather than a documented cross-model commit protocol.

## Recommended wording for external users

If someone asks what BikoDB guarantees today, the short answer is:

> Strongest guarantees are graph-scoped transaction and durability building blocks. Multi-model composition exists, but cross-model atomicity, conflict detection, and operator-facing guarantees are not yet documented as a hardened public contract.
