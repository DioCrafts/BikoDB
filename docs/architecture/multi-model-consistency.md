# Multi-model Consistency

BikoDB exposes graph, document, and vector capabilities behind one workspace and one high-level `Database` facade.

That is powerful, but it also means the consistency story must be read carefully.

## What the `Database` facade coordinates

The `Database` type currently combines:

- the graph engine,
- schema maps for types/properties/relationships,
- the document store,
- optional HNSW vector index,
- embedding and inference helpers,
- plugin hooks,
- live query and event infrastructure,
- optional cluster router.

## What is safe to say today

The repo clearly supports **multi-model composition**.

Examples include:

- creating graph nodes and querying them,
- storing documents in named collections,
- linking documents to graph nodes,
- inserting vectors for graph nodes,
- creating a linked document plus vector anchor node,
- enriching vector results with graph/document context.

## What you should not assume

The presence of one facade does **not** yet imply one hardened transactional model across every subsystem.

Today you should not assume:

- atomic commit across graph + document + vector state,
- rollback across all model combinations,
- shared versioning/conflict detection across all models,
- strong ordering guarantees between graph writes and downstream embedding/inference/index updates.

## Current consistency boundaries

| Operation type | Current interpretation |
| --- | --- |
| Pure graph transaction via `bikodb-graph::transaction` | Strongest documented atomicity story in the repo |
| Direct graph mutation through `Database` helpers | Immediate in-process mutation helpers |
| Document CRUD | Separate in-process document store operations |
| HNSW insert/search | Separate vector index lifecycle that must be enabled explicitly |
| Convenience helpers like `create_node_with_document` | Sequential composition helper, not a published distributed transaction |
| Convenience helpers like `create_document_with_vector` | Sequential composition helper, not a published cross-model commit protocol |
| Incremental embedding/inference/event reactions | Event-driven integration, not documented as synchronous transactional isolation |

## Why this matters

External users evaluating a database need to know whether “multi-model” means:

- one UI and one facade, or
- one well-defined transactional contract across all model types.

For BikoDB today, the accurate answer is closer to the first one.

## Recommended mental model

Treat BikoDB today as:

- a strong graph-centric core,
- plus useful document/vector integrations,
- plus experimental cross-model helpers,
- with future room to define tighter cross-model atomicity and conflict semantics.
