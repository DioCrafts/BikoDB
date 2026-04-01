# BikoDB Documentation Index

> Estado: Estructura planificada. Los documentos marcados con `📄` están pendientes de redacción.

---

## 1. Getting Started

| Doc | Descripción |
|-----|-------------|
| 📄 [installation.md](installation.md) | Requisitos (Rust 1.93+), compilación, plataformas soportadas (Linux, macOS, Windows) |
| 📄 [quickstart.md](quickstart.md) | Primer grafo en 5 minutos: crear nodos, aristas, queries, vector search |
| 📄 [configuration.md](configuration.md) | Opciones de configuración: durabilidad, page cache, thread pools, puertos |
| 📄 [docker.md](docker.md) | Imagen Docker, docker-compose, variables de entorno |

---

## 2. Conceptos Fundamentales

| Doc | Descripción |
|-----|-------------|
| 📄 [data-model.md](data-model.md) | Modelo multi-modelo: grafos (nodos, aristas, propiedades), documentos, vectores |
| 📄 [types.md](types.md) | Sistema de tipos: `Value` enum, `NodeId`, `EdgeId`, `TypeId`, `RID`, `Schema` |
| 📄 [transactions.md](transactions.md) | ACID, aislamiento, durabilidad, snapshots, lock-free concurrency |
| 📄 [storage-internals.md](storage-internals.md) | Páginas 64KB, page cache LRU, WAL, mmap, delta encoding, compresión LZ4 |

---

## 3. Query Languages

| Doc | Descripción |
|-----|-------------|
| 📄 [sql.md](sql.md) | Dialecto SQL para grafos: SELECT, INSERT, DELETE, COUNT, GROUP BY, ORDER BY, WHERE |
| 📄 [cypher.md](cypher.md) | Soporte Cypher: MATCH, CREATE, DELETE, SET, RETURN, WHERE, ORDER BY, patrones de traversal |
| 📄 [gremlin.md](gremlin.md) | Intérprete Gremlin: `g.V()`, `has()`, `out()`, `in()`, `both()`, `values()`, `count()` |
| 📄 [query-planner.md](query-planner.md) | Pipeline Volcano (pull-based), operadores físicos, optimizador de costes, plan cache |

---

## 4. Graph Algorithms

| Doc | Descripción |
|-----|-------------|
| 📄 [algorithms-overview.md](algorithms-overview.md) | Tabla resumen de los 11 algoritmos, CSR, cuándo usar cada uno |
| 📄 [bfs.md](bfs.md) | BFS paralelo (level-synchronous) + BFS direction-optimizing (Beamer push/pull) |
| 📄 [dfs.md](dfs.md) | DFS iterativo optimizado, iterative deepening |
| 📄 [sssp.md](sssp.md) | SSSP adaptativo: selección automática BFS / Dijkstra / Bellman-Ford / Δ-stepping |
| 📄 [pagerank.md](pagerank.md) | PageRank pull-based paralelo sobre CSR, configuración de damping/tolerancia/iteraciones |
| 📄 [community-detection.md](community-detection.md) | WCC (union-find lock-free), CDLP (label propagation), Louvain (multi-level modularity) |
| 📄 [scc.md](scc.md) | SCC: Tarjan iterativo + Kosaraju, diferencias y cuándo usar cada uno |
| 📄 [lcc.md](lcc.md) | Coeficiente de clustering local, conteo de triángulos paralelo |
| 📄 [kcore.md](kcore.md) | K-core decomposition: algoritmo peeling, degeneracy |
| 📄 [csr.md](csr.md) | CSR y WeightedCSR: construcción desde ConcurrentGraph, layout de memoria, rendimiento |

---

## 5. AI/ML

| Doc | Descripción |
|-----|-------------|
| 📄 [vector-search.md](vector-search.md) | HNSW: inserción, búsqueda k-NN, métricas (coseno, euclídea), configuración de capas |
| 📄 [gnn.md](gnn.md) | GraphSAGE: message passing, agregación de vecinos, entrenamiento, inferencia |
| 📄 [embeddings.md](embeddings.md) | Generación de embeddings desde propiedades, `Value::Embedding`, pipeline incremental |
| 📄 [incremental-inference.md](incremental-inference.md) | Re-inferencia en tiempo real ante mutaciones, event bus, actualizaciones parciales |

---

## 6. REST API

| Doc | Descripción |
|-----|-------------|
| 📄 [api-overview.md](api-overview.md) | Visión general: base URL, autenticación, formato JSON, códigos de error |
| 📄 [api-schema.md](api-schema.md) | `POST /schema/types`, `/schema/properties`, `/schema/relationships` |
| 📄 [api-nodes.md](api-nodes.md) | CRUD de nodos: crear, leer, eliminar, actualizar propiedades, obtener vecinos |
| 📄 [api-edges.md](api-edges.md) | Crear y eliminar aristas |
| 📄 [api-query.md](api-query.md) | `POST /query` — dispatch SQL / Cypher / Gremlin, formato de respuesta |
| 📄 [api-vectors.md](api-vectors.md) | `POST /vectors/insert`, `POST /vectors/search` — operaciones con vectores |
| 📄 [api-documents.md](api-documents.md) | CRUD de documentos: crear, leer por colección, queries con dot-path |
| 📄 [api-plugins.md](api-plugins.md) | `POST /udf/call`, `POST /algorithms/run` — ejecución de plugins y algoritmos |
| 📄 [api-status.md](api-status.md) | `GET /status`, `GET /health` — monitorización |

---

## 7. FFI Bindings

| Doc | Descripción |
|-----|-------------|
| 📄 [python-bindings.md](python-bindings.md) | Instalación, ctypes/cffi, `bikodb_create()`, JSON dispatch, ejemplos completos |
| 📄 [nodejs-bindings.md](nodejs-bindings.md) | Instalación, ffi-napi/koffi, 14 funciones C, ejemplos completos |
| 📄 [ffi-protocol.md](ffi-protocol.md) | Convención C ABI, gestión de memoria (`bikodb_free_string`), manejo de errores |

---

## 8. Cluster & Distribución

| Doc | Descripción |
|-----|-------------|
| 📄 [clustering.md](clustering.md) | Arquitectura distribuida: sharding, replicación, balanceo de carga |
| 📄 [sharding.md](sharding.md) | Estrategias: hash, range, graph-aware partitioning |
| 📄 [replication.md](replication.md) | Modos de replicación: One, Quorum, All — consistencia y trade-offs |
| 📄 [leader-election.md](leader-election.md) | Raft-style: heartbeat, suspect, down, failover automático |

---

## 9. Storage Engine

| Doc | Descripción |
|-----|-------------|
| 📄 [pages.md](pages.md) | Formato de página 64KB, PageHeader, slots, overflow |
| 📄 [page-cache.md](page-cache.md) | LRU con lecturas lock-free, eviction policy, configuración de tamaño |
| 📄 [wal.md](wal.md) | Write-Ahead Log: formato de entradas, checkpointing, crash recovery |
| 📄 [mmap.md](mmap.md) | Memory-mapped I/O: zero-copy reads, alignment, platform differences |
| 📄 [compression.md](compression.md) | LZ4 en codec, delta encoding, dictionary compression (string→u16) |
| 📄 [durability-modes.md](durability-modes.md) | DurabilityMode: Sync, Async, None — trade-offs rendimiento vs seguridad |

---

## 10. Plugin System

| Doc | Descripción |
|-----|-------------|
| 📄 [plugin-architecture.md](plugin-architecture.md) | Trait `Plugin`: lifecycle (`init`, `shutdown`), hooks, UDFs |
| 📄 [writing-a-plugin.md](writing-a-plugin.md) | Tutorial: crear un plugin custom paso a paso |
| 📄 [algorithm-plugins.md](algorithm-plugins.md) | Trait `AlgorithmPlugin`: `execute()`, `AlgorithmInput`, `AlgorithmResult` |
| 📄 [hooks.md](hooks.md) | Hooks pre/post: insert, update, delete — interceptar mutaciones |

---

## 11. Architecture & Internals

| Doc | Descripción |
|-----|-------------|
| 📄 [architecture.md](architecture.md) | Diagrama de 12 crates, dependencias, flujo de datos |
| 📄 [concurrency-model.md](concurrency-model.md) | DashMap, rayon, lock-free atomics, SmallVec<4>, edge segments |
| 📄 [execution-engine.md](execution-engine.md) | Volcano pull-based: operadores, backpressure, lazy evaluation |
| 📄 [memory-management.md](memory-management.md) | Resource monitor, memory budgets, estimated_heap_bytes, zero-alloc paths |
| 📄 [error-handling.md](error-handling.md) | `BikoError` enum, `BikoResult<T>`, conversión a HTTP status codes |

---

## 12. Benchmarking

| Doc | Descripción |
|-----|-------------|
| 📄 [benchmarking-guide.md](benchmarking-guide.md) | Cómo ejecutar benchmarks: criterion, comparison report, escalas |
| 📄 [ldbc-graphalytics.md](ldbc-graphalytics.md) | 6 algoritmos LDBC, generador de grafos (XS–XL), formato de resultados |
| 📄 [competitive-comparison.md](competitive-comparison.md) | Metodología de comparación vs ArcadeDB, Kuzu, Neo4j, valores de referencia |
| 📄 [ai-benchmarks.md](ai-benchmarks.md) | HNSW throughput, k-NN latencia, recall@k |

---

## 13. Contributing

| Doc | Descripción |
|-----|-------------|
| 📄 [contributing.md](contributing.md) | Guía de contribución: fork, branch, PR, estilo de código |
| 📄 [code-style.md](code-style.md) | Convenciones Rust: naming, estructura de módulos, documentación |
| 📄 [testing.md](testing.md) | Cómo escribir tests, 788 tests existentes, desglose por crate |
| 📄 [release-process.md](release-process.md) | Versionado semántico, changelog, publicación de crates |

---

## 14. Reference

| Doc | Descripción |
|-----|-------------|
| 📄 [glossary.md](glossary.md) | Glosario: CSR, HNSW, GNN, WAL, CDLP, LCC, SCC, SmallVec, etc. |
| 📄 [faq.md](faq.md) | Preguntas frecuentes |
| 📄 [changelog.md](changelog.md) | Historial de cambios por versión |
| 📄 [roadmap.md](roadmap.md) | Funcionalidades futuras y priorización |

---

## Mapa visual

```
docs/
├── INDEX.md                          ← este archivo
│
├── 01-getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   ├── configuration.md
│   └── docker.md
│
├── 02-concepts/
│   ├── data-model.md
│   ├── types.md
│   ├── transactions.md
│   └── storage-internals.md
│
├── 03-query-languages/
│   ├── sql.md
│   ├── cypher.md
│   ├── gremlin.md
│   └── query-planner.md
│
├── 04-algorithms/
│   ├── algorithms-overview.md
│   ├── bfs.md
│   ├── dfs.md
│   ├── sssp.md
│   ├── pagerank.md
│   ├── community-detection.md
│   ├── scc.md
│   ├── lcc.md
│   ├── kcore.md
│   └── csr.md
│
├── 05-ai-ml/
│   ├── vector-search.md
│   ├── gnn.md
│   ├── embeddings.md
│   └── incremental-inference.md
│
├── 06-rest-api/
│   ├── api-overview.md
│   ├── api-schema.md
│   ├── api-nodes.md
│   ├── api-edges.md
│   ├── api-query.md
│   ├── api-vectors.md
│   ├── api-documents.md
│   ├── api-plugins.md
│   └── api-status.md
│
├── 07-ffi-bindings/
│   ├── python-bindings.md
│   ├── nodejs-bindings.md
│   └── ffi-protocol.md
│
├── 08-cluster/
│   ├── clustering.md
│   ├── sharding.md
│   ├── replication.md
│   └── leader-election.md
│
├── 09-storage-engine/
│   ├── pages.md
│   ├── page-cache.md
│   ├── wal.md
│   ├── mmap.md
│   ├── compression.md
│   └── durability-modes.md
│
├── 10-plugins/
│   ├── plugin-architecture.md
│   ├── writing-a-plugin.md
│   ├── algorithm-plugins.md
│   └── hooks.md
│
├── 11-internals/
│   ├── architecture.md
│   ├── concurrency-model.md
│   ├── execution-engine.md
│   ├── memory-management.md
│   └── error-handling.md
│
├── 12-benchmarking/
│   ├── benchmarking-guide.md
│   ├── ldbc-graphalytics.md
│   ├── competitive-comparison.md
│   └── ai-benchmarks.md
│
├── 13-contributing/
│   ├── contributing.md
│   ├── code-style.md
│   ├── testing.md
│   └── release-process.md
│
└── 14-reference/
    ├── glossary.md
    ├── faq.md
    ├── changelog.md
    └── roadmap.md
```

**Total: 14 secciones, 58 documentos planificados.**
