// =============================================================================
// bikodb-graph — Graph Engine
// =============================================================================
// Motor de grafos con soporte para:
//
//   graph      → Grafo in-memory con adjacency list (ConcurrentGraph)
//   traversal  → BFS, DFS, shortest path, filtered traversals
//   mutation   → Insert/update/delete nodes y edges transaccionalmente
//
// ## Diseño
// - Adjacency list con SmallVec inline (cache-friendly para nodos con ≤4 edges)
// - DashMap para acceso concurrente lock-free a nodos y edges
// - Edge segments: cuando un nodo tiene >256 edges, se agrupan en segmentos
//   para evitar hotspots (inspirado en ArcadeDB edge segments)
//
// ## Rendimiento
// - BFS/DFS: ~40M edges/sec en grafos in-memory (target)
// - Insert vertex: O(1) amortizado
// - Insert edge: O(1) amortizado (append to adjacency list)
// =============================================================================

pub mod bitset;
pub mod bulk;
pub mod community;
pub mod csr;
pub mod document;
pub mod graph;
pub mod kcore;
pub mod lcc;
pub mod louvain;
pub mod mutation;
pub mod optimized_dfs;
pub mod pagerank;
pub mod parallel_bfs;
pub mod scc;
pub mod sssp;
pub mod transaction;
pub mod traversal;
pub mod weighted_csr;

pub use graph::ConcurrentGraph;
