// =============================================================================
// bikodb-bench — Benchmark suite
// =============================================================================
// Benchmarks para medir rendimiento del motor en operaciones clave:
//
// - Graph traversals (BFS, DFS)
// - Node/edge insertions
// - Query execution (SQL filter)
// - Vector similarity search
// - LDBC Graphalytics (BFS, SSSP, PageRank, CDLP, WCC, LCC)
// - AI/ML (embeddings, vector search, hybrid queries)
//
// ## Uso
// ```sh
// cargo bench -p bikodb-bench
// cargo bench -p bikodb-bench --bench ldbc_graphalytics
// ```
//
// ## Comparativa target (vs ArcadeDB / Neo4j)
// - Insert vertex: > 500K/sec
// - BFS traversal: > 10M edges/sec
// - SQL filter query: > 100K rows/sec
// - Vector k-NN (1K vectors, dim=128): < 1ms
// =============================================================================

pub mod comparison;
pub mod generators;
pub mod ldbc;
