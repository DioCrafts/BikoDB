// =============================================================================
// bikodb-ai — AI/ML Integration Layer
// =============================================================================
// Integración nativa de IA en el motor de knowledge graph.
//
//   embedding   → Distancias y utilidades para embeddings vectoriales
//   vector_idx  → Índice vectorial flat (brute-force) para k-NN
//   hnsw        → Índice HNSW (O(log n)) para ANN escalable
//   gnn         → Graph Neural Networks: message passing, GraphSAGE layers
//   inference   → Motor de inferencia directa (MLP + GNN heads)
//   incremental → Inferencia incremental + embeddings en tiempo real
//   embed_gen   → Generación de embeddings desde propiedades del grafo
//
// ## Diseño
// - Embeddings como tipo nativo (Value::Embedding) en nodos/edges
// - GNN engine nativo en Rust (sin PyTorch/TF)
// - Event bus para re-inferencia automática en mutaciones
// - Pipeline de embeddings actualizables en tiempo real
// - HNSW para búsqueda ANN en O(log n)
//
// ## Inspiración
// - ArcadeDB: Vector search con HNSW
// - Neo4j: Graph Data Science library
// - Qdrant/Milvus: Vector databases
// - PyG/DGL: GNN frameworks (API inspiration)
// =============================================================================

pub mod embed_gen;
pub mod embedding;
pub mod gnn;
pub mod hnsw;
pub mod incremental;
pub mod inference;
pub mod vector_idx;
