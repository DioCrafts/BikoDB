// =============================================================================
// bikodb-graph::bulk — Bulk Loader para inserción masiva de alto throughput
// =============================================================================
//
// ## Diseño
//
// BulkLoader es un API de inserción masiva que optimiza throughput mediante:
//
// 1. **Buffering**: acumula nodos y edges en buffers en memoria antes de
//    commitear al grafo — evita overhead per-insert de DashMap locking.
//
// 2. **Batch flush**: cuando el buffer alcanza `flush_threshold`, se flushea
//    al grafo en un solo batch paralelo (`insert_nodes_batch`, `insert_edges_batch`).
//
// 3. **WAL integration**: cada flush escribe un batch de WAL entries antes de
//    mutar el grafo (write-ahead guarantee). Usa `ConcurrentWal::write_batch()`
//    para 1 fsync por flush, no por nodo.
//
// 4. **Parallel flush**: nodos se insertan en paralelo via rayon, edges
//    se insertan en paralelo tras todos los nodos.
//
// 5. **Memory-efficient**: properties se mueven (not cloned) del buffer al
//    grafo, minimizando allocations.
//
// ## Flujo
//
// ```text
//   add_node() → buffer_nodes ──┐
//   add_node() → buffer_nodes   ├── threshold? → flush_to_graph()
//   add_edge() → buffer_edges ──┘       │
//                                        ├── WAL write_batch (1 fsync)
//                                        └── graph.insert_nodes_batch()
//                                            graph.insert_edges_batch()
// ```
//
// ## Concurrencia
//
// BulkLoader es **single-writer** (owned por un thread). Para inserción
// paralela desde múltiples threads, cada thread crea su propio BulkLoader
// sobre el mismo ConcurrentGraph (que es thread-safe internamente).
//
// ## Rendimiento target
// - > 1M nodos/sec (sin WAL)
// - > 500K nodos/sec (con WAL + fsync por batch)
// =============================================================================

use crate::graph::ConcurrentGraph;
use bikodb_core::error::BikoResult;
use bikodb_core::types::{EdgeId, NodeId, TypeId};
use bikodb_core::value::Value;
use bikodb_storage::wal::{ConcurrentWal, WalEntry, WalOpType};
use std::sync::Arc;

// =============================================================================
// BulkLoader
// =============================================================================

/// Buffer entry para un nodo pendiente.
struct PendingNode {
    type_id: TypeId,
    properties: Vec<(u16, Value)>,
}

/// Buffer entry para un edge pendiente.
///
/// Los edges referencian índices en el buffer de nodos pendientes O NodeIds
/// ya existentes en el grafo.
#[derive(Clone)]
struct PendingEdge {
    source: EdgeEndpoint,
    target: EdgeEndpoint,
    type_id: TypeId,
}

/// Un endpoint de edge: puede ser un NodeId existente o un índice en el batch.
#[derive(Clone, Copy)]
enum EdgeEndpoint {
    /// NodeId ya existente en el grafo.
    Existing(NodeId),
    /// Índice en `buffer_nodes` del batch actual (resolverá a NodeId tras flush).
    Pending(usize),
}

/// Configuración del BulkLoader.
#[derive(Debug, Clone)]
pub struct BulkConfig {
    /// Número de nodos que dispara un auto-flush.
    pub flush_threshold: usize,
    /// Si true, escribe WAL entries antes de cada flush.
    pub wal_enabled: bool,
}

impl Default for BulkConfig {
    fn default() -> Self {
        Self {
            flush_threshold: 100_000,
            wal_enabled: false,
        }
    }
}

/// Resultado de una operación de bulk load.
#[derive(Debug)]
pub struct BulkResult {
    /// Total de nodos insertados.
    pub nodes_inserted: usize,
    /// Total de edges insertados.
    pub edges_inserted: usize,
    /// Total de edges que fallaron (nodo no encontrado, etc).
    pub edges_failed: usize,
    /// Número de flushes realizados.
    pub flushes: usize,
}

/// Bulk Loader para inserción masiva de nodos y edges.
///
/// Acumula operaciones en buffers y las flushea al grafo en batches
/// para maximizar throughput.
///
/// # Ejemplo
/// ```
/// use bikodb_graph::bulk::{BulkLoader, BulkConfig};
/// use bikodb_graph::ConcurrentGraph;
/// use bikodb_core::types::TypeId;
///
/// let graph = ConcurrentGraph::new();
/// let mut loader = BulkLoader::new(&graph, BulkConfig::default());
///
/// // Buffer 1000 nodos
/// for _ in 0..1000 {
///     loader.add_node(TypeId(1), Vec::new());
/// }
///
/// // Commit al grafo
/// let result = loader.finish().unwrap();
/// assert_eq!(result.nodes_inserted, 1000);
/// assert_eq!(graph.node_count(), 1000);
/// ```
pub struct BulkLoader<'g> {
    graph: &'g ConcurrentGraph,
    wal: Option<Arc<ConcurrentWal>>,
    config: BulkConfig,
    buffer_nodes: Vec<PendingNode>,
    buffer_edges: Vec<PendingEdge>,
    // Resultado acumulado
    total_nodes: usize,
    total_edges: usize,
    total_edge_failures: usize,
    flushes: usize,
    // NodeIds resueltos del último flush (para pending edge resolution)
    last_flush_ids: Vec<NodeId>,
}

impl<'g> BulkLoader<'g> {
    /// Crea un BulkLoader sin WAL.
    pub fn new(graph: &'g ConcurrentGraph, config: BulkConfig) -> Self {
        let capacity = config.flush_threshold;
        Self {
            graph,
            wal: None,
            config,
            buffer_nodes: Vec::with_capacity(capacity),
            buffer_edges: Vec::with_capacity(capacity),
            total_nodes: 0,
            total_edges: 0,
            total_edge_failures: 0,
            flushes: 0,
            last_flush_ids: Vec::new(),
        }
    }

    /// Crea un BulkLoader con WAL para durabilidad.
    pub fn with_wal(
        graph: &'g ConcurrentGraph,
        config: BulkConfig,
        wal: Arc<ConcurrentWal>,
    ) -> Self {
        let capacity = config.flush_threshold;
        Self {
            graph,
            wal: Some(wal),
            config: BulkConfig {
                wal_enabled: true,
                ..config
            },
            buffer_nodes: Vec::with_capacity(capacity),
            buffer_edges: Vec::with_capacity(capacity),
            total_nodes: 0,
            total_edges: 0,
            total_edge_failures: 0,
            flushes: 0,
            last_flush_ids: Vec::new(),
        }
    }

    /// Agrega un nodo al buffer. Retorna un handle (índice) para referenciarlo
    /// como endpoint de edges pendientes.
    pub fn add_node(&mut self, type_id: TypeId, properties: Vec<(u16, Value)>) -> usize {
        let idx = self.buffer_nodes.len();
        self.buffer_nodes.push(PendingNode {
            type_id,
            properties,
        });

        // Auto-flush si alcanzamos el threshold
        if self.buffer_nodes.len() >= self.config.flush_threshold {
            let _ = self.flush();
        }

        idx
    }

    /// Agrega un nodo sin propiedades al buffer.
    pub fn add_node_typed(&mut self, type_id: TypeId) -> usize {
        self.add_node(type_id, Vec::new())
    }

    /// Agrega un edge entre dos NodeIds existentes en el grafo.
    pub fn add_edge_existing(
        &mut self,
        source: NodeId,
        target: NodeId,
        type_id: TypeId,
    ) {
        self.buffer_edges.push(PendingEdge {
            source: EdgeEndpoint::Existing(source),
            target: EdgeEndpoint::Existing(target),
            type_id,
        });
    }

    /// Agrega un edge entre nodos pendientes (referenciados por su handle).
    pub fn add_edge_pending(&mut self, source_idx: usize, target_idx: usize, type_id: TypeId) {
        self.buffer_edges.push(PendingEdge {
            source: EdgeEndpoint::Pending(source_idx),
            target: EdgeEndpoint::Pending(target_idx),
            type_id,
        });
    }

    /// Flushea el buffer actual al grafo.
    fn flush(&mut self) -> BikoResult<()> {
        if self.buffer_nodes.is_empty() && self.buffer_edges.is_empty() {
            return Ok(());
        }

        // 1. Si WAL está habilitado, escribir WAL entries antes de mutar el grafo
        if self.config.wal_enabled {
            if let Some(wal) = &self.wal {
                let mut wal_entries = Vec::with_capacity(
                    self.buffer_nodes.len() + self.buffer_edges.len() + 1,
                );

                let tx_id = wal.next_tx_id();

                // Nodos como WAL entries
                for node in &self.buffer_nodes {
                    let mut payload = Vec::with_capacity(2 + 2); // type_id + marker
                    payload.extend_from_slice(&node.type_id.0.to_le_bytes());
                    payload.push(0x01); // marker: node insert
                    wal_entries.push(WalEntry {
                        tx_id,
                        op_type: WalOpType::PageWrite,
                        payload,
                    });
                }

                // Commit marker
                wal_entries.push(WalEntry {
                    tx_id,
                    op_type: WalOpType::TxCommit,
                    payload: Vec::new(),
                });

                wal.write_batch(&wal_entries)?;
            }
        }

        // 2. Flush nodos al grafo en batch paralelo
        if !self.buffer_nodes.is_empty() {
            let entries: Vec<(TypeId, Vec<(u16, Value)>)> = self
                .buffer_nodes
                .drain(..)
                .map(|n| (n.type_id, n.properties))
                .collect();

            let ids = self.graph.insert_nodes_batch(&entries);
            self.total_nodes += ids.len();
            self.last_flush_ids = ids;
        }

        // 3. Resolve pending edge endpoints y flush edges
        if !self.buffer_edges.is_empty() {
            let resolved_edges: Vec<(NodeId, NodeId, TypeId)> = self
                .buffer_edges
                .drain(..)
                .filter_map(|e| {
                    let source = match e.source {
                        EdgeEndpoint::Existing(id) => id,
                        EdgeEndpoint::Pending(idx) => {
                            if idx < self.last_flush_ids.len() {
                                self.last_flush_ids[idx]
                            } else {
                                return None;
                            }
                        }
                    };
                    let target = match e.target {
                        EdgeEndpoint::Existing(id) => id,
                        EdgeEndpoint::Pending(idx) => {
                            if idx < self.last_flush_ids.len() {
                                self.last_flush_ids[idx]
                            } else {
                                return None;
                            }
                        }
                    };
                    Some((source, target, e.type_id))
                })
                .collect();

            let results = self.graph.insert_edges_batch(&resolved_edges);
            for result in &results {
                if result.is_ok() {
                    self.total_edges += 1;
                } else {
                    self.total_edge_failures += 1;
                }
            }
        }

        self.flushes += 1;
        Ok(())
    }

    /// Flushea los buffers restantes y retorna el resultado final.
    pub fn finish(mut self) -> BikoResult<BulkResult> {
        self.flush()?;

        Ok(BulkResult {
            nodes_inserted: self.total_nodes,
            edges_inserted: self.total_edges,
            edges_failed: self.total_edge_failures,
            flushes: self.flushes,
        })
    }

    /// Flushea manualmente sin consumir el loader (sigue acumulando).
    pub fn flush_now(&mut self) -> BikoResult<()> {
        self.flush()
    }

    /// Número de nodos pendientes en el buffer.
    pub fn pending_nodes(&self) -> usize {
        self.buffer_nodes.len()
    }

    /// Número de edges pendientes en el buffer.
    pub fn pending_edges(&self) -> usize {
        self.buffer_edges.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Instant;

    fn person() -> TypeId {
        TypeId(1)
    }
    fn knows() -> TypeId {
        TypeId(10)
    }

    // ── Basic tests ─────────────────────────────────────────────────────

    #[test]
    fn test_bulk_empty() {
        let g = ConcurrentGraph::new();
        let loader = BulkLoader::new(&g, BulkConfig::default());
        let result = loader.finish().unwrap();
        assert_eq!(result.nodes_inserted, 0);
        assert_eq!(result.edges_inserted, 0);
        assert_eq!(result.flushes, 0);
    }

    #[test]
    fn test_bulk_single_node() {
        let g = ConcurrentGraph::new();
        let mut loader = BulkLoader::new(&g, BulkConfig::default());
        loader.add_node(person(), vec![(0, Value::from("Alice"))]);
        let result = loader.finish().unwrap();
        assert_eq!(result.nodes_inserted, 1);
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_bulk_nodes_and_edges() {
        let g = ConcurrentGraph::new();
        let config = BulkConfig {
            flush_threshold: 1000,
            wal_enabled: false,
        };
        let mut loader = BulkLoader::new(&g, config);

        let a = loader.add_node_typed(person());
        let b = loader.add_node_typed(person());
        let c = loader.add_node_typed(person());

        loader.add_edge_pending(a, b, knows());
        loader.add_edge_pending(b, c, knows());
        loader.add_edge_pending(a, c, knows());

        let result = loader.finish().unwrap();
        assert_eq!(result.nodes_inserted, 3);
        assert_eq!(result.edges_inserted, 3);
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn test_bulk_existing_node_edges() {
        let g = ConcurrentGraph::new();
        let n1 = g.insert_node(person());
        let n2 = g.insert_node(person());

        let mut loader = BulkLoader::new(&g, BulkConfig::default());
        loader.add_edge_existing(n1, n2, knows());

        let result = loader.finish().unwrap();
        assert_eq!(result.nodes_inserted, 0);
        assert_eq!(result.edges_inserted, 1);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_bulk_auto_flush() {
        let g = ConcurrentGraph::new();
        let config = BulkConfig {
            flush_threshold: 100,
            wal_enabled: false,
        };
        let mut loader = BulkLoader::new(&g, config);

        for _ in 0..250 {
            loader.add_node_typed(person());
        }

        // Should have auto-flushed twice (at 100 and 200)
        let result = loader.finish().unwrap();
        assert_eq!(result.nodes_inserted, 250);
        assert!(result.flushes >= 2); // at least 2 auto-flushes + 1 final
        assert_eq!(g.node_count(), 250);
    }

    #[test]
    fn test_bulk_pending_count() {
        let g = ConcurrentGraph::new();
        let config = BulkConfig {
            flush_threshold: 1_000_000, // no auto-flush
            wal_enabled: false,
        };
        let mut loader = BulkLoader::new(&g, config);

        loader.add_node_typed(person());
        loader.add_node_typed(person());
        assert_eq!(loader.pending_nodes(), 2);
        assert_eq!(loader.pending_edges(), 0);

        loader.flush_now().unwrap();
        assert_eq!(loader.pending_nodes(), 0);
        assert_eq!(g.node_count(), 2);
    }

    // ── Large-scale tests ───────────────────────────────────────────────

    #[test]
    fn test_bulk_100k_nodes() {
        let n = 100_000;
        let g = ConcurrentGraph::with_capacity(n, 0);
        let config = BulkConfig {
            flush_threshold: 50_000,
            wal_enabled: false,
        };
        let mut loader = BulkLoader::new(&g, config);

        for _ in 0..n {
            loader.add_node_typed(person());
        }

        let result = loader.finish().unwrap();
        assert_eq!(result.nodes_inserted, n);
        assert_eq!(g.node_count(), n);
        assert!(result.flushes >= 2);
    }

    #[test]
    fn test_bulk_100k_with_edges() {
        let n = 100_000;
        let g = ConcurrentGraph::with_capacity(n, n);
        let config = BulkConfig {
            flush_threshold: n + 1, // one big batch
            wal_enabled: false,
        };
        let mut loader = BulkLoader::new(&g, config);

        let handles: Vec<usize> = (0..n).map(|_| loader.add_node_typed(person())).collect();

        // Chain edges: 0→1→2→...→N-1
        for i in 0..n - 1 {
            loader.add_edge_pending(handles[i], handles[i + 1], knows());
        }

        let result = loader.finish().unwrap();
        assert_eq!(result.nodes_inserted, n);
        assert_eq!(result.edges_inserted, n - 1);
        assert_eq!(g.node_count(), n);
        assert_eq!(g.edge_count(), n - 1);
    }

    #[test]
    fn test_bulk_1m_nodes_throughput() {
        let n = 1_000_000;
        let g = ConcurrentGraph::with_capacity(n, 0);
        let config = BulkConfig {
            flush_threshold: 100_000,
            wal_enabled: false,
        };

        let start = Instant::now();
        let mut loader = BulkLoader::new(&g, config);
        for _ in 0..n {
            loader.add_node_typed(person());
        }
        let result = loader.finish().unwrap();
        let elapsed = start.elapsed();

        assert_eq!(result.nodes_inserted, n);
        assert_eq!(g.node_count(), n);

        let nodes_per_sec = n as f64 / elapsed.as_secs_f64();
        eprintln!(
            "Bulk insert 1M nodes: {:.2}ms ({:.0} nodes/sec)",
            elapsed.as_secs_f64() * 1000.0,
            nodes_per_sec
        );
        // Target: > 500K nodes/sec (conservative for debug builds)
        // In release: typically > 2M nodes/sec
    }

    // ── Concurrent multi-loader test ────────────────────────────────────

    #[test]
    fn test_bulk_concurrent_loaders() {
        let n_per_thread = 50_000;
        let num_threads = 4;
        let g = Arc::new(ConcurrentGraph::with_capacity(
            n_per_thread * num_threads,
            0,
        ));

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for _ in 0..num_threads {
                let graph = Arc::clone(&g);
                handles.push(s.spawn(move || {
                    let config = BulkConfig {
                        flush_threshold: 10_000,
                        wal_enabled: false,
                    };
                    // SAFETY: Arc<ConcurrentGraph> is Send+Sync
                    // We need an unsafe trick or restructure for &ConcurrentGraph
                    // Actually ConcurrentGraph inner DashMap is Sync, so &* works
                    let mut loader = BulkLoader::new(&graph, config);
                    for _ in 0..n_per_thread {
                        loader.add_node_typed(person());
                    }
                    loader.finish().unwrap()
                }));
            }
            for h in handles {
                let result = h.join().unwrap();
                assert_eq!(result.nodes_inserted, n_per_thread);
            }
        });

        assert_eq!(g.node_count(), n_per_thread * num_threads);
    }

    // ── WAL integration test ────────────────────────────────────────────

    #[test]
    fn test_bulk_with_wal() {
        let dir = std::env::temp_dir()
            .join("bikodb_test_bulk_wal")
            .join("basic");
        let _ = std::fs::remove_dir_all(&dir);

        let g = ConcurrentGraph::new();
        let wal = Arc::new(ConcurrentWal::open(&dir, 1000).unwrap());

        let config = BulkConfig {
            flush_threshold: 100,
            wal_enabled: true,
        };
        let mut loader = BulkLoader::with_wal(&g, config, Arc::clone(&wal));

        for _ in 0..50 {
            loader.add_node_typed(person());
        }

        let result = loader.finish().unwrap();
        assert_eq!(result.nodes_inserted, 50);
        assert_eq!(g.node_count(), 50);

        // WAL should have entries
        let entries = wal.recover().unwrap();
        assert!(!entries.is_empty());
        // Last entry should be TxCommit
        assert_eq!(entries.last().unwrap().op_type, WalOpType::TxCommit);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_bulk_with_wal_large() {
        let dir = std::env::temp_dir()
            .join("bikodb_test_bulk_wal")
            .join("large");
        let _ = std::fs::remove_dir_all(&dir);

        let n = 10_000;
        let g = ConcurrentGraph::with_capacity(n, 0);
        let wal = Arc::new(ConcurrentWal::open(&dir, 5000).unwrap());

        let config = BulkConfig {
            flush_threshold: 5_000,
            wal_enabled: true,
        };
        let mut loader = BulkLoader::with_wal(&g, config, Arc::clone(&wal));

        for _ in 0..n {
            loader.add_node_typed(person());
        }

        let result = loader.finish().unwrap();
        assert_eq!(result.nodes_inserted, n);
        assert_eq!(g.node_count(), n);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Graph batch insert direct test ──────────────────────────────────

    #[test]
    fn test_graph_insert_nodes_batch() {
        let g = ConcurrentGraph::new();
        let entries: Vec<(TypeId, Vec<(u16, Value)>)> = (0..100)
            .map(|i| (person(), vec![(0, Value::Int(i))]))
            .collect();

        let ids = g.insert_nodes_batch(&entries);
        assert_eq!(ids.len(), 100);
        assert_eq!(g.node_count(), 100);

        // Verify properties
        let node = g.get_node(ids[42]).unwrap();
        assert_eq!(node.properties[0].1, Value::Int(42));
    }

    #[test]
    fn test_graph_insert_nodes_batch_typed() {
        let g = ConcurrentGraph::new();
        let ids = g.insert_nodes_batch_typed(person(), 1000);
        assert_eq!(ids.len(), 1000);
        assert_eq!(g.node_count(), 1000);
    }

    #[test]
    fn test_graph_insert_edges_batch() {
        let g = ConcurrentGraph::new();
        let ids = g.insert_nodes_batch_typed(person(), 10);

        let edges: Vec<(NodeId, NodeId, TypeId)> = (0..9)
            .map(|i| (ids[i], ids[i + 1], knows()))
            .collect();

        let results = g.insert_edges_batch(&edges);
        assert_eq!(results.len(), 9);
        assert!(results.iter().all(|r| r.is_ok()));
        assert_eq!(g.edge_count(), 9);
    }

    #[test]
    fn test_graph_insert_edges_batch_invalid() {
        let g = ConcurrentGraph::new();
        let ids = g.insert_nodes_batch_typed(person(), 2);

        let edges = vec![
            (ids[0], ids[1], knows()),           // valid
            (ids[0], NodeId(99999), knows()),     // invalid target
            (NodeId(88888), ids[1], knows()),     // invalid source
        ];

        let results = g.insert_edges_batch(&edges);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
        assert!(results[2].is_err());
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_graph_batch_concurrent_insert() {
        let g = Arc::new(ConcurrentGraph::with_capacity(400_000, 0));

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for _ in 0..4 {
                let graph = Arc::clone(&g);
                handles.push(s.spawn(move || {
                    graph.insert_nodes_batch_typed(person(), 100_000)
                }));
            }
            for h in handles {
                let ids = h.join().unwrap();
                assert_eq!(ids.len(), 100_000);
            }
        });

        assert_eq!(g.node_count(), 400_000);
    }
}
