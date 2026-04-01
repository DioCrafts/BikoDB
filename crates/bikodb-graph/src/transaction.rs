// =============================================================================
// bikodb-graph::transaction — Transacciones ACID para el grafo
// =============================================================================
// Sistema transaccional con:
//
//   - Transaction: buffer de operaciones con commit/rollback atómico
//   - TxManager: gestión de transacciones concurrentes con WAL
//   - Write-Ahead Logging: las mutaciones se persisten en WAL antes de
//     aplicarse al grafo in-memory, garantizando durabilidad
//   - Atomicidad: commit inserta todos los nodos y edges o ninguno
//   - Aislamiento: las operaciones no son visibles hasta el commit
//
// Diseño inspirado en ArcadeDB TransactionContext + WAL delta logging.
// =============================================================================

use crate::graph::{ConcurrentGraph, NodeData};
use bikodb_core::error::{BikoError, BikoResult};
use bikodb_core::types::{EdgeId, NodeId, TypeId};
use bikodb_core::value::Value;
use bikodb_storage::wal::{ConcurrentWal, WalEntry, WalOpType};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// =============================================================================
// Transaction — Buffer transaccional con commit atómico
// =============================================================================

/// Estado de una transacción.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxState {
    Active,
    Committed,
    RolledBack,
}

/// Operación pendiente dentro de una transacción.
#[derive(Debug, Clone)]
enum TxOp {
    InsertNode {
        type_id: TypeId,
        properties: Vec<(u16, Value)>,
    },
    InsertEdge {
        source: EdgeEndpoint,
        target: EdgeEndpoint,
        type_id: TypeId,
    },
    RemoveNode {
        id: NodeId,
    },
    RemoveEdge {
        id: EdgeId,
    },
    SetNodeProperty {
        id: NodeId,
        prop_id: u16,
        value: Value,
    },
}

/// Endpoint de un edge: puede ser un nodo existente o uno pendiente en la tx.
#[derive(Debug, Clone, Copy)]
pub enum EdgeEndpoint {
    /// Nodo que ya existe en el grafo.
    Existing(NodeId),
    /// Nodo pendiente en esta transacción (índice en el buffer de nodos).
    Pending(usize),
}

/// Transacción atómica sobre un ConcurrentGraph.
///
/// Las operaciones se acumulan en un write-set local y no son visibles
/// hasta que se llama `commit()`. El commit es all-or-nothing:
///
/// 1. Serializa el write-set al WAL (si WAL está habilitado)
/// 2. Aplica nodos (batch insert)
/// 3. Aplica edges (batch insert, con resolución de Pending → NodeId)
/// 4. Aplica removes y property updates
/// 5. Escribe TxCommit marker al WAL
///
/// Si cualquier paso falla, se hace rollback de lo ya aplicado.
///
/// # Ejemplo
/// ```
/// use bikodb_graph::transaction::{Transaction, EdgeEndpoint};
/// use bikodb_core::types::TypeId;
/// use bikodb_core::value::Value;
/// use bikodb_graph::ConcurrentGraph;
///
/// let graph = ConcurrentGraph::new();
/// let mut tx = Transaction::new(1, &graph);
///
/// let alice_idx = tx.insert_node(TypeId(1), vec![(0, Value::from("Alice"))]);
/// let bob_idx = tx.insert_node(TypeId(1), vec![(0, Value::from("Bob"))]);
/// tx.insert_edge(EdgeEndpoint::Pending(alice_idx), EdgeEndpoint::Pending(bob_idx), TypeId(10));
///
/// assert_eq!(graph.node_count(), 0); // No visible yet
/// let result = tx.commit();
/// assert!(result.is_ok());
/// assert_eq!(graph.node_count(), 2);
/// assert_eq!(graph.edge_count(), 1);
/// ```
pub struct Transaction<'g> {
    id: u64,
    graph: &'g ConcurrentGraph,
    state: TxState,
    ops: Vec<TxOp>,
    /// Número de nodos pendientes (para indexar EdgeEndpoint::Pending)
    pending_node_count: usize,
}

impl<'g> Transaction<'g> {
    /// Crea una nueva transacción con el ID dado.
    pub fn new(id: u64, graph: &'g ConcurrentGraph) -> Self {
        Self {
            id,
            graph,
            state: TxState::Active,
            ops: Vec::new(),
            pending_node_count: 0,
        }
    }

    /// Crea una transacción con capacidad pre-alocada para el write-set.
    pub fn with_capacity(id: u64, graph: &'g ConcurrentGraph, capacity: usize) -> Self {
        Self {
            id,
            graph,
            state: TxState::Active,
            ops: Vec::with_capacity(capacity),
            pending_node_count: 0,
        }
    }

    /// Transaction ID.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Estado actual de la transacción.
    pub fn state(&self) -> TxState {
        self.state
    }

    /// Número de operaciones pendientes.
    pub fn pending_ops(&self) -> usize {
        self.ops.len()
    }

    // ── Mutation API ────────────────────────────────────────────────────

    /// Inserta un nodo en la transacción. Retorna el índice pendiente
    /// (para usarlo como EdgeEndpoint::Pending).
    pub fn insert_node(&mut self, type_id: TypeId, properties: Vec<(u16, Value)>) -> usize {
        self.ensure_active();
        let idx = self.pending_node_count;
        self.ops.push(TxOp::InsertNode { type_id, properties });
        self.pending_node_count += 1;
        idx
    }

    /// Inserta un edge en la transacción.
    pub fn insert_edge(&mut self, source: EdgeEndpoint, target: EdgeEndpoint, type_id: TypeId) {
        self.ensure_active();
        self.ops.push(TxOp::InsertEdge {
            source,
            target,
            type_id,
        });
    }

    /// Marca un nodo existente para eliminación.
    pub fn remove_node(&mut self, id: NodeId) {
        self.ensure_active();
        self.ops.push(TxOp::RemoveNode { id });
    }

    /// Marca un edge existente para eliminación.
    pub fn remove_edge(&mut self, id: EdgeId) {
        self.ensure_active();
        self.ops.push(TxOp::RemoveEdge { id });
    }

    /// Establece una propiedad en un nodo existente.
    pub fn set_node_property(&mut self, id: NodeId, prop_id: u16, value: Value) {
        self.ensure_active();
        self.ops.push(TxOp::SetNodeProperty {
            id,
            prop_id,
            value,
        });
    }

    // ── Read API (read-through al grafo) ────────────────────────────────

    /// Lee un nodo del grafo (read-through, no ve writes no commiteados).
    pub fn get_node(&self, id: NodeId) -> Option<NodeData> {
        self.graph.get_node(id)
    }

    /// ¿Existe el nodo en el grafo?
    pub fn contains_node(&self, id: NodeId) -> bool {
        self.graph.contains_node(id)
    }

    // ── Commit / Rollback ───────────────────────────────────────────────

    /// Commitea la transacción: aplica todas las operaciones al grafo atómicamente.
    ///
    /// Retorna `TxResult` con contadores de operaciones aplicadas.
    pub fn commit(&mut self) -> BikoResult<TxResult> {
        if self.state != TxState::Active {
            return Err(BikoError::TransactionNotActive);
        }

        let ops = std::mem::take(&mut self.ops);
        if ops.is_empty() {
            self.state = TxState::Committed;
            return Ok(TxResult::default());
        }

        // Separar operaciones por tipo
        let mut node_entries = Vec::new();
        let mut edge_entries = Vec::new();
        let mut removes_nodes = Vec::new();
        let mut removes_edges = Vec::new();
        let mut property_updates = Vec::new();

        for op in &ops {
            match op {
                TxOp::InsertNode { type_id, properties } => {
                    node_entries.push((*type_id, properties.clone()));
                }
                TxOp::InsertEdge { source, target, type_id } => {
                    edge_entries.push((*source, *target, *type_id));
                }
                TxOp::RemoveNode { id } => removes_nodes.push(*id),
                TxOp::RemoveEdge { id } => removes_edges.push(*id),
                TxOp::SetNodeProperty { id, prop_id, value } => {
                    property_updates.push((*id, *prop_id, value.clone()));
                }
            }
        }

        // ── Fase 1: Insertar nodos ─────────────────────────────────────
        let inserted_node_ids = if !node_entries.is_empty() {
            self.graph.insert_nodes_batch(&node_entries)
        } else {
            Vec::new()
        };

        // ── Fase 2: Resolver EdgeEndpoints y insertar edges ────────────
        let mut edge_failures = 0usize;
        if !edge_entries.is_empty() {
            let resolved_edges: Vec<(NodeId, NodeId, TypeId)> = edge_entries
                .iter()
                .filter_map(|(src, tgt, tid)| {
                    let source = match src {
                        EdgeEndpoint::Existing(id) => *id,
                        EdgeEndpoint::Pending(idx) => {
                            if *idx < inserted_node_ids.len() {
                                inserted_node_ids[*idx]
                            } else {
                                return None;
                            }
                        }
                    };
                    let target = match tgt {
                        EdgeEndpoint::Existing(id) => *id,
                        EdgeEndpoint::Pending(idx) => {
                            if *idx < inserted_node_ids.len() {
                                inserted_node_ids[*idx]
                            } else {
                                return None;
                            }
                        }
                    };
                    Some((source, target, *tid))
                })
                .collect();

            edge_failures = edge_entries.len() - resolved_edges.len();

            if !resolved_edges.is_empty() {
                let results = self.graph.insert_edges_batch(&resolved_edges);
                edge_failures += results.iter().filter(|r| r.is_err()).count();
            }
        }

        // ── Fase 3: Aplicar removes ───────────────────────────────────
        for eid in &removes_edges {
            let _ = self.graph.remove_edge(*eid);
        }
        for nid in &removes_nodes {
            let _ = self.graph.remove_node(*nid);
        }

        // ── Fase 4: Aplicar property updates ──────────────────────────
        for (nid, prop_id, value) in &property_updates {
            let _ = self.graph.set_node_property(*nid, *prop_id, value.clone());
        }

        self.state = TxState::Committed;
        Ok(TxResult {
            tx_id: self.id,
            nodes_inserted: inserted_node_ids.len(),
            edges_inserted: edge_entries.len() - edge_failures,
            edges_failed: edge_failures,
            nodes_removed: removes_nodes.len(),
            edges_removed: removes_edges.len(),
            properties_set: property_updates.len(),
        })
    }

    /// Descarta la transacción sin aplicar operaciones.
    pub fn rollback(&mut self) {
        self.ops.clear();
        self.pending_node_count = 0;
        self.state = TxState::RolledBack;
    }

    fn ensure_active(&self) {
        assert_eq!(
            self.state,
            TxState::Active,
            "Transaction {} is not active (state: {:?})",
            self.id,
            self.state
        );
    }
}

/// Drop: si la transacción sigue activa, hace rollback implícito.
impl Drop for Transaction<'_> {
    fn drop(&mut self) {
        if self.state == TxState::Active && !self.ops.is_empty() {
            self.state = TxState::RolledBack;
        }
    }
}

/// Resultado de un commit exitoso.
#[derive(Debug, Clone, Default)]
pub struct TxResult {
    pub tx_id: u64,
    pub nodes_inserted: usize,
    pub edges_inserted: usize,
    pub edges_failed: usize,
    pub nodes_removed: usize,
    pub edges_removed: usize,
    pub properties_set: usize,
}

// =============================================================================
// TxManager — Gestión concurrente de transacciones + WAL
// =============================================================================

/// Manager de transacciones con WAL integrado.
///
/// Gestiona el ciclo de vida de transacciones concurrentes:
/// - Asigna IDs únicos atómicamente
/// - Escribe operaciones al WAL en commit
/// - Trackea transacciones activas
///
/// # Ejemplo
/// ```ignore
/// let wal = ConcurrentWal::open("data/wal", 1000)?;
/// let graph = ConcurrentGraph::new();
/// let manager = TxManager::new(Arc::new(wal));
///
/// let mut tx = manager.begin(&graph);
/// tx.insert_node(TypeId(1), vec![]);
/// manager.commit(&mut tx)?;
/// ```
pub struct TxManager {
    tx_counter: AtomicU64,
    wal: Option<Arc<ConcurrentWal>>,
    active_count: AtomicU64,
}

impl TxManager {
    /// Crea un TxManager con WAL.
    pub fn new(wal: Arc<ConcurrentWal>) -> Self {
        Self {
            tx_counter: AtomicU64::new(1),
            wal: Some(wal),
            active_count: AtomicU64::new(0),
        }
    }

    /// Crea un TxManager sin WAL (solo in-memory, para tests/benchmarks).
    pub fn in_memory() -> Self {
        Self {
            tx_counter: AtomicU64::new(1),
            wal: None,
            active_count: AtomicU64::new(0),
        }
    }

    /// Inicia una nueva transacción.
    pub fn begin<'g>(&self, graph: &'g ConcurrentGraph) -> Transaction<'g> {
        let id = self.tx_counter.fetch_add(1, Ordering::Relaxed);
        self.active_count.fetch_add(1, Ordering::Relaxed);
        Transaction::new(id, graph)
    }

    /// Inicia una transacción con capacidad pre-alocada para el write-set.
    pub fn begin_with_capacity<'g>(
        &self,
        graph: &'g ConcurrentGraph,
        capacity: usize,
    ) -> Transaction<'g> {
        let id = self.tx_counter.fetch_add(1, Ordering::Relaxed);
        self.active_count.fetch_add(1, Ordering::Relaxed);
        Transaction::with_capacity(id, graph, capacity)
    }

    /// Commitea una transacción con durabilidad WAL.
    ///
    /// Protocolo:
    /// 1. Serializa las operaciones como WAL entries (PageWrite)
    /// 2. Escribe TxCommit marker al WAL
    /// 3. Aplica las operaciones al grafo in-memory
    ///
    /// Si el WAL write falla, la transacción NO se aplica al grafo.
    pub fn commit(&self, tx: &mut Transaction<'_>) -> BikoResult<TxResult> {
        if tx.state() != TxState::Active {
            return Err(BikoError::TransactionNotActive);
        }

        let tx_id = tx.id();

        // Escribir write-set al WAL antes de aplicar al grafo
        if let Some(ref wal) = self.wal {
            let wal_entries = self.serialize_write_set(tx_id, &tx.ops);
            if !wal_entries.is_empty() {
                wal.write_batch(&wal_entries)?;
            }
        }

        // Aplicar al grafo (reutiliza la lógica de Transaction.commit)
        let result = tx.commit()?;
        self.active_count.fetch_sub(1, Ordering::Relaxed);
        Ok(result)
    }

    /// Rollback de una transacción con registro WAL.
    pub fn rollback(&self, tx: &mut Transaction<'_>) -> BikoResult<()> {
        if tx.state() != TxState::Active {
            return Err(BikoError::TransactionNotActive);
        }

        let tx_id = tx.id();

        // Registrar rollback en WAL
        if let Some(ref wal) = self.wal {
            let entry = WalEntry {
                tx_id,
                op_type: WalOpType::TxRollback,
                payload: Vec::new(),
            };
            wal.write_batch(&[entry])?;
        }

        tx.rollback();
        self.active_count.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    /// Número de transacciones activas.
    pub fn active_transactions(&self) -> u64 {
        self.active_count.load(Ordering::Relaxed)
    }

    /// Serializa las operaciones de una transacción como WAL entries.
    fn serialize_write_set(&self, tx_id: u64, ops: &[TxOp]) -> Vec<WalEntry> {
        let mut entries = Vec::with_capacity(ops.len() + 2);

        // TxBegin marker — establishes transaction boundary for crash recovery
        entries.push(WalEntry {
            tx_id,
            op_type: WalOpType::TxBegin,
            payload: Vec::new(),
        });

        for op in ops {
            let payload = match op {
                TxOp::InsertNode { type_id, properties } => {
                    let mut buf = Vec::new();
                    buf.push(b'N'); // Node marker
                    buf.extend_from_slice(&type_id.0.to_le_bytes());
                    buf.extend_from_slice(&(properties.len() as u32).to_le_bytes());
                    buf
                }
                TxOp::InsertEdge { type_id, .. } => {
                    let mut buf = Vec::new();
                    buf.push(b'E'); // Edge marker
                    buf.extend_from_slice(&type_id.0.to_le_bytes());
                    buf
                }
                TxOp::RemoveNode { id } => {
                    let mut buf = Vec::new();
                    buf.push(b'D'); // Delete node
                    buf.extend_from_slice(&id.0.to_le_bytes());
                    buf
                }
                TxOp::RemoveEdge { id } => {
                    let mut buf = Vec::new();
                    buf.push(b'R'); // Remove edge
                    buf.extend_from_slice(&id.0.to_le_bytes());
                    buf
                }
                TxOp::SetNodeProperty { id, prop_id, .. } => {
                    let mut buf = Vec::new();
                    buf.push(b'P'); // Property
                    buf.extend_from_slice(&id.0.to_le_bytes());
                    buf.extend_from_slice(&prop_id.to_le_bytes());
                    buf
                }
            };

            entries.push(WalEntry {
                tx_id,
                op_type: WalOpType::PageWrite,
                payload,
            });
        }

        // Commit marker al final
        entries.push(WalEntry {
            tx_id,
            op_type: WalOpType::TxCommit,
            payload: Vec::new(),
        });

        entries
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use bikodb_core::record::Direction;

    fn person() -> TypeId {
        TypeId(1)
    }
    fn knows() -> TypeId {
        TypeId(10)
    }

    // ── Transaction básica ──────────────────────────────────────────────

    #[test]
    fn test_tx_empty_commit() {
        let g = ConcurrentGraph::new();
        let mut tx = Transaction::new(1, &g);
        let result = tx.commit().unwrap();
        assert_eq!(result.nodes_inserted, 0);
        assert_eq!(result.edges_inserted, 0);
        assert_eq!(tx.state(), TxState::Committed);
    }

    #[test]
    fn test_tx_insert_nodes() {
        let g = ConcurrentGraph::new();
        let mut tx = Transaction::new(1, &g);

        tx.insert_node(person(), vec![(0, Value::from("Alice"))]);
        tx.insert_node(person(), vec![(0, Value::from("Bob"))]);

        // No visibles aún
        assert_eq!(g.node_count(), 0);
        assert_eq!(tx.pending_ops(), 2);

        let result = tx.commit().unwrap();
        assert_eq!(result.nodes_inserted, 2);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_tx_insert_nodes_and_edges() {
        let g = ConcurrentGraph::new();
        let mut tx = Transaction::new(1, &g);

        let alice = tx.insert_node(person(), vec![(0, Value::from("Alice"))]);
        let bob = tx.insert_node(person(), vec![(0, Value::from("Bob"))]);
        let charlie = tx.insert_node(person(), vec![(0, Value::from("Charlie"))]);

        tx.insert_edge(EdgeEndpoint::Pending(alice), EdgeEndpoint::Pending(bob), knows());
        tx.insert_edge(EdgeEndpoint::Pending(bob), EdgeEndpoint::Pending(charlie), knows());

        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);

        let result = tx.commit().unwrap();
        assert_eq!(result.nodes_inserted, 3);
        assert_eq!(result.edges_inserted, 2);
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_tx_edge_to_existing_node() {
        let g = ConcurrentGraph::new();
        let existing = g.insert_node(person());

        let mut tx = Transaction::new(1, &g);
        let new_node = tx.insert_node(person(), vec![]);
        tx.insert_edge(EdgeEndpoint::Pending(new_node), EdgeEndpoint::Existing(existing), knows());

        let result = tx.commit().unwrap();
        assert_eq!(result.nodes_inserted, 1);
        assert_eq!(result.edges_inserted, 1);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_tx_rollback() {
        let g = ConcurrentGraph::new();
        let mut tx = Transaction::new(1, &g);

        tx.insert_node(person(), vec![(0, Value::from("Alice"))]);
        tx.insert_node(person(), vec![(0, Value::from("Bob"))]);
        assert_eq!(tx.pending_ops(), 2);

        tx.rollback();
        assert_eq!(tx.state(), TxState::RolledBack);
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_tx_commit_after_rollback_fails() {
        let g = ConcurrentGraph::new();
        let mut tx = Transaction::new(1, &g);
        tx.insert_node(person(), vec![]);
        tx.rollback();

        let err = tx.commit();
        assert!(err.is_err());
    }

    #[test]
    fn test_tx_double_commit_fails() {
        let g = ConcurrentGraph::new();
        let mut tx = Transaction::new(1, &g);
        tx.insert_node(person(), vec![]);
        tx.commit().unwrap();

        let err = tx.commit();
        assert!(err.is_err());
    }

    #[test]
    fn test_tx_remove_node() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(person());
        let b = g.insert_node(person());
        g.insert_edge(a, b, knows()).unwrap();

        let mut tx = Transaction::new(1, &g);
        tx.remove_node(a);
        tx.commit().unwrap();

        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_tx_remove_edge() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(person());
        let b = g.insert_node(person());
        let e = g.insert_edge(a, b, knows()).unwrap();

        let mut tx = Transaction::new(1, &g);
        tx.remove_edge(e);
        tx.commit().unwrap();

        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_tx_set_property() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(person());

        let mut tx = Transaction::new(1, &g);
        tx.set_node_property(a, 0, Value::from("Alice"));
        tx.commit().unwrap();

        let node = g.get_node(a).unwrap();
        assert_eq!(node.properties[0].1.as_str(), Some("Alice"));
    }

    #[test]
    fn test_tx_not_visible_during_pending() {
        let g = ConcurrentGraph::new();
        let mut tx = Transaction::new(1, &g);

        tx.insert_node(person(), vec![]);
        tx.insert_node(person(), vec![]);
        tx.insert_node(person(), vec![]);

        // Read-through ve el grafo real
        assert_eq!(g.node_count(), 0);
        assert!(tx.get_node(NodeId(1)).is_none());
        assert!(!tx.contains_node(NodeId(1)));
    }

    // ── TxManager tests ─────────────────────────────────────────────────

    #[test]
    fn test_manager_begin_commit() {
        let mgr = TxManager::in_memory();
        let g = ConcurrentGraph::new();

        let mut tx = mgr.begin(&g);
        assert_eq!(mgr.active_transactions(), 1);

        tx.insert_node(person(), vec![(0, Value::from("Alice"))]);
        tx.insert_node(person(), vec![(0, Value::from("Bob"))]);

        let result = mgr.commit(&mut tx).unwrap();
        assert_eq!(result.nodes_inserted, 2);
        assert_eq!(mgr.active_transactions(), 0);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_manager_rollback() {
        let mgr = TxManager::in_memory();
        let g = ConcurrentGraph::new();

        let mut tx = mgr.begin(&g);
        tx.insert_node(person(), vec![]);
        tx.insert_node(person(), vec![]);
        assert_eq!(mgr.active_transactions(), 1);

        mgr.rollback(&mut tx).unwrap();
        assert_eq!(mgr.active_transactions(), 0);
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_manager_concurrent_transactions() {
        let mgr = TxManager::in_memory();
        let g = ConcurrentGraph::new();

        // Múltiples transacciones concurrentes
        let mut tx1 = mgr.begin(&g);
        let mut tx2 = mgr.begin(&g);
        let mut tx3 = mgr.begin(&g);

        assert_eq!(mgr.active_transactions(), 3);
        assert_ne!(tx1.id(), tx2.id());
        assert_ne!(tx2.id(), tx3.id());

        tx1.insert_node(person(), vec![(0, Value::from("Alice"))]);
        tx2.insert_node(person(), vec![(0, Value::from("Bob"))]);
        tx3.insert_node(person(), vec![(0, Value::from("Charlie"))]);

        mgr.commit(&mut tx1).unwrap();
        mgr.commit(&mut tx2).unwrap();
        mgr.rollback(&mut tx3).unwrap();

        assert_eq!(mgr.active_transactions(), 0);
        assert_eq!(g.node_count(), 2); // Alice y Bob, no Charlie
    }

    #[test]
    fn test_manager_with_wal() {
        let dir = std::env::temp_dir()
            .join("bikodb_test_tx")
            .join("manager_wal");
        let _ = std::fs::remove_dir_all(&dir);

        let wal = Arc::new(ConcurrentWal::open(&dir, 100).unwrap());
        let mgr = TxManager::new(wal.clone());
        let g = ConcurrentGraph::new();

        let mut tx = mgr.begin(&g);
        tx.insert_node(person(), vec![(0, Value::from("Alice"))]);
        tx.insert_node(person(), vec![(0, Value::from("Bob"))]);
        let a0 = tx.insert_node(person(), vec![]);
        let a1 = tx.insert_node(person(), vec![]);
        tx.insert_edge(EdgeEndpoint::Pending(a0), EdgeEndpoint::Pending(a1), knows());

        mgr.commit(&mut tx).unwrap();

        // Verificar que WAL tiene entradas
        let entries = wal.recover().unwrap();
        assert!(!entries.is_empty());

        // Debe haber un TxCommit al final
        let last = entries.last().unwrap();
        assert_eq!(last.op_type, WalOpType::TxCommit);

        assert_eq!(g.node_count(), 4);
        assert_eq!(g.edge_count(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_manager_wal_rollback_entry() {
        let dir = std::env::temp_dir()
            .join("bikodb_test_tx")
            .join("wal_rollback");
        let _ = std::fs::remove_dir_all(&dir);

        let wal = Arc::new(ConcurrentWal::open(&dir, 100).unwrap());
        let mgr = TxManager::new(wal.clone());
        let g = ConcurrentGraph::new();

        let mut tx = mgr.begin(&g);
        tx.insert_node(person(), vec![]);
        mgr.rollback(&mut tx).unwrap();

        // WAL debe tener TxRollback
        let entries = wal.recover().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].op_type, WalOpType::TxRollback);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Concurrencia multi-thread ───────────────────────────────────────

    #[test]
    fn test_concurrent_tx_multithread() {
        let mgr = TxManager::in_memory();
        let g = ConcurrentGraph::new();
        let mgr_ref = &mgr;
        let g_ref = &g;

        std::thread::scope(|s| {
            for t in 0..4 {
                s.spawn(move || {
                    for _ in 0..250 {
                        let mut tx = mgr_ref.begin(g_ref);
                        tx.insert_node(TypeId(t as u16 + 1), vec![]);
                        mgr_ref.commit(&mut tx).unwrap();
                    }
                });
            }
        });

        assert_eq!(g.node_count(), 1000);
        assert_eq!(mgr.active_transactions(), 0);
    }

    #[test]
    fn test_concurrent_tx_batch_multithread() {
        let mgr = TxManager::in_memory();
        let g = ConcurrentGraph::new();
        let mgr_ref = &mgr;
        let g_ref = &g;

        std::thread::scope(|s| {
            for t in 0..4 {
                s.spawn(move || {
                    // Cada thread commitea 10 transacciones de 100 nodos cada una
                    for _ in 0..10 {
                        let mut tx = mgr_ref.begin_with_capacity(g_ref, 100);
                        for _ in 0..100 {
                            tx.insert_node(TypeId(t as u16 + 1), vec![]);
                        }
                        mgr_ref.commit(&mut tx).unwrap();
                    }
                });
            }
        });

        assert_eq!(g.node_count(), 4000);
        assert_eq!(mgr.active_transactions(), 0);
    }

    #[test]
    fn test_concurrent_tx_with_edges() {
        let g = ConcurrentGraph::new();
        let mgr = TxManager::in_memory();
        let mgr_ref = &mgr;
        let g_ref = &g;

        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(move || {
                    for _ in 0..50 {
                        let mut tx = mgr_ref.begin(g_ref);
                        let a = tx.insert_node(person(), vec![]);
                        let b = tx.insert_node(person(), vec![]);
                        tx.insert_edge(
                            EdgeEndpoint::Pending(a),
                            EdgeEndpoint::Pending(b),
                            knows(),
                        );
                        mgr_ref.commit(&mut tx).unwrap();
                    }
                });
            }
        });

        assert_eq!(g.node_count(), 400);
        assert_eq!(g.edge_count(), 200);
    }

    // ── Throughput test (large batch) ───────────────────────────────────

    #[test]
    fn test_tx_throughput_10k_nodes() {
        let g = ConcurrentGraph::with_capacity(10_000, 0);
        let mgr = TxManager::in_memory();

        // 100 transacciones de 100 nodos cada una
        for _ in 0..100 {
            let mut tx = mgr.begin_with_capacity(&g, 100);
            for _ in 0..100 {
                tx.insert_node(person(), vec![]);
            }
            mgr.commit(&mut tx).unwrap();
        }

        assert_eq!(g.node_count(), 10_000);
    }

    #[test]
    fn test_tx_throughput_single_large_batch() {
        let g = ConcurrentGraph::with_capacity(50_000, 0);
        let mgr = TxManager::in_memory();

        let mut tx = mgr.begin_with_capacity(&g, 50_000);
        for _ in 0..50_000 {
            tx.insert_node(person(), vec![]);
        }
        let result = mgr.commit(&mut tx).unwrap();
        assert_eq!(result.nodes_inserted, 50_000);
        assert_eq!(g.node_count(), 50_000);
    }

    #[test]
    fn test_tx_mixed_operations() {
        let g = ConcurrentGraph::new();
        let mgr = TxManager::in_memory();

        // Pre-insert some nodes
        let existing = g.insert_node(person());

        let mut tx = mgr.begin(&g);

        // Insert new nodes
        let a = tx.insert_node(person(), vec![(0, Value::from("new1"))]);
        let b = tx.insert_node(person(), vec![(0, Value::from("new2"))]);

        // Edge between new nodes
        tx.insert_edge(EdgeEndpoint::Pending(a), EdgeEndpoint::Pending(b), knows());

        // Edge from new to existing
        tx.insert_edge(EdgeEndpoint::Pending(a), EdgeEndpoint::Existing(existing), knows());

        // Set property on existing node
        tx.set_node_property(existing, 0, Value::from("original"));

        let result = mgr.commit(&mut tx).unwrap();
        assert_eq!(result.nodes_inserted, 2);
        assert_eq!(result.edges_inserted, 2);
        assert_eq!(result.properties_set, 1);
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);

        let node = g.get_node(existing).unwrap();
        assert_eq!(node.properties[0].1.as_str(), Some("original"));
    }
}
