// =============================================================================
// bikodb-graph::mutation — API de alto nivel para mutaciones del grafo
// =============================================================================
// Dos APIs de mutación:
//
// 1. `GraphMutator` — pass-through directo al grafo (bajo overhead, sin buffering)
// 2. `BufferedMutator` — acumula mutaciones en buffers y las commitea en batch
//    al llamar `commit()`. Ideal para operaciones transaccionales y bulk insert
//    con control explícito del flush point.
//
// En v0.2 integrará WAL + transacciones ACID.
// =============================================================================

use crate::graph::{ConcurrentGraph, NodeData};
use bikodb_core::error::BikoResult;
use bikodb_core::record::Direction;
use bikodb_core::types::{EdgeId, NodeId, TypeId};
use bikodb_core::value::Value;

/// Mutator directo: cada operación se aplica inmediatamente al grafo.
///
/// # Ejemplo
/// ```
/// use bikodb_graph::{ConcurrentGraph, mutation::GraphMutator};
/// use bikodb_core::types::TypeId;
/// use bikodb_core::value::Value;
///
/// let graph = ConcurrentGraph::new();
/// let mut mutator = GraphMutator::new(&graph);
///
/// let alice = mutator.add_node(TypeId(1), vec![(0, Value::from("Alice"))]);
/// let bob = mutator.add_node(TypeId(1), vec![(0, Value::from("Bob"))]);
/// mutator.add_edge(alice, bob, TypeId(10)).unwrap();
///
/// assert_eq!(graph.node_count(), 2);
/// assert_eq!(graph.edge_count(), 1);
/// ```
pub struct GraphMutator<'g> {
    graph: &'g ConcurrentGraph,
}

impl<'g> GraphMutator<'g> {
    pub fn new(graph: &'g ConcurrentGraph) -> Self {
        Self { graph }
    }

    pub fn add_node(&mut self, type_id: TypeId, properties: Vec<(u16, Value)>) -> NodeId {
        self.graph.insert_node_with_props(type_id, properties)
    }

    pub fn add_edge(
        &mut self,
        source: NodeId,
        target: NodeId,
        type_id: TypeId,
    ) -> BikoResult<EdgeId> {
        self.graph.insert_edge(source, target, type_id)
    }

    pub fn add_edge_with_props(
        &mut self,
        source: NodeId,
        target: NodeId,
        type_id: TypeId,
        properties: Vec<(u16, Value)>,
    ) -> BikoResult<EdgeId> {
        self.graph
            .insert_edge_with_props(source, target, type_id, properties)
    }

    pub fn remove_node(&mut self, id: NodeId) -> BikoResult<()> {
        self.graph.remove_node(id)
    }

    pub fn remove_edge(&mut self, id: EdgeId) -> BikoResult<()> {
        self.graph.remove_edge(id)
    }

    pub fn set_property(&self, node_id: NodeId, prop_id: u16, value: Value) -> BikoResult<()> {
        self.graph.set_node_property(node_id, prop_id, value)
    }

    pub fn get_node(&self, id: NodeId) -> Option<NodeData> {
        self.graph.get_node(id)
    }

    pub fn neighbors(&self, id: NodeId, direction: Direction) -> BikoResult<Vec<NodeId>> {
        self.graph.neighbors(id, direction)
    }
}

// =============================================================================
// BufferedMutator — Acumulador con commit batch
// =============================================================================

/// Operación pendiente en el buffer.
enum PendingOp {
    InsertNode {
        type_id: TypeId,
        properties: Vec<(u16, Value)>,
    },
    InsertEdge {
        source: NodeId,
        target: NodeId,
        type_id: TypeId,
    },
}

/// Mutator con buffering: acumula operaciones y las commitea en batch.
///
/// A diferencia de `GraphMutator`, las operaciones no se aplican hasta
/// llamar `commit()`. Esto permite:
/// - Batch insert paralelizado con `insert_nodes_batch`/`insert_edges_batch`
/// - Rollback implícito si se dropea sin commit
/// - Menor contención: menos lock acquisitions en DashMap
///
/// # Ejemplo
/// ```
/// use bikodb_graph::{ConcurrentGraph, mutation::BufferedMutator};
/// use bikodb_core::types::TypeId;
/// use bikodb_core::value::Value;
///
/// let graph = ConcurrentGraph::new();
/// let mut buf = BufferedMutator::new(&graph);
///
/// buf.add_node(TypeId(1), vec![(0, Value::from("Alice"))]);
/// buf.add_node(TypeId(1), vec![(0, Value::from("Bob"))]);
/// assert_eq!(graph.node_count(), 0); // Not committed yet
///
/// let (nodes, edges) = buf.commit();
/// assert_eq!(graph.node_count(), 2); // Now visible
/// assert_eq!(nodes, 2);
/// ```
pub struct BufferedMutator<'g> {
    graph: &'g ConcurrentGraph,
    ops: Vec<PendingOp>,
}

impl<'g> BufferedMutator<'g> {
    pub fn new(graph: &'g ConcurrentGraph) -> Self {
        Self {
            graph,
            ops: Vec::new(),
        }
    }

    pub fn with_capacity(graph: &'g ConcurrentGraph, capacity: usize) -> Self {
        Self {
            graph,
            ops: Vec::with_capacity(capacity),
        }
    }

    /// Agrega un nodo al buffer (no se inserta hasta commit).
    pub fn add_node(&mut self, type_id: TypeId, properties: Vec<(u16, Value)>) {
        self.ops.push(PendingOp::InsertNode { type_id, properties });
    }

    /// Agrega un edge al buffer (source/target deben existir al commit).
    pub fn add_edge(&mut self, source: NodeId, target: NodeId, type_id: TypeId) {
        self.ops.push(PendingOp::InsertEdge {
            source,
            target,
            type_id,
        });
    }

    /// Número de operaciones pendientes.
    pub fn pending(&self) -> usize {
        self.ops.len()
    }

    /// Descarta todas las operaciones pendientes (rollback implícito).
    pub fn discard(&mut self) {
        self.ops.clear();
    }

    /// Commitea todas las operaciones pendientes al grafo en batch.
    ///
    /// Nodos se insertan primero (en paralelo), luego edges (en paralelo).
    /// Retorna (nodes_inserted, edges_inserted).
    pub fn commit(&mut self) -> (usize, usize) {
        let ops = std::mem::take(&mut self.ops);

        let mut node_entries = Vec::new();
        let mut edge_entries = Vec::new();

        for op in ops {
            match op {
                PendingOp::InsertNode { type_id, properties } => {
                    node_entries.push((type_id, properties));
                }
                PendingOp::InsertEdge {
                    source,
                    target,
                    type_id,
                } => {
                    edge_entries.push((source, target, type_id));
                }
            }
        }

        let node_count = node_entries.len();
        if !node_entries.is_empty() {
            self.graph.insert_nodes_batch(&node_entries);
        }

        let edge_results = if !edge_entries.is_empty() {
            self.graph.insert_edges_batch(&edge_entries)
        } else {
            Vec::new()
        };
        let edge_count = edge_results.iter().filter(|r| r.is_ok()).count();

        (node_count, edge_count)
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutator_add_and_query() {
        let g = ConcurrentGraph::new();
        let mut m = GraphMutator::new(&g);

        let alice = m.add_node(TypeId(1), vec![(0, Value::from("Alice"))]);
        let bob = m.add_node(TypeId(1), vec![(0, Value::from("Bob"))]);
        let charlie = m.add_node(TypeId(1), vec![(0, Value::from("Charlie"))]);

        m.add_edge(alice, bob, TypeId(10)).unwrap();
        m.add_edge(alice, charlie, TypeId(10)).unwrap();
        m.add_edge(bob, charlie, TypeId(10)).unwrap();

        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 3);

        let neighbors = m.neighbors(alice, Direction::Out).unwrap();
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_mutator_remove() {
        let g = ConcurrentGraph::new();
        let mut m = GraphMutator::new(&g);

        let a = m.add_node(TypeId(1), vec![]);
        let b = m.add_node(TypeId(1), vec![]);
        m.add_edge(a, b, TypeId(10)).unwrap();

        m.remove_node(a).unwrap();
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_mutator_set_property() {
        let g = ConcurrentGraph::new();
        let mut m = GraphMutator::new(&g);

        let a = m.add_node(TypeId(1), vec![]);
        m.set_property(a, 0, Value::from("name_value")).unwrap();

        let node = m.get_node(a).unwrap();
        assert_eq!(node.properties[0].1.as_str(), Some("name_value"));
    }

    // ── BufferedMutator tests ───────────────────────────────────────────

    #[test]
    fn test_buffered_not_visible_before_commit() {
        let g = ConcurrentGraph::new();
        let mut buf = BufferedMutator::new(&g);

        buf.add_node(TypeId(1), vec![(0, Value::from("Alice"))]);
        buf.add_node(TypeId(1), vec![(0, Value::from("Bob"))]);

        assert_eq!(g.node_count(), 0, "Should not be visible before commit");
        assert_eq!(buf.pending(), 2);

        let (nodes, edges) = buf.commit();
        assert_eq!(nodes, 2);
        assert_eq!(edges, 0);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_buffered_discard() {
        let g = ConcurrentGraph::new();
        let mut buf = BufferedMutator::new(&g);

        buf.add_node(TypeId(1), vec![]);
        buf.add_node(TypeId(1), vec![]);
        buf.discard();

        assert_eq!(buf.pending(), 0);
        let (nodes, _) = buf.commit();
        assert_eq!(nodes, 0);
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_buffered_with_edges() {
        let g = ConcurrentGraph::new();
        // Pre-insert nodes so edges can reference them
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));

        let mut buf = BufferedMutator::new(&g);
        buf.add_edge(a, b, TypeId(10));
        buf.add_edge(b, a, TypeId(10));

        assert_eq!(g.edge_count(), 0);

        let (_, edges) = buf.commit();
        assert_eq!(edges, 2);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_buffered_10k_batch_commit() {
        let g = ConcurrentGraph::new();
        let mut buf = BufferedMutator::with_capacity(&g, 10_000);

        for _ in 0..10_000 {
            buf.add_node(TypeId(1), vec![]);
        }

        let (nodes, _) = buf.commit();
        assert_eq!(nodes, 10_000);
        assert_eq!(g.node_count(), 10_000);
    }
}
