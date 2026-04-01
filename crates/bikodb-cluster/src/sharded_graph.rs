// =============================================================================
// bikodb-cluster::sharded_graph — Grafo distribuido en múltiples shards
// =============================================================================
// Distribuye nodos y edges entre múltiples ConcurrentGraph instances,
// permitiendo escalado horizontal y queries cross-shard.
// =============================================================================

use crate::shard::{ShardId, ShardStrategy};
use bikodb_core::error::{BikoError, BikoResult};
use bikodb_core::record::Direction;
use bikodb_core::types::{EdgeId, NodeId, TypeId};
use bikodb_core::value::Value;
use bikodb_graph::graph::{EdgeData, NodeData};
use bikodb_graph::ConcurrentGraph;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Grafo distribuido en múltiples shards.
///
/// Cada shard contiene un `ConcurrentGraph` independiente.
/// El `ShardStrategy` determina en qué shard vive cada nodo.
pub struct ShardedGraph {
    /// Estrategia de particionamiento.
    strategy: RwLock<ShardStrategy>,
    /// Shards: shard_id → grafo.
    shards: RwLock<HashMap<ShardId, Arc<ConcurrentGraph>>>,
    /// Estadísticas por shard.
    shard_stats: RwLock<HashMap<ShardId, ShardStats>>,
}

/// Estadísticas de un shard (para load balancing).
#[derive(Debug, Clone, Default)]
pub struct ShardStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub query_count: u64,
}

impl ShardedGraph {
    /// Crea un grafo sharded con la estrategia dada.
    pub fn new(strategy: ShardStrategy) -> Self {
        let num = strategy.num_shards();
        let mut shards = HashMap::new();
        let mut stats = HashMap::new();
        for i in 0..num {
            let sid = ShardId(i);
            shards.insert(sid, Arc::new(ConcurrentGraph::new()));
            stats.insert(sid, ShardStats::default());
        }
        Self {
            strategy: RwLock::new(strategy),
            shards: RwLock::new(shards),
            shard_stats: RwLock::new(stats),
        }
    }

    /// Referencia a la estrategia actual.
    pub fn strategy(&self) -> ShardStrategy {
        self.strategy.read().clone()
    }

    /// Número de shards.
    pub fn num_shards(&self) -> usize {
        self.shards.read().len()
    }

    /// Obtiene el grafo de un shard específico.
    pub fn shard(&self, id: ShardId) -> Option<Arc<ConcurrentGraph>> {
        self.shards.read().get(&id).cloned()
    }

    /// Determina en qué shard vive un nodo.
    pub fn shard_for(&self, node_id: NodeId) -> ShardId {
        self.strategy.read().shard_for(node_id)
    }

    // ── Node operations ─────────────────────────────────────────────────

    /// Inserta un nodo en el shard correspondiente.
    pub fn insert_node(&self, type_id: TypeId, props: Vec<(u16, Value)>) -> NodeId {
        // Insert into shard 0 first to get a NodeId, then re-route
        let temp_graph = ConcurrentGraph::new();
        let node_id = temp_graph.insert_node_with_props(type_id, props.clone());

        let shard_id = self.shard_for(node_id);
        let shards = self.shards.read();
        if let Some(graph) = shards.get(&shard_id) {
            // We need to insert with the same NodeId — use the graph directly
            // The temp graph gave us a unique ID; insert into actual shard
            let mut data = NodeData::new(node_id, type_id);
            data.properties = props;
            graph.insert_node_data(node_id, data);

            // Update stats
            drop(shards);
            self.update_stats(shard_id);
        }
        node_id
    }

    /// Inserta un nodo con un NodeId específico (para rebalancing).
    pub fn insert_node_with_id(
        &self,
        node_id: NodeId,
        type_id: TypeId,
        props: Vec<(u16, Value)>,
    ) {
        let shard_id = self.shard_for(node_id);
        let shards = self.shards.read();
        if let Some(graph) = shards.get(&shard_id) {
            let mut data = NodeData::new(node_id, type_id);
            data.properties = props;
            graph.insert_node_data(node_id, data);
        }
        drop(shards);
        self.update_stats(shard_id);
    }

    /// Obtiene un nodo buscando en el shard correspondiente.
    pub fn get_node(&self, node_id: NodeId) -> Option<NodeData> {
        let shard_id = self.shard_for(node_id);
        let shards = self.shards.read();
        shards.get(&shard_id).and_then(|g| g.get_node(node_id))
    }

    /// Elimina un nodo del shard correspondiente.
    pub fn remove_node(&self, node_id: NodeId) -> BikoResult<()> {
        let shard_id = self.shard_for(node_id);
        let shards = self.shards.read();
        let result = shards
            .get(&shard_id)
            .ok_or(BikoError::NodeNotFound(node_id))?
            .remove_node(node_id);
        drop(shards);
        self.update_stats(shard_id);
        result
    }

    /// Establece propiedad de un nodo.
    pub fn set_node_property(
        &self,
        node_id: NodeId,
        prop_id: u16,
        value: Value,
    ) -> BikoResult<()> {
        let shard_id = self.shard_for(node_id);
        let shards = self.shards.read();
        shards
            .get(&shard_id)
            .ok_or(BikoError::NodeNotFound(node_id))?
            .set_node_property(node_id, prop_id, value)
    }

    // ── Edge operations ─────────────────────────────────────────────────

    /// Inserta un edge. Si source y target están en el mismo shard, el edge
    /// vive ahí. Si están en shards distintos (cross-shard edge), se almacena
    /// en el shard del source.
    pub fn insert_edge(
        &self,
        source: NodeId,
        target: NodeId,
        type_id: TypeId,
    ) -> BikoResult<EdgeId> {
        let source_shard = self.shard_for(source);
        let target_shard = self.shard_for(target);
        let shards = self.shards.read();

        if source_shard == target_shard {
            // Same shard — straightforward
            let graph = shards
                .get(&source_shard)
                .ok_or(BikoError::NodeNotFound(source))?;
            let result = graph.insert_edge(source, target, type_id);
            drop(shards);
            self.update_stats(source_shard);
            result
        } else {
            // Cross-shard edge — store in source shard
            // First verify target exists in its shard
            let target_exists = shards
                .get(&target_shard)
                .map_or(false, |g| g.get_node(target).is_some());

            if !target_exists {
                return Err(BikoError::NodeNotFound(target));
            }

            // Insert a proxy edge in the source shard
            let graph = shards
                .get(&source_shard)
                .ok_or(BikoError::NodeNotFound(source))?;

            // We need target to exist locally for the edge — create a phantom node
            if graph.get_node(target).is_none() {
                let phantom = NodeData::new(target, TypeId(0));
                graph.insert_node_data(target, phantom);
            }

            let result = graph.insert_edge(source, target, type_id);
            drop(shards);
            self.update_stats(source_shard);
            result
        }
    }

    /// Elimina un edge.
    pub fn remove_edge(&self, source: NodeId, edge_id: EdgeId) -> BikoResult<()> {
        let shard_id = self.shard_for(source);
        let shards = self.shards.read();
        let result = shards
            .get(&shard_id)
            .ok_or(BikoError::Generic(format!("Shard {:?} not found", shard_id)))?
            .remove_edge(edge_id);
        drop(shards);
        self.update_stats(shard_id);
        result
    }

    // ── Cross-shard queries ─────────────────────────────────────────────

    /// Vecinos de un nodo (puede cruzar shards).
    pub fn neighbors(&self, node_id: NodeId, direction: Direction) -> BikoResult<Vec<NodeId>> {
        let shard_id = self.shard_for(node_id);
        let shards = self.shards.read();
        let graph = shards
            .get(&shard_id)
            .ok_or(BikoError::NodeNotFound(node_id))?;

        // Record query
        drop(shards);
        let mut stats = self.shard_stats.write();
        if let Some(s) = stats.get_mut(&shard_id) {
            s.query_count += 1;
        }
        drop(stats);

        let shards = self.shards.read();
        let graph = shards.get(&shard_id).unwrap();
        graph.neighbors(node_id, direction)
    }

    /// Número total de nodos en todos los shards.
    pub fn total_node_count(&self) -> usize {
        let shards = self.shards.read();
        shards.values().map(|g| g.node_count()).sum()
    }

    /// Número total de edges en todos los shards.
    pub fn total_edge_count(&self) -> usize {
        let shards = self.shards.read();
        shards.values().map(|g| g.edge_count()).sum()
    }

    /// Scatter-gather: ejecuta una función en todos los shards y combina resultados.
    pub fn scatter_gather<T, F>(&self, f: F) -> Vec<(ShardId, T)>
    where
        F: Fn(&ConcurrentGraph) -> T,
    {
        let shards = self.shards.read();
        let mut results = Vec::with_capacity(shards.len());
        for (&shard_id, graph) in shards.iter() {
            let mut stats = self.shard_stats.write();
            if let Some(s) = stats.get_mut(&shard_id) {
                s.query_count += 1;
            }
            drop(stats);
            results.push((shard_id, f(graph)));
        }
        results
    }

    /// Itera todos los nodos en todos los shards.
    pub fn iter_all_nodes<F>(&self, mut f: F)
    where
        F: FnMut(ShardId, NodeId, &NodeData),
    {
        let shards = self.shards.read();
        for (&shard_id, graph) in shards.iter() {
            graph.iter_nodes(|nid, ndata| {
                f(shard_id, nid, ndata);
            });
        }
    }

    // ── Stats & Load Balancing ──────────────────────────────────────────

    /// Estadísticas de todos los shards.
    pub fn shard_stats(&self) -> HashMap<ShardId, ShardStats> {
        self.shard_stats.read().clone()
    }

    /// Estadísticas de un shard específico.
    pub fn stats_for(&self, shard_id: ShardId) -> Option<ShardStats> {
        self.shard_stats.read().get(&shard_id).cloned()
    }

    /// Actualiza estadísticas de un shard.
    fn update_stats(&self, shard_id: ShardId) {
        let shards = self.shards.read();
        if let Some(graph) = shards.get(&shard_id) {
            let nc = graph.node_count();
            let ec = graph.edge_count();
            drop(shards);
            let mut stats = self.shard_stats.write();
            if let Some(s) = stats.get_mut(&shard_id) {
                s.node_count = nc;
                s.edge_count = ec;
            }
        }
    }

    /// Calcula el factor de desbalance (max_load / avg_load).
    /// Un valor de 1.0 indica balance perfecto.
    pub fn imbalance_factor(&self) -> f64 {
        let stats = self.shard_stats.read();
        if stats.is_empty() {
            return 1.0;
        }
        let loads: Vec<usize> = stats.values().map(|s| s.node_count).collect();
        let total: usize = loads.iter().sum();
        if total == 0 {
            return 1.0;
        }
        let avg = total as f64 / loads.len() as f64;
        let max = *loads.iter().max().unwrap_or(&0) as f64;
        max / avg
    }

    // ── Rebalancing ─────────────────────────────────────────────────────

    /// Añade un nuevo shard al cluster.
    ///
    /// Crea el shard vacío. Usa `rebalance()` para redistribuir nodos.
    pub fn add_shard(&self) -> ShardId {
        let mut shards = self.shards.write();
        let new_id = ShardId(shards.len() as u16);
        shards.insert(new_id, Arc::new(ConcurrentGraph::new()));
        drop(shards);
        self.shard_stats.write().insert(new_id, ShardStats::default());
        new_id
    }

    /// Rebalancea nodos entre shards usando la estrategia actual.
    ///
    /// Mueve nodos que están en el shard incorrecto según la estrategia.
    /// Retorna el número de nodos movidos.
    pub fn rebalance(&self) -> usize {
        // Collect all nodes and their current shard
        let mut all_nodes: Vec<(NodeId, TypeId, Vec<(u16, Value)>, ShardId)> = Vec::new();

        {
            let shards = self.shards.read();
            for (&shard_id, graph) in shards.iter() {
                graph.iter_nodes(|nid, ndata| {
                    all_nodes.push((nid, ndata.type_id, ndata.properties.clone(), shard_id));
                });
            }
        }

        let strategy = self.strategy.read();
        let mut moved = 0;

        for (node_id, type_id, props, current_shard) in all_nodes {
            let target_shard = strategy.shard_for(node_id);
            if target_shard != current_shard {
                // Move: remove from current, insert into target
                let shards = self.shards.read();
                if let Some(src) = shards.get(&current_shard) {
                    let _ = src.remove_node(node_id);
                }
                if let Some(dst) = shards.get(&target_shard) {
                    let mut data = NodeData::new(node_id, type_id);
                    data.properties = props;
                    dst.insert_node_data(node_id, data);
                }
                moved += 1;
            }
        }

        // Update all stats
        let shards = self.shards.read();
        for (&shard_id, graph) in shards.iter() {
            let nc = graph.node_count();
            let ec = graph.edge_count();
            let mut stats = self.shard_stats.write();
            if let Some(s) = stats.get_mut(&shard_id) {
                s.node_count = nc;
                s.edge_count = ec;
            }
        }

        moved
    }

    /// Cambia la estrategia de sharding y rebalancea.
    ///
    /// Retorna el número de nodos movidos.
    pub fn repartition(&self, new_strategy: ShardStrategy) -> usize {
        // Ensure we have enough shards
        let needed = new_strategy.num_shards();
        {
            let mut shards = self.shards.write();
            for i in shards.len() as u16..needed {
                let sid = ShardId(i);
                shards.insert(sid, Arc::new(ConcurrentGraph::new()));
                self.shard_stats.write().insert(sid, ShardStats::default());
            }
        }

        *self.strategy.write() = new_strategy;
        self.rebalance()
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharded_graph_basic() {
        let sg = ShardedGraph::new(ShardStrategy::HashBased { num_shards: 4 });
        assert_eq!(sg.num_shards(), 4);
        assert_eq!(sg.total_node_count(), 0);
    }

    #[test]
    fn test_insert_and_get_node() {
        let sg = ShardedGraph::new(ShardStrategy::HashBased { num_shards: 2 });

        sg.insert_node_with_id(NodeId(1), TypeId(1), vec![(0, Value::Int(42))]);
        sg.insert_node_with_id(NodeId(2), TypeId(1), vec![(0, Value::Int(99))]);

        assert_eq!(sg.total_node_count(), 2);
        let n1 = sg.get_node(NodeId(1)).unwrap();
        assert_eq!(n1.type_id, TypeId(1));
    }

    #[test]
    fn test_nodes_distributed_across_shards() {
        let sg = ShardedGraph::new(ShardStrategy::HashBased { num_shards: 4 });

        for i in 1..=100 {
            sg.insert_node_with_id(NodeId(i), TypeId(1), vec![]);
        }

        assert_eq!(sg.total_node_count(), 100);

        // Verify distribution: each shard should have some nodes
        let stats = sg.shard_stats();
        let total: usize = stats.values().map(|s| s.node_count).sum();
        assert_eq!(total, 100);

        // With hash-based sharding and 100 nodes across 4 shards, each shard
        // should have a reasonable share (not all in one)
        let max = stats.values().map(|s| s.node_count).max().unwrap();
        assert!(max < 100); // Not all in one shard
    }

    #[test]
    fn test_insert_edge_same_shard() {
        let sg = ShardedGraph::new(ShardStrategy::HashBased { num_shards: 1 });
        sg.insert_node_with_id(NodeId(1), TypeId(1), vec![]);
        sg.insert_node_with_id(NodeId(2), TypeId(1), vec![]);

        let edge_id = sg.insert_edge(NodeId(1), NodeId(2), TypeId(10)).unwrap();
        assert_eq!(sg.total_edge_count(), 1);
    }

    #[test]
    fn test_cross_shard_edge() {
        // Force nodes to different shards using range-based
        let sg = ShardedGraph::new(ShardStrategy::RangeBased {
            boundaries: vec![5],
        });
        sg.insert_node_with_id(NodeId(1), TypeId(1), vec![]);  // shard 0
        sg.insert_node_with_id(NodeId(10), TypeId(1), vec![]); // shard 1

        // Cross-shard edge
        let result = sg.insert_edge(NodeId(1), NodeId(10), TypeId(10));
        assert!(result.is_ok());
    }

    #[test]
    fn test_remove_node() {
        let sg = ShardedGraph::new(ShardStrategy::HashBased { num_shards: 2 });
        sg.insert_node_with_id(NodeId(1), TypeId(1), vec![]);
        assert_eq!(sg.total_node_count(), 1);

        sg.remove_node(NodeId(1)).unwrap();
        assert_eq!(sg.total_node_count(), 0);
    }

    #[test]
    fn test_neighbors() {
        let sg = ShardedGraph::new(ShardStrategy::HashBased { num_shards: 1 });
        sg.insert_node_with_id(NodeId(1), TypeId(1), vec![]);
        sg.insert_node_with_id(NodeId(2), TypeId(1), vec![]);
        sg.insert_edge(NodeId(1), NodeId(2), TypeId(10)).unwrap();

        let neighbors = sg.neighbors(NodeId(1), Direction::Out).unwrap();
        assert_eq!(neighbors, vec![NodeId(2)]);
    }

    #[test]
    fn test_scatter_gather() {
        let sg = ShardedGraph::new(ShardStrategy::HashBased { num_shards: 3 });
        for i in 1..=30 {
            sg.insert_node_with_id(NodeId(i), TypeId(1), vec![]);
        }

        let counts: Vec<(ShardId, usize)> =
            sg.scatter_gather(|g| g.node_count());
        let total: usize = counts.iter().map(|(_, c)| c).sum();
        assert_eq!(total, 30);
    }

    #[test]
    fn test_imbalance_factor() {
        let sg = ShardedGraph::new(ShardStrategy::HashBased { num_shards: 2 });
        // Empty graph → perfect balance
        assert!((sg.imbalance_factor() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_add_shard() {
        let sg = ShardedGraph::new(ShardStrategy::HashBased { num_shards: 2 });
        assert_eq!(sg.num_shards(), 2);

        let new_id = sg.add_shard();
        assert_eq!(sg.num_shards(), 3);
        assert!(sg.shard(new_id).is_some());
    }

    #[test]
    fn test_rebalance() {
        // Start with 2 shards, put all nodes in shard 0
        let sg = ShardedGraph::new(ShardStrategy::HashBased { num_shards: 2 });

        // Insert nodes directly into shard 0
        let shard0 = sg.shard(ShardId(0)).unwrap();
        for i in 1..=10 {
            let mut data = NodeData::new(NodeId(i), TypeId(1));
            data.properties = vec![(0, Value::Int(i as i64))];
            shard0.insert_node_data(NodeId(i), data);
        }

        // Some of these nodes should belong on shard 1 per hash strategy
        let moved = sg.rebalance();
        assert!(moved > 0, "Should have moved some nodes to shard 1");
    }

    #[test]
    fn test_repartition_to_more_shards() {
        let sg = ShardedGraph::new(ShardStrategy::HashBased { num_shards: 2 });
        for i in 1..=20 {
            sg.insert_node_with_id(NodeId(i), TypeId(1), vec![]);
        }
        assert_eq!(sg.total_node_count(), 20);
        assert_eq!(sg.num_shards(), 2);

        // Repartition to 4 shards
        let moved = sg.repartition(ShardStrategy::HashBased { num_shards: 4 });
        assert_eq!(sg.num_shards(), 4);
        assert_eq!(sg.total_node_count(), 20); // No data lost
    }

    #[test]
    fn test_set_node_property() {
        let sg = ShardedGraph::new(ShardStrategy::HashBased { num_shards: 2 });
        sg.insert_node_with_id(NodeId(1), TypeId(1), vec![(0, Value::Int(1))]);

        sg.set_node_property(NodeId(1), 0, Value::Int(42)).unwrap();

        let node = sg.get_node(NodeId(1)).unwrap();
        let val = node.properties.iter().find(|(k, _)| *k == 0).map(|(_, v)| v.clone());
        assert_eq!(val, Some(Value::Int(42)));
    }
}
