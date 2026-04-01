// =============================================================================
// bikodb-cluster::router — Query Router y Load Balancer
// =============================================================================
// Dirige queries al shard correcto, ejecuta scatter-gather para queries
// cross-shard, y balancea carga entre nodos.
// =============================================================================

use crate::cluster::{ClusterManager, ClusterNodeId};
use crate::shard::ShardId;
use crate::sharded_graph::ShardedGraph;
use bikodb_core::error::{BikoError, BikoResult};
use bikodb_core::record::Direction;
use bikodb_core::types::{NodeId, TypeId};
use bikodb_core::value::Value;
use bikodb_graph::graph::NodeData;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Estrategia de routing de queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Siempre al shard primario.
    Primary,
    /// Round-robin entre primario y réplicas.
    RoundRobin,
    /// Al nodo con menor carga.
    LeastLoaded,
}

/// Router de queries para el cluster.
///
/// Determina a qué shard enviar cada operación, ejecuta scatter-gather
/// para queries que abarcan múltiples shards, y balancea la carga.
pub struct QueryRouter {
    /// Grafo sharded subyacente.
    graph: Arc<ShardedGraph>,
    /// Manager del cluster (para load info).
    cluster: Arc<ClusterManager>,
    /// Estrategia de routing.
    strategy: RwLock<RoutingStrategy>,
    /// Contador round-robin.
    rr_counter: AtomicU64,
    /// Métricas de routing.
    metrics: RwLock<RouterMetrics>,
}

/// Métricas del router.
#[derive(Debug, Clone, Default)]
pub struct RouterMetrics {
    /// Total de queries ruteados.
    pub total_queries: u64,
    /// Queries que fueron a un solo shard.
    pub single_shard_queries: u64,
    /// Queries que necesitaron scatter-gather.
    pub scatter_gather_queries: u64,
    /// Queries por shard.
    pub queries_per_shard: std::collections::HashMap<ShardId, u64>,
}

impl QueryRouter {
    /// Crea un router con los componentes dados.
    pub fn new(graph: Arc<ShardedGraph>, cluster: Arc<ClusterManager>) -> Self {
        Self {
            graph,
            cluster,
            strategy: RwLock::new(RoutingStrategy::Primary),
            rr_counter: AtomicU64::new(0),
            metrics: RwLock::new(RouterMetrics::default()),
        }
    }

    /// Cambia la estrategia de routing.
    pub fn set_strategy(&self, strategy: RoutingStrategy) {
        *self.strategy.write() = strategy;
    }

    /// Estrategia actual.
    pub fn strategy(&self) -> RoutingStrategy {
        *self.strategy.read()
    }

    // ── Node Operations (routed) ────────────────────────────────────────

    /// Inserta un nodo, ruteándolo al shard apropiado.
    pub fn insert_node(
        &self,
        node_id: NodeId,
        type_id: TypeId,
        props: Vec<(u16, Value)>,
    ) {
        self.graph.insert_node_with_id(node_id, type_id, props);
        self.record_query(self.graph.shard_for(node_id));
    }

    /// Obtiene un nodo por ID.
    pub fn get_node(&self, node_id: NodeId) -> Option<NodeData> {
        let shard = self.graph.shard_for(node_id);
        self.record_query(shard);
        self.graph.get_node(node_id)
    }

    /// Elimina un nodo.
    pub fn remove_node(&self, node_id: NodeId) -> BikoResult<()> {
        let shard = self.graph.shard_for(node_id);
        self.record_query(shard);
        self.graph.remove_node(node_id)
    }

    /// Vecinos de un nodo (puede cruzar shards).
    pub fn neighbors(&self, node_id: NodeId, direction: Direction) -> BikoResult<Vec<NodeId>> {
        let shard = self.graph.shard_for(node_id);
        self.record_query(shard);
        self.graph.neighbors(node_id, direction)
    }

    // ── Scatter-Gather Queries ──────────────────────────────────────────

    /// Busca nodos por tipo en todos los shards (scatter-gather).
    pub fn find_nodes_by_type(&self, type_id: TypeId) -> Vec<NodeData> {
        self.record_scatter_gather();

        let results = self.graph.scatter_gather(|graph| {
            let mut nodes = Vec::new();
            graph.iter_nodes(|_, ndata| {
                if ndata.type_id == type_id {
                    nodes.push(ndata.clone());
                }
            });
            nodes
        });

        results.into_iter().flat_map(|(_, nodes)| nodes).collect()
    }

    /// Cuenta nodos por tipo en todos los shards.
    pub fn count_nodes_by_type(&self, type_id: TypeId) -> usize {
        self.record_scatter_gather();

        let counts = self.graph.scatter_gather(|graph| {
            let mut count = 0;
            graph.iter_nodes(|_, ndata| {
                if ndata.type_id == type_id {
                    count += 1;
                }
            });
            count
        });

        counts.into_iter().map(|(_, c)| c).sum()
    }

    /// Busca nodos que cumplan un predicado en todos los shards.
    pub fn find_nodes<F>(&self, predicate: F) -> Vec<NodeData>
    where
        F: Fn(&NodeData) -> bool + Send + Sync,
    {
        self.record_scatter_gather();

        let pred = Arc::new(predicate);
        let results = self.graph.scatter_gather(|graph| {
            let pred = pred.clone();
            let mut matches = Vec::new();
            graph.iter_nodes(|_, ndata| {
                if pred(ndata) {
                    matches.push(ndata.clone());
                }
            });
            matches
        });

        results.into_iter().flat_map(|(_, nodes)| nodes).collect()
    }

    // ── Load Balancing ──────────────────────────────────────────────────

    /// Selecciona el mejor nodo para un shard según la estrategia de routing.
    pub fn select_node_for_shard(&self, shard: ShardId) -> Option<ClusterNodeId> {
        match *self.strategy.read() {
            RoutingStrategy::Primary => {
                // Find node with this shard as primary
                let nodes = self.cluster.load_distribution();
                // First try to find a node with this shard as primary
                // Since we don't have direct shard→node mapping here, use least loaded
                self.cluster.least_loaded_node()
            }
            RoutingStrategy::RoundRobin => {
                let dist = self.cluster.load_distribution();
                if dist.is_empty() {
                    return None;
                }
                let idx = self.rr_counter.fetch_add(1, Ordering::Relaxed) as usize;
                Some(dist[idx % dist.len()].0)
            }
            RoutingStrategy::LeastLoaded => {
                self.cluster.least_loaded_node()
            }
        }
    }

    // ── Metrics ─────────────────────────────────────────────────────────

    /// Métricas del router.
    pub fn metrics(&self) -> RouterMetrics {
        self.metrics.read().clone()
    }

    /// Registra un query a un shard específico.
    fn record_query(&self, shard: ShardId) {
        let mut m = self.metrics.write();
        m.total_queries += 1;
        m.single_shard_queries += 1;
        *m.queries_per_shard.entry(shard).or_insert(0) += 1;
    }

    /// Registra un scatter-gather query.
    fn record_scatter_gather(&self) {
        let mut m = self.metrics.write();
        m.total_queries += 1;
        m.scatter_gather_queries += 1;
    }

    /// Grafo sharded subyacente.
    pub fn graph(&self) -> &ShardedGraph {
        &self.graph
    }

    /// Cluster manager.
    pub fn cluster(&self) -> &ClusterManager {
        &self.cluster
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard::ShardStrategy;

    fn setup_router() -> QueryRouter {
        let graph = Arc::new(ShardedGraph::new(ShardStrategy::HashBased { num_shards: 2 }));
        let cluster = Arc::new(ClusterManager::new());
        cluster.add_node(ClusterNodeId(1));
        cluster.add_node(ClusterNodeId(2));
        QueryRouter::new(graph, cluster)
    }

    #[test]
    fn test_router_insert_and_get() {
        let router = setup_router();
        router.insert_node(NodeId(1), TypeId(1), vec![(0, Value::Int(42))]);

        let node = router.get_node(NodeId(1)).unwrap();
        assert_eq!(node.type_id, TypeId(1));
    }

    #[test]
    fn test_router_scatter_gather_find_by_type() {
        let router = setup_router();
        router.insert_node(NodeId(1), TypeId(1), vec![]);
        router.insert_node(NodeId(2), TypeId(1), vec![]);
        router.insert_node(NodeId(3), TypeId(2), vec![]);

        let persons = router.find_nodes_by_type(TypeId(1));
        assert_eq!(persons.len(), 2);

        let others = router.find_nodes_by_type(TypeId(2));
        assert_eq!(others.len(), 1);
    }

    #[test]
    fn test_router_count_by_type() {
        let router = setup_router();
        router.insert_node(NodeId(1), TypeId(1), vec![]);
        router.insert_node(NodeId(2), TypeId(1), vec![]);
        router.insert_node(NodeId(3), TypeId(2), vec![]);

        assert_eq!(router.count_nodes_by_type(TypeId(1)), 2);
        assert_eq!(router.count_nodes_by_type(TypeId(2)), 1);
        assert_eq!(router.count_nodes_by_type(TypeId(99)), 0);
    }

    #[test]
    fn test_router_find_with_predicate() {
        let router = setup_router();
        router.insert_node(NodeId(1), TypeId(1), vec![(0, Value::Int(10))]);
        router.insert_node(NodeId(2), TypeId(1), vec![(0, Value::Int(20))]);
        router.insert_node(NodeId(3), TypeId(1), vec![(0, Value::Int(30))]);

        let big = router.find_nodes(|n| {
            n.properties.iter().any(|(_, v)| matches!(v, Value::Int(x) if *x > 15))
        });
        assert_eq!(big.len(), 2); // nodes 2 and 3
    }

    #[test]
    fn test_router_metrics() {
        let router = setup_router();
        router.insert_node(NodeId(1), TypeId(1), vec![]);
        router.get_node(NodeId(1));
        router.find_nodes_by_type(TypeId(1));

        let m = router.metrics();
        assert_eq!(m.total_queries, 3);
        assert_eq!(m.single_shard_queries, 2);
        assert_eq!(m.scatter_gather_queries, 1);
    }

    #[test]
    fn test_routing_strategy_round_robin() {
        let router = setup_router();
        router.set_strategy(RoutingStrategy::RoundRobin);

        let n1 = router.select_node_for_shard(ShardId(0));
        let n2 = router.select_node_for_shard(ShardId(0));
        assert!(n1.is_some());
        assert!(n2.is_some());
        // Round-robin should alternate between nodes
        assert_ne!(n1, n2);
    }

    #[test]
    fn test_routing_strategy_least_loaded() {
        let router = setup_router();
        router.set_strategy(RoutingStrategy::LeastLoaded);

        router.cluster().update_load(ClusterNodeId(1), 0.9);
        router.cluster().update_load(ClusterNodeId(2), 0.1);

        let node = router.select_node_for_shard(ShardId(0)).unwrap();
        assert_eq!(node, ClusterNodeId(2)); // Least loaded
    }

    #[test]
    fn test_router_remove_node() {
        let router = setup_router();
        router.insert_node(NodeId(1), TypeId(1), vec![]);
        assert!(router.get_node(NodeId(1)).is_some());

        router.remove_node(NodeId(1)).unwrap();
        assert!(router.get_node(NodeId(1)).is_none());
    }

    #[test]
    fn test_router_neighbors() {
        let graph = Arc::new(ShardedGraph::new(ShardStrategy::HashBased { num_shards: 1 }));
        let cluster = Arc::new(ClusterManager::new());
        cluster.add_node(ClusterNodeId(1));
        let router = QueryRouter::new(graph, cluster);

        router.insert_node(NodeId(1), TypeId(1), vec![]);
        router.insert_node(NodeId(2), TypeId(1), vec![]);
        router.graph().insert_edge(NodeId(1), NodeId(2), TypeId(10)).unwrap();

        let neighbors = router.neighbors(NodeId(1), Direction::Out).unwrap();
        assert_eq!(neighbors, vec![NodeId(2)]);
    }
}
