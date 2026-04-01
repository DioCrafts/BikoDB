// =============================================================================
// bikodb-cluster::shard — Estrategia de particionamiento
// =============================================================================

use bikodb_core::types::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Identificador de shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShardId(pub u16);

/// Estrategia de sharding del grafo.
///
/// Define cómo se distribuyen los nodos entre shards (nodos de cluster).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardStrategy {
    /// Hash del NodeId módulo número de shards.
    /// Simple y uniforme, pero puede separar vecinos.
    HashBased { num_shards: u16 },

    /// Sharding por rangos de NodeId.
    /// Mejor locality pero puede generar hotspots.
    RangeBased { boundaries: Vec<u64> },

    /// Sharding basado en comunidades/labels del grafo.
    /// Mantiene vecinos juntos para minimizar edge-cuts entre shards.
    GraphAware {
        num_shards: u16,
        /// Asignación explícita: node_id → shard_id.
        /// Los nodos no mapeados usan hash-based fallback.
        assignments: HashMap<u64, u16>,
    },
}

impl ShardStrategy {
    /// Determina en qué shard vive un nodo.
    pub fn shard_for(&self, node_id: NodeId) -> ShardId {
        match self {
            ShardStrategy::HashBased { num_shards } => {
                // FNV-1a hash simplified
                let hash = node_id.0.wrapping_mul(0x517cc1b727220a95);
                ShardId((hash % *num_shards as u64) as u16)
            }
            ShardStrategy::RangeBased { boundaries } => {
                for (i, &boundary) in boundaries.iter().enumerate() {
                    if node_id.0 < boundary {
                        return ShardId(i as u16);
                    }
                }
                ShardId(boundaries.len() as u16)
            }
            ShardStrategy::GraphAware {
                num_shards,
                assignments,
            } => {
                if let Some(&shard) = assignments.get(&node_id.0) {
                    ShardId(shard)
                } else {
                    // Fallback to hash
                    let hash = node_id.0.wrapping_mul(0x517cc1b727220a95);
                    ShardId((hash % *num_shards as u64) as u16)
                }
            }
        }
    }

    /// Número total de shards en esta estrategia.
    pub fn num_shards(&self) -> u16 {
        match self {
            ShardStrategy::HashBased { num_shards } => *num_shards,
            ShardStrategy::RangeBased { boundaries } => boundaries.len() as u16 + 1,
            ShardStrategy::GraphAware { num_shards, .. } => *num_shards,
        }
    }

    /// Genera una estrategia graph-aware a partir de comunidades detectadas.
    ///
    /// `communities`: mapa de node_id → community_id (de community detection).
    /// Distribuye comunidades entre shards usando round-robin ponderado.
    pub fn from_communities(
        communities: &HashMap<NodeId, u32>,
        num_shards: u16,
    ) -> ShardStrategy {
        // Group nodes by community
        let mut community_nodes: HashMap<u32, Vec<NodeId>> = HashMap::new();
        for (&node_id, &community) in communities {
            community_nodes
                .entry(community)
                .or_default()
                .push(node_id);
        }

        // Sort communities by size (largest first) for better balance
        let mut sorted_communities: Vec<(u32, Vec<NodeId>)> =
            community_nodes.into_iter().collect();
        sorted_communities.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        // Assign communities to shards using greedy load balancing
        let mut shard_loads = vec![0usize; num_shards as usize];
        let mut assignments = HashMap::new();

        for (_community_id, nodes) in sorted_communities {
            // Find the least loaded shard
            let target_shard = shard_loads
                .iter()
                .enumerate()
                .min_by_key(|(_, &load)| load)
                .map(|(i, _)| i as u16)
                .unwrap_or(0);

            shard_loads[target_shard as usize] += nodes.len();
            for node in nodes {
                assignments.insert(node.0, target_shard);
            }
        }

        ShardStrategy::GraphAware {
            num_shards,
            assignments,
        }
    }
}

/// Nivel de consistencia para operaciones de lectura.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Lee de cualquier réplica (más rápido, puede ser stale)
    One,
    /// Lee de la mayoría de réplicas
    Quorum,
    /// Lee de todas las réplicas (más lento, siempre consistente)
    All,
}

/// Configuración de replicación.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
    /// Factor de replicación (cuántas copias de cada shard)
    pub replication_factor: u8,
    /// Consistencia por defecto para reads
    pub read_consistency: ConsistencyLevel,
    /// Consistencia por defecto para writes
    pub write_consistency: ConsistencyLevel,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            replication_factor: 3,
            read_consistency: ConsistencyLevel::Quorum,
            write_consistency: ConsistencyLevel::Quorum,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_sharding() {
        let strategy = ShardStrategy::HashBased { num_shards: 4 };
        let s1 = strategy.shard_for(NodeId(1));
        let s2 = strategy.shard_for(NodeId(1));
        assert_eq!(s1, s2); // Deterministic

        // All shards should be within range
        for i in 0..100 {
            let s = strategy.shard_for(NodeId(i));
            assert!(s.0 < 4);
        }
    }

    #[test]
    fn test_range_sharding() {
        let strategy = ShardStrategy::RangeBased {
            boundaries: vec![100, 200, 300],
        };

        assert_eq!(strategy.shard_for(NodeId(50)).0, 0);
        assert_eq!(strategy.shard_for(NodeId(150)).0, 1);
        assert_eq!(strategy.shard_for(NodeId(250)).0, 2);
        assert_eq!(strategy.shard_for(NodeId(350)).0, 3);
    }

    #[test]
    fn test_graph_aware_sharding() {
        let mut assignments = HashMap::new();
        // Community A → shard 0
        assignments.insert(1, 0);
        assignments.insert(2, 0);
        assignments.insert(3, 0);
        // Community B → shard 1
        assignments.insert(4, 1);
        assignments.insert(5, 1);

        let strategy = ShardStrategy::GraphAware {
            num_shards: 2,
            assignments,
        };

        assert_eq!(strategy.shard_for(NodeId(1)).0, 0);
        assert_eq!(strategy.shard_for(NodeId(4)).0, 1);

        // Unmapped node → hash fallback
        let s = strategy.shard_for(NodeId(999));
        assert!(s.0 < 2);
    }

    #[test]
    fn test_from_communities() {
        let mut communities = HashMap::new();
        // Community 0: nodes 1,2,3
        communities.insert(NodeId(1), 0);
        communities.insert(NodeId(2), 0);
        communities.insert(NodeId(3), 0);
        // Community 1: nodes 4,5
        communities.insert(NodeId(4), 1);
        communities.insert(NodeId(5), 1);
        // Community 2: node 6
        communities.insert(NodeId(6), 2);

        let strategy = ShardStrategy::from_communities(&communities, 2);

        // All nodes in same community should be on same shard
        let s1 = strategy.shard_for(NodeId(1));
        let s2 = strategy.shard_for(NodeId(2));
        let s3 = strategy.shard_for(NodeId(3));
        assert_eq!(s1, s2);
        assert_eq!(s2, s3);

        let s4 = strategy.shard_for(NodeId(4));
        let s5 = strategy.shard_for(NodeId(5));
        assert_eq!(s4, s5);

        // Communities should be on different shards (greedy balancing)
        // Community 0 (3 nodes) assigned first, then community 1 (2 nodes) to other shard
        assert_ne!(s1, s4);
    }

    #[test]
    fn test_num_shards() {
        assert_eq!(ShardStrategy::HashBased { num_shards: 4 }.num_shards(), 4);
        assert_eq!(
            ShardStrategy::RangeBased { boundaries: vec![100, 200] }.num_shards(),
            3
        );
        assert_eq!(
            ShardStrategy::GraphAware { num_shards: 8, assignments: HashMap::new() }.num_shards(),
            8
        );
    }
}
