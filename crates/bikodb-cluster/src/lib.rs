// =============================================================================
// bikodb-cluster — Cluster Layer: Sharding, Replicación y Tolerancia a Fallos
// =============================================================================
// Distribución del grafo en múltiples shards con:
//
//   shard          → Estrategias de particionamiento (hash, range, graph-aware)
//   sharded_graph  → Grafo distribuido en múltiples ConcurrentGraph shards
//   cluster        → ClusterManager: membresía, health, failover, elección de líder
//   router         → Query router con scatter-gather y load balancing
//
// ## Diseño
// - Sharding inteligente: hash, range, y graph-aware (basado en comunidades)
// - Replicación configurable con consistency levels (One, Quorum, All)
// - Elección de líder estilo Raft (término + lowest-ID)
// - Health checks con heartbeat, suspect y down detection
// - Failover automático: reasignación de shards de nodos caídos
// - Load balancing: least-loaded, round-robin
// - Scatter-gather para queries cross-shard
// - Rebalanceo dinámico al añadir/remover shards
// =============================================================================

pub mod cluster;
pub mod router;
pub mod shard;
pub mod sharded_graph;

pub use cluster::{ClusterEvent, ClusterManager, ClusterNodeId, NodeRole, NodeStatus};
pub use router::{QueryRouter, RouterMetrics, RoutingStrategy};
pub use shard::{ConsistencyLevel, ReplicationConfig, ShardId, ShardStrategy};
pub use sharded_graph::{ShardedGraph, ShardStats};
