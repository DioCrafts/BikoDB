// =============================================================================
// bikodb-cluster::cluster — Cluster Manager con tolerancia a fallos
// =============================================================================
// Gestiona nodos lógicos del cluster: health checks, failover, replicación,
// y estado del cluster.
// =============================================================================

use crate::shard::{ConsistencyLevel, ReplicationConfig, ShardId};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Identificador de un nodo del cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClusterNodeId(pub u16);

/// Estado de un nodo del cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Nodo activo y saludable.
    Active,
    /// Nodo sospechoso (heartbeat tardío).
    Suspect,
    /// Nodo caído / no responde.
    Down,
    /// Nodo en proceso de unirse al cluster.
    Joining,
    /// Nodo en proceso de abandonar el cluster.
    Leaving,
}

/// Rol de un nodo en el cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Nodo líder: coordina writes y replicación.
    Leader,
    /// Nodo seguidor: replica datos del líder.
    Follower,
    /// Nodo candidato (durante elección).
    Candidate,
}

/// Información de un nodo del cluster.
#[derive(Debug, Clone)]
pub struct ClusterNode {
    pub id: ClusterNodeId,
    pub status: NodeStatus,
    pub role: NodeRole,
    /// Shards primarios asignados a este nodo.
    pub primary_shards: Vec<ShardId>,
    /// Shards réplica asignados a este nodo.
    pub replica_shards: Vec<ShardId>,
    /// Último heartbeat recibido.
    pub last_heartbeat: Instant,
    /// Carga actual (proporción 0.0 - 1.0).
    pub load: f64,
}

impl ClusterNode {
    fn new(id: ClusterNodeId) -> Self {
        Self {
            id,
            status: NodeStatus::Joining,
            role: NodeRole::Follower,
            primary_shards: Vec::new(),
            replica_shards: Vec::new(),
            last_heartbeat: Instant::now(),
            load: 0.0,
        }
    }

    /// ¿Está el nodo disponible para recibir operaciones?
    pub fn is_available(&self) -> bool {
        self.status == NodeStatus::Active
    }
}

/// Evento del cluster.
#[derive(Debug, Clone)]
pub enum ClusterEvent {
    /// Nodo se unió al cluster.
    NodeJoined(ClusterNodeId),
    /// Nodo abandonó el cluster.
    NodeLeft(ClusterNodeId),
    /// Nodo detectado como caído.
    NodeDown(ClusterNodeId),
    /// Nodo recuperado.
    NodeRecovered(ClusterNodeId),
    /// Nuevo líder elegido.
    LeaderElected(ClusterNodeId),
    /// Shard reasignado.
    ShardReassigned {
        shard: ShardId,
        from: ClusterNodeId,
        to: ClusterNodeId,
    },
    /// Rebalanceo completado.
    RebalanceCompleted { nodes_moved: usize },
}

/// Callback para eventos del cluster.
pub type ClusterEventCallback = Box<dyn Fn(&ClusterEvent) + Send + Sync>;

/// Manager del cluster.
///
/// Gestiona membresía, health checks, elección de líder y failover.
pub struct ClusterManager {
    /// Nodos registrados en el cluster.
    nodes: RwLock<HashMap<ClusterNodeId, ClusterNode>>,
    /// Configuración de replicación.
    replication_config: RwLock<ReplicationConfig>,
    /// ID del líder actual.
    leader_id: RwLock<Option<ClusterNodeId>>,
    /// Timeout para considerar un nodo como suspect.
    suspect_timeout: Duration,
    /// Timeout para considerar un nodo como down.
    down_timeout: Duration,
    /// Log de eventos del cluster.
    event_log: RwLock<Vec<ClusterEvent>>,
    /// Listeners de eventos.
    event_listeners: RwLock<Vec<ClusterEventCallback>>,
    /// Término actual (para elección de líder estilo Raft).
    current_term: RwLock<u64>,
}

impl ClusterManager {
    /// Crea un nuevo ClusterManager.
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            replication_config: RwLock::new(ReplicationConfig::default()),
            leader_id: RwLock::new(None),
            suspect_timeout: Duration::from_secs(5),
            down_timeout: Duration::from_secs(15),
            event_log: RwLock::new(Vec::new()),
            event_listeners: RwLock::new(Vec::new()),
            current_term: RwLock::new(0),
        }
    }

    /// Crea un ClusterManager con configuración personalizada.
    pub fn with_config(
        replication: ReplicationConfig,
        suspect_timeout: Duration,
        down_timeout: Duration,
    ) -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            replication_config: RwLock::new(replication),
            leader_id: RwLock::new(None),
            suspect_timeout,
            down_timeout,
            event_log: RwLock::new(Vec::new()),
            event_listeners: RwLock::new(Vec::new()),
            current_term: RwLock::new(0),
        }
    }

    // ── Membresía ───────────────────────────────────────────────────────

    /// Registra un nuevo nodo en el cluster.
    pub fn add_node(&self, id: ClusterNodeId) {
        let mut node = ClusterNode::new(id);
        node.status = NodeStatus::Active;

        // If no leader, first node becomes leader
        let mut leader = self.leader_id.write();
        if leader.is_none() {
            node.role = NodeRole::Leader;
            *leader = Some(id);
            drop(leader);
            self.emit_event(ClusterEvent::LeaderElected(id));
        } else {
            drop(leader);
        }

        self.nodes.write().insert(id, node);
        self.emit_event(ClusterEvent::NodeJoined(id));
    }

    /// Remueve un nodo del cluster (salida limpia).
    pub fn remove_node(&self, id: ClusterNodeId) {
        let was_leader = {
            let mut nodes = self.nodes.write();
            if let Some(mut node) = nodes.get_mut(&id) {
                node.status = NodeStatus::Leaving;
            }
            nodes.remove(&id);
            *self.leader_id.read() == Some(id)
        };

        self.emit_event(ClusterEvent::NodeLeft(id));

        // If leader left, elect new one
        if was_leader {
            self.elect_leader();
        }
    }

    /// Número de nodos en el cluster.
    pub fn node_count(&self) -> usize {
        self.nodes.read().len()
    }

    /// Número de nodos activos.
    pub fn active_node_count(&self) -> usize {
        self.nodes.read().values().filter(|n| n.is_available()).count()
    }

    /// Obtiene información de un nodo.
    pub fn get_node(&self, id: ClusterNodeId) -> Option<ClusterNode> {
        self.nodes.read().get(&id).cloned()
    }

    /// ID del líder actual.
    pub fn leader(&self) -> Option<ClusterNodeId> {
        *self.leader_id.read()
    }

    /// ¿Está el nodo dado como líder?
    pub fn is_leader(&self, id: ClusterNodeId) -> bool {
        *self.leader_id.read() == Some(id)
    }

    // ── Health Checks ───────────────────────────────────────────────────

    /// Registra un heartbeat de un nodo.
    pub fn heartbeat(&self, id: ClusterNodeId) {
        let mut nodes = self.nodes.write();
        if let Some(node) = nodes.get_mut(&id) {
            node.last_heartbeat = Instant::now();
            if node.status == NodeStatus::Suspect || node.status == NodeStatus::Down {
                let was_down = node.status == NodeStatus::Down;
                node.status = NodeStatus::Active;
                drop(nodes);
                if was_down {
                    self.emit_event(ClusterEvent::NodeRecovered(id));
                }
            }
        }
    }

    /// Actualiza la carga de un nodo.
    pub fn update_load(&self, id: ClusterNodeId, load: f64) {
        let mut nodes = self.nodes.write();
        if let Some(node) = nodes.get_mut(&id) {
            node.load = load.clamp(0.0, 1.0);
        }
    }

    /// Ejecuta health check: marca nodos sospechosos o caídos.
    ///
    /// Retorna los nodos que cambiaron de estado.
    pub fn check_health(&self) -> Vec<ClusterEvent> {
        let now = Instant::now();
        let mut events = Vec::new();
        let mut needs_election = false;

        {
            let mut nodes = self.nodes.write();
            for (id, node) in nodes.iter_mut() {
                let elapsed = now.duration_since(node.last_heartbeat);

                match node.status {
                    NodeStatus::Active if elapsed > self.suspect_timeout => {
                        node.status = NodeStatus::Suspect;
                    }
                    NodeStatus::Suspect if elapsed > self.down_timeout => {
                        node.status = NodeStatus::Down;
                        events.push(ClusterEvent::NodeDown(*id));
                        if *self.leader_id.read() == Some(*id) {
                            needs_election = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        for event in &events {
            self.emit_event(event.clone());
        }

        if needs_election {
            self.elect_leader();
        }

        events
    }

    // ── Elección de líder ───────────────────────────────────────────────

    /// Elección de líder simplificada (estilo Raft).
    ///
    /// El nodo activo con menor ID gana. En un sistema real, esto usaría
    /// votación con términos Raft.
    pub fn elect_leader(&self) -> Option<ClusterNodeId> {
        let mut term = self.current_term.write();
        *term += 1;

        let mut nodes = self.nodes.write();

        // Find the active node with lowest ID
        let new_leader = nodes
            .iter()
            .filter(|(_, n)| n.is_available())
            .min_by_key(|(id, _)| id.0)
            .map(|(id, _)| *id);

        // Demote old leader
        for node in nodes.values_mut() {
            if node.role == NodeRole::Leader {
                node.role = NodeRole::Follower;
            }
        }

        if let Some(leader_id) = new_leader {
            if let Some(node) = nodes.get_mut(&leader_id) {
                node.role = NodeRole::Leader;
            }
            drop(nodes);
            *self.leader_id.write() = Some(leader_id);
            self.emit_event(ClusterEvent::LeaderElected(leader_id));
        } else {
            drop(nodes);
            *self.leader_id.write() = None;
        }

        drop(term);
        new_leader
    }

    /// Término actual de la elección.
    pub fn current_term(&self) -> u64 {
        *self.current_term.read()
    }

    // ── Shard Assignment ────────────────────────────────────────────────

    /// Asigna shards primarios a nodos usando round-robin.
    pub fn assign_shards(&self, shard_ids: &[ShardId]) {
        let mut nodes = self.nodes.write();
        let active: Vec<ClusterNodeId> = nodes
            .iter()
            .filter(|(_, n)| n.is_available())
            .map(|(id, _)| *id)
            .collect();

        if active.is_empty() {
            return;
        }

        // Clear existing assignments
        for node in nodes.values_mut() {
            node.primary_shards.clear();
            node.replica_shards.clear();
        }

        // Round-robin primary assignment
        for (i, &shard) in shard_ids.iter().enumerate() {
            let node_id = active[i % active.len()];
            if let Some(node) = nodes.get_mut(&node_id) {
                node.primary_shards.push(shard);
            }
        }

        // Assign replicas (each shard replicated to replication_factor - 1 other nodes)
        let rep_factor = self.replication_config.read().replication_factor as usize;
        if active.len() > 1 && rep_factor > 1 {
            for (i, &shard) in shard_ids.iter().enumerate() {
                let primary_idx = i % active.len();
                for r in 1..rep_factor.min(active.len()) {
                    let replica_idx = (primary_idx + r) % active.len();
                    let node_id = active[replica_idx];
                    if let Some(node) = nodes.get_mut(&node_id) {
                        if !node.replica_shards.contains(&shard) {
                            node.replica_shards.push(shard);
                        }
                    }
                }
            }
        }
    }

    /// Failover: reasigna shards de un nodo caído a nodos activos.
    ///
    /// Retorna los eventos de reasignación.
    pub fn failover(&self, failed_node: ClusterNodeId) -> Vec<ClusterEvent> {
        let mut events = Vec::new();

        let orphan_shards: Vec<ShardId> = {
            let nodes = self.nodes.read();
            nodes
                .get(&failed_node)
                .map(|n| n.primary_shards.clone())
                .unwrap_or_default()
        };

        if orphan_shards.is_empty() {
            return events;
        }

        let mut nodes = self.nodes.write();

        // Find active nodes that have replicas of the orphaned shards
        for shard in &orphan_shards {
            // Find a node that has this shard as replica
            let new_primary = nodes
                .iter()
                .filter(|(id, n)| **id != failed_node && n.is_available())
                .find(|(_, n)| n.replica_shards.contains(shard))
                .map(|(id, _)| *id)
                // Fallback: any active node
                .or_else(|| {
                    nodes
                        .iter()
                        .filter(|(id, n)| **id != failed_node && n.is_available())
                        .min_by_key(|(_, n)| n.primary_shards.len())
                        .map(|(id, _)| *id)
                });

            if let Some(new_id) = new_primary {
                // Promote replica to primary
                if let Some(node) = nodes.get_mut(&new_id) {
                    node.replica_shards.retain(|s| s != shard);
                    node.primary_shards.push(*shard);
                }

                events.push(ClusterEvent::ShardReassigned {
                    shard: *shard,
                    from: failed_node,
                    to: new_id,
                });
            }
        }

        // Clear the failed node's assignments
        if let Some(node) = nodes.get_mut(&failed_node) {
            node.primary_shards.clear();
        }

        drop(nodes);

        for event in &events {
            self.emit_event(event.clone());
        }

        events
    }

    // ── Load Balancing ──────────────────────────────────────────────────

    /// Encuentra el nodo con menor carga.
    pub fn least_loaded_node(&self) -> Option<ClusterNodeId> {
        self.nodes
            .read()
            .iter()
            .filter(|(_, n)| n.is_available())
            .min_by(|(_, a), (_, b)| a.load.partial_cmp(&b.load).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| *id)
    }

    /// Obtiene la distribución de carga por nodo.
    pub fn load_distribution(&self) -> Vec<(ClusterNodeId, f64, usize)> {
        self.nodes
            .read()
            .iter()
            .filter(|(_, n)| n.is_available())
            .map(|(id, n)| (*id, n.load, n.primary_shards.len()))
            .collect()
    }

    /// Verifica si el quorum de lectura se cumple para un shard.
    pub fn has_read_quorum(&self, shard: ShardId) -> bool {
        let config = self.replication_config.read();
        let nodes = self.nodes.read();

        let available_copies: usize = nodes
            .values()
            .filter(|n| {
                n.is_available()
                    && (n.primary_shards.contains(&shard) || n.replica_shards.contains(&shard))
            })
            .count();

        match config.read_consistency {
            ConsistencyLevel::One => available_copies >= 1,
            ConsistencyLevel::Quorum => {
                let total = config.replication_factor as usize;
                available_copies > total / 2
            }
            ConsistencyLevel::All => {
                available_copies >= config.replication_factor as usize
            }
        }
    }

    /// Verifica si el quorum de escritura se cumple para un shard.
    pub fn has_write_quorum(&self, shard: ShardId) -> bool {
        let config = self.replication_config.read();
        let nodes = self.nodes.read();

        let available_copies: usize = nodes
            .values()
            .filter(|n| {
                n.is_available()
                    && (n.primary_shards.contains(&shard) || n.replica_shards.contains(&shard))
            })
            .count();

        match config.write_consistency {
            ConsistencyLevel::One => available_copies >= 1,
            ConsistencyLevel::Quorum => {
                let total = config.replication_factor as usize;
                available_copies > total / 2
            }
            ConsistencyLevel::All => {
                available_copies >= config.replication_factor as usize
            }
        }
    }

    // ── Events ──────────────────────────────────────────────────────────

    /// Registra un listener de eventos del cluster.
    pub fn on_event(&self, callback: ClusterEventCallback) {
        self.event_listeners.write().push(callback);
    }

    /// Emite un evento a todos los listeners.
    fn emit_event(&self, event: ClusterEvent) {
        self.event_log.write().push(event.clone());
        let listeners = self.event_listeners.read();
        for cb in listeners.iter() {
            cb(&event);
        }
    }

    /// Obtiene el log de eventos.
    pub fn event_log(&self) -> Vec<ClusterEvent> {
        self.event_log.read().clone()
    }

    /// Configuración de replicación actual.
    pub fn replication_config(&self) -> ReplicationConfig {
        self.replication_config.read().clone()
    }
}

impl Default for ClusterManager {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_node_first_becomes_leader() {
        let cm = ClusterManager::new();
        cm.add_node(ClusterNodeId(1));

        assert_eq!(cm.leader(), Some(ClusterNodeId(1)));
        assert!(cm.is_leader(ClusterNodeId(1)));
        assert_eq!(cm.node_count(), 1);
        assert_eq!(cm.active_node_count(), 1);
    }

    #[test]
    fn test_add_multiple_nodes() {
        let cm = ClusterManager::new();
        cm.add_node(ClusterNodeId(1));
        cm.add_node(ClusterNodeId(2));
        cm.add_node(ClusterNodeId(3));

        assert_eq!(cm.node_count(), 3);
        assert_eq!(cm.leader(), Some(ClusterNodeId(1))); // First is leader
        assert!(!cm.is_leader(ClusterNodeId(2)));
    }

    #[test]
    fn test_remove_leader_triggers_election() {
        let cm = ClusterManager::new();
        cm.add_node(ClusterNodeId(1));
        cm.add_node(ClusterNodeId(2));

        assert_eq!(cm.leader(), Some(ClusterNodeId(1)));

        cm.remove_node(ClusterNodeId(1));
        // Node 2 should become leader
        assert_eq!(cm.leader(), Some(ClusterNodeId(2)));
        assert_eq!(cm.node_count(), 1);
    }

    #[test]
    fn test_heartbeat_and_health_check() {
        let cm = ClusterManager::with_config(
            ReplicationConfig::default(),
            Duration::from_millis(1), // Very short timeout for testing
            Duration::from_millis(2),
        );
        cm.add_node(ClusterNodeId(1));
        cm.add_node(ClusterNodeId(2));

        // Wait well beyond the down timeout
        std::thread::sleep(Duration::from_millis(20));

        // First check: Active → Suspect
        let _ = cm.check_health();
        // Second check: Suspect → Down (elapsed still > down_timeout)
        let events = cm.check_health();
        // Both nodes should be down
        assert!(!events.is_empty());

        // Heartbeat node 1 → should recover
        cm.heartbeat(ClusterNodeId(1));
        let node = cm.get_node(ClusterNodeId(1)).unwrap();
        assert_eq!(node.status, NodeStatus::Active);
    }

    #[test]
    fn test_elect_leader() {
        let cm = ClusterManager::new();
        cm.add_node(ClusterNodeId(3));
        cm.add_node(ClusterNodeId(1));
        cm.add_node(ClusterNodeId(2));

        // Force re-election
        let leader = cm.elect_leader();
        // Lowest ID wins
        assert_eq!(leader, Some(ClusterNodeId(1)));
        assert_eq!(cm.current_term(), 1);
    }

    #[test]
    fn test_assign_shards_round_robin() {
        let cm = ClusterManager::new();
        cm.add_node(ClusterNodeId(1));
        cm.add_node(ClusterNodeId(2));

        let shards = vec![ShardId(0), ShardId(1), ShardId(2), ShardId(3)];
        cm.assign_shards(&shards);

        let n1 = cm.get_node(ClusterNodeId(1)).unwrap();
        let n2 = cm.get_node(ClusterNodeId(2)).unwrap();

        assert_eq!(n1.primary_shards.len(), 2);
        assert_eq!(n2.primary_shards.len(), 2);
    }

    #[test]
    fn test_assign_shards_with_replicas() {
        let cm = ClusterManager::new();
        cm.add_node(ClusterNodeId(1));
        cm.add_node(ClusterNodeId(2));
        cm.add_node(ClusterNodeId(3));

        let shards = vec![ShardId(0), ShardId(1), ShardId(2)];
        cm.assign_shards(&shards);

        let n1 = cm.get_node(ClusterNodeId(1)).unwrap();
        let n2 = cm.get_node(ClusterNodeId(2)).unwrap();
        let n3 = cm.get_node(ClusterNodeId(3)).unwrap();

        // Each node should have 1 primary
        assert_eq!(n1.primary_shards.len(), 1);
        assert_eq!(n2.primary_shards.len(), 1);
        assert_eq!(n3.primary_shards.len(), 1);

        // Each node should have replicas (replication_factor=3, so 2 replicas each)
        assert!(!n1.replica_shards.is_empty());
        assert!(!n2.replica_shards.is_empty());
        assert!(!n3.replica_shards.is_empty());
    }

    #[test]
    fn test_failover() {
        let cm = ClusterManager::new();
        cm.add_node(ClusterNodeId(1));
        cm.add_node(ClusterNodeId(2));
        cm.add_node(ClusterNodeId(3));

        let shards = vec![ShardId(0), ShardId(1), ShardId(2)];
        cm.assign_shards(&shards);

        // Get node 1's primary shards
        let n1_primaries = cm.get_node(ClusterNodeId(1)).unwrap().primary_shards.clone();
        assert!(!n1_primaries.is_empty());

        // Failover node 1
        let events = cm.failover(ClusterNodeId(1));
        assert!(!events.is_empty());

        // Check that node 1's shards were reassigned
        let n1 = cm.get_node(ClusterNodeId(1)).unwrap();
        assert!(n1.primary_shards.is_empty());

        // Some other node should now have those shards
        let n2 = cm.get_node(ClusterNodeId(2)).unwrap();
        let n3 = cm.get_node(ClusterNodeId(3)).unwrap();
        let total_primaries = n2.primary_shards.len() + n3.primary_shards.len();
        assert_eq!(total_primaries, 3); // All shards accounted for
    }

    #[test]
    fn test_load_balancing() {
        let cm = ClusterManager::new();
        cm.add_node(ClusterNodeId(1));
        cm.add_node(ClusterNodeId(2));
        cm.add_node(ClusterNodeId(3));

        cm.update_load(ClusterNodeId(1), 0.8);
        cm.update_load(ClusterNodeId(2), 0.3);
        cm.update_load(ClusterNodeId(3), 0.5);

        let least = cm.least_loaded_node().unwrap();
        assert_eq!(least, ClusterNodeId(2)); // 0.3 is lowest

        let dist = cm.load_distribution();
        assert_eq!(dist.len(), 3);
    }

    #[test]
    fn test_quorum_with_all_nodes() {
        let cm = ClusterManager::new();
        cm.add_node(ClusterNodeId(1));
        cm.add_node(ClusterNodeId(2));
        cm.add_node(ClusterNodeId(3));

        let shards = vec![ShardId(0)];
        cm.assign_shards(&shards);

        assert!(cm.has_read_quorum(ShardId(0)));
        assert!(cm.has_write_quorum(ShardId(0)));
    }

    #[test]
    fn test_event_log() {
        let cm = ClusterManager::new();
        cm.add_node(ClusterNodeId(1));
        cm.add_node(ClusterNodeId(2));

        let log = cm.event_log();
        assert!(log.len() >= 2); // At least LeaderElected + NodeJoined events
    }

    #[test]
    fn test_cluster_event_listener() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let cm = ClusterManager::new();
        let count = Arc::new(AtomicUsize::new(0));
        let c = count.clone();

        cm.on_event(Box::new(move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        }));

        cm.add_node(ClusterNodeId(1));
        // Should have fired for LeaderElected + NodeJoined
        assert!(count.load(Ordering::Relaxed) >= 2);
    }
}
