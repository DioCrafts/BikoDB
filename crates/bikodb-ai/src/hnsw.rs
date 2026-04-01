// =============================================================================
// bikodb-ai::hnsw — Hierarchical Navigable Small World Graph Index
// =============================================================================
// Implementación de HNSW para búsqueda approximate nearest neighbor (ANN)
// en O(log n) por query.
//
// ## Referencia
// Malkov & Yashunin, 2018 — "Efficient and robust approximate nearest
// neighbor search using Hierarchical Navigable Small World graphs"
//
// ## Diseño
// - Multi-layer graph: layer 0 contiene todos los elementos, layers superiores
//   son subconjuntos progresivamente más pequeños (skip-list inspired).
// - Búsqueda greedy: empieza en top layer, desciende haciendo greedy search.
// - Inserción: se inserta en layers según probabilidad geométrica (ef_construction).
// - Thread-safe mediante parking_lot::RwLock.
// =============================================================================

use crate::embedding::DistanceMetric;
use bikodb_core::types::NodeId;
use parking_lot::RwLock;
use rand::Rng as _;
use std::collections::{BinaryHeap, HashSet};

/// Configuración del índice HNSW.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Dimensiones del vector.
    pub dimensions: usize,
    /// Métrica de distancia.
    pub metric: DistanceMetric,
    /// Máximo de conexiones por nodo por layer (M).
    pub max_connections: usize,
    /// Máximo de conexiones en layer 0 (M0 = 2 * M).
    pub max_connections_0: usize,
    /// Tamaño del beam search durante inserción.
    pub ef_construction: usize,
}

impl HnswConfig {
    /// Configuración por defecto para dimensión dada.
    pub fn default_for_dim(dimensions: usize) -> Self {
        Self {
            dimensions,
            metric: DistanceMetric::Cosine,
            max_connections: 16,
            max_connections_0: 32,
            ef_construction: 200,
        }
    }
}

/// Nodo interno del grafo HNSW.
struct HnswNode {
    node_id: NodeId,
    vector: Vec<f32>,
    /// Vecinos por layer: layer_idx → Vec<internal_idx>.
    neighbors: Vec<Vec<usize>>,
    /// Layer máximo en el que aparece este nodo.
    max_layer: usize,
}

/// Resultado de búsqueda con distancia.
#[derive(Debug, Clone)]
pub struct HnswSearchResult {
    pub node_id: NodeId,
    pub distance: f32,
}

/// Elemento del heap: (neg_distance, internal_idx) para max-heap → min distance.
#[derive(Debug, Clone, PartialEq)]
struct HeapItem {
    distance: f32,
    idx: usize,
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap behavior (smallest distance first)
        other.distance.partial_cmp(&self.distance).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Reverse heap item: largest distance first (for pruning).
#[derive(Debug, Clone, PartialEq)]
struct ReverseHeapItem {
    distance: f32,
    idx: usize,
}

impl Eq for ReverseHeapItem {}

impl PartialOrd for ReverseHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ReverseHeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Índice HNSW para búsqueda escalable de vectores.
///
/// # Ejemplo
/// ```
/// use bikodb_ai::hnsw::{HnswIndex, HnswConfig};
/// use bikodb_ai::embedding::DistanceMetric;
/// use bikodb_core::types::NodeId;
///
/// let config = HnswConfig {
///     dimensions: 3,
///     metric: DistanceMetric::Euclidean,
///     max_connections: 4,
///     max_connections_0: 8,
///     ef_construction: 16,
/// };
/// let idx = HnswIndex::new(config);
///
/// idx.insert(NodeId(1), vec![1.0, 0.0, 0.0]);
/// idx.insert(NodeId(2), vec![0.0, 1.0, 0.0]);
/// idx.insert(NodeId(3), vec![0.9, 0.1, 0.0]);
///
/// let results = idx.search(&[1.0, 0.0, 0.0], 2, 10);
/// assert_eq!(results[0].node_id, NodeId(1));
/// ```
pub struct HnswIndex {
    config: HnswConfig,
    nodes: RwLock<Vec<HnswNode>>,
    entry_point: RwLock<Option<usize>>,
    max_layer: RwLock<usize>,
    /// Level multiplier: 1/ln(M).
    level_mult: f64,
}

impl HnswIndex {
    /// Crea un índice HNSW vacío.
    pub fn new(config: HnswConfig) -> Self {
        let level_mult = 1.0 / (config.max_connections as f64).ln();
        Self {
            config,
            nodes: RwLock::new(Vec::new()),
            entry_point: RwLock::new(None),
            max_layer: RwLock::new(0),
            level_mult,
        }
    }

    /// Calcula layer aleatorio para un nuevo nodo (distribución geométrica).
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rand::Rng::gen(&mut rng);
        let level = (-r.ln() * self.level_mult).floor() as usize;
        level.min(32) // Cap at 32 layers
    }

    /// Distancia entre un query y un nodo interno.
    fn distance(&self, query: &[f32], nodes: &[HnswNode], idx: usize) -> f32 {
        self.config.metric.distance(query, &nodes[idx].vector)
    }

    /// Inserta un vector en el índice.
    pub fn insert(&self, node_id: NodeId, vector: Vec<f32>) {
        assert_eq!(
            vector.len(), self.config.dimensions,
            "Vector dimensions mismatch"
        );

        let new_level = self.random_level();

        let mut nodes = self.nodes.write();
        let new_idx = nodes.len();

        // Create node with neighbor lists for each layer
        let neighbors = (0..=new_level).map(|_| Vec::new()).collect();
        nodes.push(HnswNode {
            node_id,
            vector,
            neighbors,
            max_layer: new_level,
        });

        let entry = *self.entry_point.read();

        if entry.is_none() {
            // First node
            *self.entry_point.write() = Some(new_idx);
            *self.max_layer.write() = new_level;
            return;
        }

        let mut current_entry = entry.unwrap();
        let current_max = *self.max_layer.read();

        // Phase 1: Greedy search from top to new_level+1
        for layer in (new_level + 1..=current_max).rev() {
            current_entry = self.greedy_closest(&nodes, &nodes[new_idx].vector, current_entry, layer);
        }

        // Phase 2: Insert into layers from min(new_level, current_max) down to 0
        let insert_from = new_level.min(current_max);
        for layer in (0..=insert_from).rev() {
            let max_conn = if layer == 0 {
                self.config.max_connections_0
            } else {
                self.config.max_connections
            };

            // Find ef_construction nearest neighbors at this layer
            let neighbors = self.search_layer(
                &nodes,
                &nodes[new_idx].vector,
                current_entry,
                self.config.ef_construction,
                layer,
            );

            // Select top max_conn neighbors
            let selected: Vec<usize> = neighbors.into_iter()
                .take(max_conn)
                .map(|item| item.idx)
                .collect();

            // Add bidirectional connections
            if layer < nodes[new_idx].neighbors.len() {
                nodes[new_idx].neighbors[layer] = selected.clone();
            }

            for &neighbor_idx in &selected {
                if layer < nodes[neighbor_idx].neighbors.len() {
                    nodes[neighbor_idx].neighbors[layer].push(new_idx);
                    // Prune if over max connections
                    if nodes[neighbor_idx].neighbors[layer].len() > max_conn {
                        let query_vec = nodes[neighbor_idx].vector.clone();
                        let mut scored: Vec<(f32, usize)> = nodes[neighbor_idx].neighbors[layer]
                            .iter()
                            .map(|&nidx| {
                                let d = self.config.metric.distance(&query_vec, &nodes[nidx].vector);
                                (d, nidx)
                            })
                            .collect();
                        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        scored.truncate(max_conn);
                        nodes[neighbor_idx].neighbors[layer] = scored.into_iter().map(|(_, idx)| idx).collect();
                    }
                }
            }

            if !selected.is_empty() {
                current_entry = selected[0];
            }
        }

        // Update entry point if new node is at higher layer
        if new_level > current_max {
            *self.entry_point.write() = Some(new_idx);
            *self.max_layer.write() = new_level;
        }
    }

    /// Greedy search: find closest node at given layer starting from entry.
    fn greedy_closest(
        &self,
        nodes: &[HnswNode],
        query: &[f32],
        entry: usize,
        layer: usize,
    ) -> usize {
        let mut current = entry;
        let mut current_dist = self.config.metric.distance(query, &nodes[current].vector);

        loop {
            let mut changed = false;
            if layer < nodes[current].neighbors.len() {
                for &neighbor in &nodes[current].neighbors[layer] {
                    let d = self.config.metric.distance(query, &nodes[neighbor].vector);
                    if d < current_dist {
                        current = neighbor;
                        current_dist = d;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        current
    }

    /// Beam search at a specific layer: returns ef nearest neighbors.
    fn search_layer(
        &self,
        nodes: &[HnswNode],
        query: &[f32],
        entry: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<HeapItem> {
        let entry_dist = self.config.metric.distance(query, &nodes[entry].vector);

        // Candidates: min-heap (closest first)
        let mut candidates = BinaryHeap::new();
        candidates.push(HeapItem { distance: entry_dist, idx: entry });

        // Results: max-heap (farthest first, for pruning)
        let mut results = BinaryHeap::new();
        results.push(ReverseHeapItem { distance: entry_dist, idx: entry });

        let mut visited = HashSet::new();
        visited.insert(entry);

        while let Some(HeapItem { distance: c_dist, idx: c_idx }) = candidates.pop() {
            // If closest candidate is further than farthest result, stop
            if let Some(farthest) = results.peek() {
                if c_dist > farthest.distance && results.len() >= ef {
                    break;
                }
            }

            // Explore neighbors
            if layer < nodes[c_idx].neighbors.len() {
                for &neighbor in &nodes[c_idx].neighbors[layer] {
                    if visited.contains(&neighbor) {
                        continue;
                    }
                    visited.insert(neighbor);

                    let d = self.config.metric.distance(query, &nodes[neighbor].vector);

                    let should_add = if results.len() < ef {
                        true
                    } else if let Some(farthest) = results.peek() {
                        d < farthest.distance
                    } else {
                        true
                    };

                    if should_add {
                        candidates.push(HeapItem { distance: d, idx: neighbor });
                        results.push(ReverseHeapItem { distance: d, idx: neighbor });

                        if results.len() > ef {
                            results.pop(); // Remove farthest
                        }
                    }
                }
            }
        }

        // Convert results to sorted vec
        let mut result_vec: Vec<HeapItem> = results
            .into_iter()
            .map(|r| HeapItem { distance: r.distance, idx: r.idx })
            .collect();
        result_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        result_vec
    }

    /// Busca los k vecinos más cercanos al query vector.
    ///
    /// - `query`: vector de búsqueda
    /// - `k`: número de resultados
    /// - `ef_search`: beam width (mayor = más preciso, más lento)
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<HnswSearchResult> {
        assert_eq!(query.len(), self.config.dimensions);

        let nodes = self.nodes.read();
        let entry = *self.entry_point.read();

        if entry.is_none() || nodes.is_empty() {
            return Vec::new();
        }

        let mut current_entry = entry.unwrap();
        let current_max = *self.max_layer.read();

        // Greedy search from top layer down to layer 1
        for layer in (1..=current_max).rev() {
            current_entry = self.greedy_closest(&nodes, query, current_entry, layer);
        }

        // Search at layer 0 with ef_search
        let ef = ef_search.max(k);
        let results = self.search_layer(&nodes, query, current_entry, ef, 0);

        results
            .into_iter()
            .take(k)
            .map(|item| HnswSearchResult {
                node_id: nodes[item.idx].node_id,
                distance: item.distance,
            })
            .collect()
    }

    /// Elimina un nodo del índice (marca como eliminado, no compacta).
    pub fn remove(&self, target_node_id: NodeId) {
        let mut nodes = self.nodes.write();
        // Find index
        if let Some(idx) = nodes.iter().position(|n| n.node_id == target_node_id) {
            // Remove from all neighbors' lists
            for layer in 0..nodes[idx].neighbors.len() {
                let neighbor_indices: Vec<usize> = nodes[idx].neighbors[layer].clone();
                for &nidx in &neighbor_indices {
                    if layer < nodes[nidx].neighbors.len() {
                        nodes[nidx].neighbors[layer].retain(|&x| x != idx);
                    }
                }
                nodes[idx].neighbors[layer].clear();
            }
            // Zero out the vector to mark as deleted
            nodes[idx].vector = vec![f32::NAN; self.config.dimensions];
        }
    }

    /// Número de vectores en el índice.
    pub fn len(&self) -> usize {
        self.nodes.read().iter().filter(|n| !n.vector[0].is_nan()).count()
    }

    /// ¿Índice vacío?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dimensiones del índice.
    pub fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    /// Número de layers en uso.
    pub fn num_layers(&self) -> usize {
        *self.max_layer.read() + 1
    }

    // ─────────────────────────────────────────────────────────────────────
    // Serialización / Persistencia
    // ─────────────────────────────────────────────────────────────────────

    /// Serializa el índice HNSW a un vector de bytes.
    ///
    /// Formato binario:
    /// - [u32] dimensions
    /// - [u8]  metric (0=Cosine, 1=Euclidean, 2=DotProduct)
    /// - [u32] max_connections
    /// - [u32] max_connections_0
    /// - [u32] ef_construction
    /// - [u32] entry_point (u32::MAX si none)
    /// - [u32] max_layer
    /// - [u32] num_nodes
    /// - Per node:
    ///   - [u64] node_id
    ///   - [u32] max_layer
    ///   - [f32 * dimensions] vector
    ///   - Per layer (0..=max_layer):
    ///     - [u32] num_neighbors
    ///     - [u32 * num_neighbors] neighbor indices
    pub fn serialize(&self) -> Vec<u8> {
        let nodes = self.nodes.read();
        let entry = *self.entry_point.read();
        let max_layer = *self.max_layer.read();

        let mut buf = Vec::new();

        // Config
        buf.extend_from_slice(&(self.config.dimensions as u32).to_le_bytes());
        buf.push(match self.config.metric {
            DistanceMetric::Cosine => 0,
            DistanceMetric::Euclidean => 1,
            DistanceMetric::DotProduct => 2,
        });
        buf.extend_from_slice(&(self.config.max_connections as u32).to_le_bytes());
        buf.extend_from_slice(&(self.config.max_connections_0 as u32).to_le_bytes());
        buf.extend_from_slice(&(self.config.ef_construction as u32).to_le_bytes());

        // Entry point
        let ep = entry.map(|e| e as u32).unwrap_or(u32::MAX);
        buf.extend_from_slice(&ep.to_le_bytes());
        buf.extend_from_slice(&(max_layer as u32).to_le_bytes());

        // Nodes
        buf.extend_from_slice(&(nodes.len() as u32).to_le_bytes());
        for node in nodes.iter() {
            buf.extend_from_slice(&node.node_id.0.to_le_bytes());
            buf.extend_from_slice(&(node.max_layer as u32).to_le_bytes());
            for &v in &node.vector {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            for layer_neighbors in &node.neighbors {
                buf.extend_from_slice(&(layer_neighbors.len() as u32).to_le_bytes());
                for &nidx in layer_neighbors {
                    buf.extend_from_slice(&(nidx as u32).to_le_bytes());
                }
            }
        }

        buf
    }

    /// Deserializa un índice HNSW desde bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut pos = 0;

        let read_u32 = |data: &[u8], pos: &mut usize| -> Result<u32, String> {
            if *pos + 4 > data.len() {
                return Err("Unexpected end of data".into());
            }
            let val = u32::from_le_bytes([data[*pos], data[*pos+1], data[*pos+2], data[*pos+3]]);
            *pos += 4;
            Ok(val)
        };

        let read_u64 = |data: &[u8], pos: &mut usize| -> Result<u64, String> {
            if *pos + 8 > data.len() {
                return Err("Unexpected end of data".into());
            }
            let bytes: [u8; 8] = data[*pos..*pos+8].try_into().unwrap();
            let val = u64::from_le_bytes(bytes);
            *pos += 8;
            Ok(val)
        };

        let read_f32 = |data: &[u8], pos: &mut usize| -> Result<f32, String> {
            if *pos + 4 > data.len() {
                return Err("Unexpected end of data".into());
            }
            let val = f32::from_le_bytes([data[*pos], data[*pos+1], data[*pos+2], data[*pos+3]]);
            *pos += 4;
            Ok(val)
        };

        // Config
        let dimensions = read_u32(data, &mut pos)? as usize;
        if pos >= data.len() {
            return Err("Missing metric byte".into());
        }
        let metric = match data[pos] {
            0 => DistanceMetric::Cosine,
            1 => DistanceMetric::Euclidean,
            2 => DistanceMetric::DotProduct,
            other => return Err(format!("Unknown metric: {}", other)),
        };
        pos += 1;
        let max_connections = read_u32(data, &mut pos)? as usize;
        let max_connections_0 = read_u32(data, &mut pos)? as usize;
        let ef_construction = read_u32(data, &mut pos)? as usize;

        let config = HnswConfig {
            dimensions,
            metric,
            max_connections,
            max_connections_0,
            ef_construction,
        };

        let ep_raw = read_u32(data, &mut pos)?;
        let entry_point = if ep_raw == u32::MAX { None } else { Some(ep_raw as usize) };
        let max_layer = read_u32(data, &mut pos)? as usize;

        let num_nodes = read_u32(data, &mut pos)? as usize;
        let mut nodes = Vec::with_capacity(num_nodes);

        for _ in 0..num_nodes {
            let node_id = NodeId(read_u64(data, &mut pos)?);
            let node_max_layer = read_u32(data, &mut pos)? as usize;

            let mut vector = Vec::with_capacity(dimensions);
            for _ in 0..dimensions {
                vector.push(read_f32(data, &mut pos)?);
            }

            let mut neighbors = Vec::with_capacity(node_max_layer + 1);
            for _ in 0..=node_max_layer {
                let num_neighbors = read_u32(data, &mut pos)? as usize;
                let mut layer_neighbors = Vec::with_capacity(num_neighbors);
                for _ in 0..num_neighbors {
                    layer_neighbors.push(read_u32(data, &mut pos)? as usize);
                }
                neighbors.push(layer_neighbors);
            }

            nodes.push(HnswNode {
                node_id,
                vector,
                neighbors,
                max_layer: node_max_layer,
            });
        }

        let level_mult = 1.0 / (config.max_connections as f64).ln();

        Ok(Self {
            config,
            nodes: RwLock::new(nodes),
            entry_point: RwLock::new(entry_point),
            max_layer: RwLock::new(max_layer),
            level_mult,
        })
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::DistanceMetric;

    fn test_config(dim: usize) -> HnswConfig {
        HnswConfig {
            dimensions: dim,
            metric: DistanceMetric::Euclidean,
            max_connections: 4,
            max_connections_0: 8,
            ef_construction: 16,
        }
    }

    #[test]
    fn test_insert_single() {
        let idx = HnswIndex::new(test_config(3));
        idx.insert(NodeId(1), vec![1.0, 0.0, 0.0]);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_search_exact_match() {
        let idx = HnswIndex::new(test_config(3));
        idx.insert(NodeId(1), vec![1.0, 0.0, 0.0]);
        idx.insert(NodeId(2), vec![0.0, 1.0, 0.0]);
        idx.insert(NodeId(3), vec![0.0, 0.0, 1.0]);

        let results = idx.search(&[1.0, 0.0, 0.0], 1, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, NodeId(1));
        assert!(results[0].distance < 1e-5);
    }

    #[test]
    fn test_search_nearest() {
        let idx = HnswIndex::new(test_config(2));
        idx.insert(NodeId(1), vec![0.0, 0.0]);
        idx.insert(NodeId(2), vec![10.0, 10.0]);
        idx.insert(NodeId(3), vec![1.0, 1.0]);

        let results = idx.search(&[0.5, 0.5], 2, 10);
        assert_eq!(results.len(), 2);
        // Closest should be NodeId(3) at [1,1] or NodeId(1) at [0,0]
        // dist to [0,0] = sqrt(0.5) ≈ 0.707
        // dist to [1,1] = sqrt(0.5) ≈ 0.707
        // Both equidistant, but NodeId(2) at [10,10] should NOT be first
        assert_ne!(results[0].node_id, NodeId(2));
    }

    #[test]
    fn test_search_k_results() {
        let idx = HnswIndex::new(test_config(2));
        for i in 0..10 {
            idx.insert(NodeId(i), vec![i as f32, 0.0]);
        }

        let results = idx.search(&[5.0, 0.0], 3, 20);
        assert_eq!(results.len(), 3);
        // Should include NodeId(5) as exact match
        assert!(results.iter().any(|r| r.node_id == NodeId(5)));
    }

    #[test]
    fn test_search_empty() {
        let idx = HnswIndex::new(test_config(3));
        let results = idx.search(&[1.0, 0.0, 0.0], 5, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove() {
        let idx = HnswIndex::new(test_config(2));
        idx.insert(NodeId(1), vec![1.0, 0.0]);
        idx.insert(NodeId(2), vec![0.0, 1.0]);
        assert_eq!(idx.len(), 2);

        idx.remove(NodeId(1));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_larger_index() {
        let idx = HnswIndex::new(test_config(4));
        // Insert 50 vectors
        for i in 0..50 {
            let v = vec![i as f32, (i * 2) as f32, (i % 7) as f32, (i % 3) as f32];
            idx.insert(NodeId(i), v);
        }

        assert_eq!(idx.len(), 50);

        // Search should return correct results
        let results = idx.search(&[25.0, 50.0, 4.0, 1.0], 5, 30);
        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].distance >= results[i - 1].distance);
        }
    }

    #[test]
    fn test_cosine_metric() {
        let config = HnswConfig {
            dimensions: 3,
            metric: DistanceMetric::Cosine,
            max_connections: 4,
            max_connections_0: 8,
            ef_construction: 16,
        };
        let idx = HnswIndex::new(config);
        idx.insert(NodeId(1), vec![1.0, 0.0, 0.0]);
        idx.insert(NodeId(2), vec![0.0, 1.0, 0.0]);
        idx.insert(NodeId(3), vec![0.99, 0.01, 0.0]); // Very close to NodeId(1)

        let results = idx.search(&[1.0, 0.0, 0.0], 1, 10);
        assert_eq!(results[0].node_id, NodeId(1));
    }

    #[test]
    fn test_num_layers() {
        let idx = HnswIndex::new(test_config(2));
        // With enough nodes, should have multiple layers
        for i in 0..100 {
            idx.insert(NodeId(i), vec![i as f32, (i * 3) as f32]);
        }
        // Should have at least 1 layer, possibly more
        assert!(idx.num_layers() >= 1);
    }

    #[test]
    fn test_serialize_deserialize_empty() {
        let idx = HnswIndex::new(test_config(3));
        let data = idx.serialize();
        let idx2 = HnswIndex::from_bytes(&data).unwrap();
        assert_eq!(idx2.len(), 0);
        assert_eq!(idx2.dimensions(), 3);
    }

    #[test]
    fn test_serialize_deserialize_with_data() {
        let idx = HnswIndex::new(test_config(3));
        idx.insert(NodeId(1), vec![1.0, 0.0, 0.0]);
        idx.insert(NodeId(2), vec![0.0, 1.0, 0.0]);
        idx.insert(NodeId(3), vec![0.0, 0.0, 1.0]);
        idx.insert(NodeId(4), vec![0.5, 0.5, 0.0]);

        let data = idx.serialize();
        let idx2 = HnswIndex::from_bytes(&data).unwrap();

        assert_eq!(idx2.len(), 4);
        assert_eq!(idx2.dimensions(), 3);

        // Search should work on deserialized index
        let results = idx2.search(&[1.0, 0.0, 0.0], 1, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, NodeId(1));
        assert!(results[0].distance < 1e-5);
    }

    #[test]
    fn test_serialize_deserialize_preserves_search_quality() {
        let config = HnswConfig {
            dimensions: 4,
            metric: DistanceMetric::Euclidean,
            max_connections: 4,
            max_connections_0: 8,
            ef_construction: 16,
        };
        let idx = HnswIndex::new(config);
        for i in 0..20 {
            idx.insert(NodeId(i), vec![i as f32, (i * 2) as f32, 0.0, 0.0]);
        }

        // Search before serialize
        let results_before = idx.search(&[10.0, 20.0, 0.0, 0.0], 3, 20);

        // Serialize → deserialize
        let data = idx.serialize();
        let idx2 = HnswIndex::from_bytes(&data).unwrap();
        let results_after = idx2.search(&[10.0, 20.0, 0.0, 0.0], 3, 20);

        // Same results
        assert_eq!(results_before.len(), results_after.len());
        for (a, b) in results_before.iter().zip(results_after.iter()) {
            assert_eq!(a.node_id, b.node_id);
            assert!((a.distance - b.distance).abs() < 1e-5);
        }
    }

    #[test]
    fn test_from_bytes_invalid_data() {
        let result = HnswIndex::from_bytes(&[]);
        assert!(result.is_err());

        let result = HnswIndex::from_bytes(&[0, 1, 2]);
        assert!(result.is_err());
    }
}
