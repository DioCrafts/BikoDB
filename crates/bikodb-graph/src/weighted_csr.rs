// =============================================================================
// bikodb-graph::weighted_csr — CSR con pesos de aristas para SSSP
// =============================================================================
// Extiende el CSR base con un array contiguo de pesos `weights: Vec<f64>`.
//
// ## Layout de memoria
//
//  offsets:  [0, 3, 5, 8, ...]           ← un entry por nodo + sentinel
//  targets:  [2, 5, 7, 1, 4, 0, 3, 6, ...]   ← targets de edges
//  weights:  [1.2, 0.5, 3.0, ...]        ← peso de cada edge, alineado con targets
//
// Los vecinos del nodo i con sus pesos son:
//   for j in offsets[i]..offsets[i+1]:
//       (targets[j], weights[j])
//
// ## Extracción de pesos
// El peso se extrae de `EdgeData.properties` buscando una property key
// configurable (default: prop_id 0). Si no existe, se usa peso 1.0.
//
// ## Detección de pesos negativos
// Durante la construcción se detecta si hay pesos negativos, lo que
// determina la selección del algoritmo SSSP (Dijkstra vs Bellman-Ford).
// =============================================================================

use crate::graph::ConcurrentGraph;
use bikodb_core::record::Direction;
use bikodb_core::types::NodeId;
use bikodb_core::value::Value;

/// CSR con pesos de aristas. Inmutable, contiguo, óptimo para SSSP.
pub struct WeightedCsrGraph {
    /// `offsets[i]` = inicio de vecinos del nodo con internal id `i`.
    offsets: Vec<u32>,
    /// IDs internos de los targets, concatenados por nodo.
    targets: Vec<u32>,
    /// Peso de cada edge, alineado con `targets`.
    weights: Vec<f64>,
    /// Mapeo NodeId externo → internal id.
    node_to_internal: Vec<u32>,
    /// Mapeo internal id → NodeId externo.
    internal_to_node: Vec<NodeId>,
    /// Número de nodos.
    num_nodes: usize,
    /// ¿Hay algún peso negativo?
    has_negative_weights: bool,
    /// ¿Todos los pesos son 1.0? (unweighted)
    all_unit_weights: bool,
}

impl WeightedCsrGraph {
    /// Construye un WeightedCsrGraph desde un ConcurrentGraph.
    ///
    /// `weight_prop_id`: property key del edge que contiene el peso.
    /// Si un edge no tiene la property, se asigna peso `default_weight`.
    pub fn from_concurrent(
        graph: &ConcurrentGraph,
        direction: Direction,
        weight_prop_id: u16,
        default_weight: f64,
    ) -> Self {
        let node_count = graph.node_count();

        // Recopilar NodeIds y crear mapeo denso
        let mut internal_to_node: Vec<NodeId> = Vec::with_capacity(node_count);
        let mut max_node_id: u64 = 0;

        graph.iter_nodes(|id, _| {
            internal_to_node.push(id);
            if id.0 > max_node_id {
                max_node_id = id.0;
            }
        });

        internal_to_node.sort_unstable_by_key(|n| n.0);
        let num_nodes = internal_to_node.len();

        let map_size = if max_node_id > 0 {
            (max_node_id as usize) + 1
        } else if num_nodes > 0 {
            1
        } else {
            0
        };
        let mut node_to_internal = vec![u32::MAX; map_size];
        for (internal, &ext) in internal_to_node.iter().enumerate() {
            node_to_internal[ext.0 as usize] = internal as u32;
        }

        // Paso 1: contar edges para offsets
        let mut offsets = Vec::with_capacity(num_nodes + 1);
        let mut total_edges: u32 = 0;

        for &ext_id in &internal_to_node {
            offsets.push(total_edges);
            if let Ok(edges) = graph.edges_of(ext_id, direction) {
                total_edges += edges.len() as u32;
            }
        }
        offsets.push(total_edges);

        // Paso 2: llenar targets y weights
        let mut targets = Vec::with_capacity(total_edges as usize);
        let mut weights = Vec::with_capacity(total_edges as usize);
        let mut has_negative = false;
        let mut all_unit = true;

        for &ext_id in &internal_to_node {
            if let Ok(edges) = graph.edges_of(ext_id, direction) {
                for edge in edges {
                    let neighbor = if direction == Direction::In {
                        edge.source
                    } else {
                        edge.target
                    };
                    let internal = node_to_internal[neighbor.0 as usize];
                    targets.push(internal);

                    // Extraer peso de properties
                    let w = edge
                        .properties
                        .iter()
                        .find(|(k, _)| *k == weight_prop_id)
                        .and_then(|(_, v)| match v {
                            Value::Float(f) => Some(*f),
                            Value::Int(i) => Some(*i as f64),
                            _ => None,
                        })
                        .unwrap_or(default_weight);

                    if w < 0.0 {
                        has_negative = true;
                    }
                    if (w - 1.0).abs() > f64::EPSILON {
                        all_unit = false;
                    }

                    weights.push(w);
                }
            }
        }

        Self {
            offsets,
            targets,
            weights,
            node_to_internal,
            internal_to_node,
            num_nodes,
            has_negative_weights: has_negative,
            all_unit_weights: all_unit,
        }
    }

    /// Construye un WeightedCsrGraph directamente desde arrays (para tests).
    pub fn from_raw(
        offsets: Vec<u32>,
        targets: Vec<u32>,
        weights: Vec<f64>,
        internal_to_node: Vec<NodeId>,
        node_to_internal: Vec<u32>,
    ) -> Self {
        let num_nodes = internal_to_node.len();
        let has_negative = weights.iter().any(|&w| w < 0.0);
        let all_unit = weights.iter().all(|&w| (w - 1.0).abs() <= f64::EPSILON);
        Self {
            offsets,
            targets,
            weights,
            node_to_internal,
            internal_to_node,
            num_nodes,
            has_negative_weights: has_negative,
            all_unit_weights: all_unit,
        }
    }

    /// Vecinos del nodo con internal id dado, con pesos.
    #[inline]
    pub fn weighted_neighbors(&self, internal_id: u32) -> impl Iterator<Item = (u32, f64)> + '_ {
        let start = self.offsets[internal_id as usize] as usize;
        let end = self.offsets[internal_id as usize + 1] as usize;
        self.targets[start..end]
            .iter()
            .zip(&self.weights[start..end])
            .map(|(&t, &w)| (t, w))
    }

    /// Vecinos (solo targets, sin pesos).
    #[inline]
    pub fn neighbors(&self, internal_id: u32) -> &[u32] {
        let start = self.offsets[internal_id as usize] as usize;
        let end = self.offsets[internal_id as usize + 1] as usize;
        &self.targets[start..end]
    }

    #[inline]
    pub fn to_internal(&self, node_id: NodeId) -> Option<u32> {
        let idx = node_id.0 as usize;
        if idx < self.node_to_internal.len() {
            let internal = self.node_to_internal[idx];
            if internal != u32::MAX {
                return Some(internal);
            }
        }
        None
    }

    #[inline]
    pub fn to_external(&self, internal_id: u32) -> NodeId {
        self.internal_to_node[internal_id as usize]
    }

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    #[inline]
    pub fn num_edges(&self) -> usize {
        self.targets.len()
    }

    #[inline]
    pub fn has_negative_weights(&self) -> bool {
        self.has_negative_weights
    }

    #[inline]
    pub fn all_unit_weights(&self) -> bool {
        self.all_unit_weights
    }

    #[inline]
    pub fn degree(&self, internal_id: u32) -> usize {
        let start = self.offsets[internal_id as usize] as usize;
        let end = self.offsets[internal_id as usize + 1] as usize;
        end - start
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bikodb_core::types::TypeId;

    #[test]
    fn test_weighted_csr_construction() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));

        // a→b weight=2.5, a→c weight=1.0 (default)
        g.insert_edge_with_props(a, b, TypeId(10), vec![(0, Value::Float(2.5))])
            .unwrap();
        g.insert_edge(a, c, TypeId(10)).unwrap();

        let wcsr = WeightedCsrGraph::from_concurrent(&g, Direction::Out, 0, 1.0);
        assert_eq!(wcsr.num_nodes(), 3);
        assert_eq!(wcsr.num_edges(), 2);
        assert!(!wcsr.has_negative_weights());
        assert!(!wcsr.all_unit_weights());
    }

    #[test]
    fn test_weighted_neighbors() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));

        g.insert_edge_with_props(a, b, TypeId(10), vec![(0, Value::Float(3.0))])
            .unwrap();
        g.insert_edge_with_props(a, c, TypeId(10), vec![(0, Value::Float(1.5))])
            .unwrap();

        let wcsr = WeightedCsrGraph::from_concurrent(&g, Direction::Out, 0, 1.0);
        let a_int = wcsr.to_internal(a).unwrap();

        let neighbors: Vec<(u32, f64)> = wcsr.weighted_neighbors(a_int).collect();
        assert_eq!(neighbors.len(), 2);

        // Verify weights present (order may vary)
        let total_weight: f64 = neighbors.iter().map(|(_, w)| w).sum();
        assert!((total_weight - 4.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_negative_weight_detection() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));

        g.insert_edge_with_props(a, b, TypeId(10), vec![(0, Value::Float(-2.0))])
            .unwrap();

        let wcsr = WeightedCsrGraph::from_concurrent(&g, Direction::Out, 0, 1.0);
        assert!(wcsr.has_negative_weights());
    }

    #[test]
    fn test_all_unit_weights() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));

        g.insert_edge(a, b, TypeId(10)).unwrap();
        g.insert_edge(b, c, TypeId(10)).unwrap();

        let wcsr = WeightedCsrGraph::from_concurrent(&g, Direction::Out, 0, 1.0);
        assert!(wcsr.all_unit_weights());
        assert!(!wcsr.has_negative_weights());
    }

    #[test]
    fn test_int_weight_extraction() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));

        // Int weight should be converted to f64
        g.insert_edge_with_props(a, b, TypeId(10), vec![(0, Value::Int(5))])
            .unwrap();

        let wcsr = WeightedCsrGraph::from_concurrent(&g, Direction::Out, 0, 1.0);
        let a_int = wcsr.to_internal(a).unwrap();
        let neighbors: Vec<(u32, f64)> = wcsr.weighted_neighbors(a_int).collect();
        assert_eq!(neighbors.len(), 1);
        assert!((neighbors[0].1 - 5.0).abs() < f64::EPSILON);
    }
}
