// =============================================================================
// bikodb-graph::csr — Compressed Sparse Row (CSR) para traversals OLAP
// =============================================================================
// Representación contigua en memoria para máximo rendimiento en traversals.
//
// ## Layout de memoria
//
//  offsets:  [0, 3, 5, 8, ...]   ← un entry por nodo + 1 sentinel
//  targets:  [2, 5, 7, 1, 4, 0, 3, 6, ...]   ← targets de todos los edges
//
// Los vecinos del nodo i son:  targets[offsets[i] .. offsets[i+1]]
//
// ## Ventajas sobre DashMap + adjacency list:
// - Acceso secuencial a vecinos → cache lines llenas (64B = 8 NodeIds)
// - Zero overhead de hashing: acceso directo por index
// - Prefetch amigable: los targets son contiguos
// - Inmutable → zero locks, ideal para BFS/DFS paralelo
//
// ## Cuándo usar
// - Traversals masivos (BFS, DFS, PageRank, SSSP)
// - Después de ingest masivo: snapshot a CSR, ejecutar algoritmos
// - OLAP workloads (read-heavy, write-rare)
//
// ## Inspiración
// - LDBC Graphalytics: todos los benchmarks usan CSR o variantes
// - Ligra/Galois: frameworks de grafos usan CSR como base
// =============================================================================

use crate::graph::ConcurrentGraph;
use bikodb_core::record::Direction;
use bikodb_core::types::NodeId;

/// Grafo en formato Compressed Sparse Row (CSR).
///
/// Inmutable, contiguo en memoria, óptimo para traversals paralelos.
/// Se construye como snapshot desde un `ConcurrentGraph`.
pub struct CsrGraph {
    /// `offsets[i]` = inicio de vecinos del nodo con internal id `i`.
    /// `offsets[num_nodes]` = total de edges (sentinel).
    offsets: Vec<u32>,
    /// IDs internos de los targets, concatenados por nodo.
    targets: Vec<u32>,
    /// Mapeo NodeId externo → internal id (0-based, denso).
    node_to_internal: Vec<u32>, // indexado por NodeId.0, valor = internal id
    /// Mapeo internal id → NodeId externo.
    internal_to_node: Vec<NodeId>,
    /// Número de nodos.
    num_nodes: usize,
}

impl CsrGraph {
    /// Construye un CSR snapshot desde un ConcurrentGraph.
    ///
    /// Reindexación densa: asigna IDs internos 0..N-1 para acceso por array.
    /// La dirección indica qué edges seguir en los traversals.
    pub fn from_concurrent(graph: &ConcurrentGraph, direction: Direction) -> Self {
        let node_count = graph.node_count();

        // Recopilar todos los NodeId y asignar internal ids
        let mut internal_to_node: Vec<NodeId> = Vec::with_capacity(node_count);
        // Necesitamos saber el NodeId máximo para el array de mapeo
        let mut max_node_id: u64 = 0;

        graph.iter_nodes(|id, _| {
            internal_to_node.push(id);
            if id.0 > max_node_id {
                max_node_id = id.0;
            }
        });

        // Ordenar para determinismo
        internal_to_node.sort_unstable_by_key(|n| n.0);
        let num_nodes = internal_to_node.len();

        // Mapeo externo → interno
        let map_size = (max_node_id as usize) + 1;
        let mut node_to_internal = vec![u32::MAX; map_size];
        for (internal, &ext) in internal_to_node.iter().enumerate() {
            node_to_internal[ext.0 as usize] = internal as u32;
        }

        // Paso 1: contar vecinos por nodo (para offsets)
        let mut offsets = Vec::with_capacity(num_nodes + 1);
        let mut total_edges: u32 = 0;

        for &ext_id in &internal_to_node {
            offsets.push(total_edges);
            if let Ok(neighbors) = graph.neighbors(ext_id, direction) {
                total_edges += neighbors.len() as u32;
            }
        }
        offsets.push(total_edges); // sentinel

        // Paso 2: llenar targets
        let mut targets = Vec::with_capacity(total_edges as usize);
        for &ext_id in &internal_to_node {
            if let Ok(neighbors) = graph.neighbors(ext_id, direction) {
                for neighbor in neighbors {
                    let internal = node_to_internal[neighbor.0 as usize];
                    targets.push(internal);
                }
            }
        }

        Self {
            offsets,
            targets,
            node_to_internal,
            internal_to_node,
            num_nodes,
        }
    }

    /// Vecinos del nodo con internal id dado. Retorna slice contiguo.
    #[inline]
    pub fn neighbors(&self, internal_id: u32) -> &[u32] {
        let start = self.offsets[internal_id as usize] as usize;
        let end = self.offsets[internal_id as usize + 1] as usize;
        &self.targets[start..end]
    }

    /// Convierte NodeId externo a internal id.
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

    /// Convierte internal id a NodeId externo.
    #[inline]
    pub fn to_external(&self, internal_id: u32) -> NodeId {
        self.internal_to_node[internal_id as usize]
    }

    /// Número de nodos.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Número de edges totales.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.targets.len()
    }

    /// Grado del nodo con internal id dado.
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

    /// Helper: crea un grafo lineal A → B → C → D.
    fn linear_graph() -> ConcurrentGraph {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));
        let d = g.insert_node(TypeId(1));
        g.insert_edge(a, b, TypeId(10)).unwrap();
        g.insert_edge(b, c, TypeId(10)).unwrap();
        g.insert_edge(c, d, TypeId(10)).unwrap();
        g
    }

    #[test]
    fn test_csr_from_linear_graph() {
        let g = linear_graph();
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        assert_eq!(csr.num_nodes(), 4);
        assert_eq!(csr.num_edges(), 3);

        // Cada nodo (excepto el último) tiene exactamente 1 vecino out
        let mut degrees: Vec<usize> = (0..4).map(|i| csr.degree(i)).collect();
        degrees.sort();
        assert_eq!(degrees, vec![0, 1, 1, 1]); // d=0, a/b/c=1
    }

    #[test]
    fn test_csr_neighbors_contiguous() {
        let g = ConcurrentGraph::new();
        let center = g.insert_node(TypeId(1));
        let mut leaves = Vec::new();
        for _ in 0..10 {
            let leaf = g.insert_node(TypeId(1));
            g.insert_edge(center, leaf, TypeId(10)).unwrap();
            leaves.push(leaf);
        }

        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let center_internal = csr.to_internal(center).unwrap();
        let neighbors = csr.neighbors(center_internal);

        // Vecinos son contiguos en memoria (slice de array subyacente)
        assert_eq!(neighbors.len(), 10);
        // Verificar que cada vecino mapea a un leaf real
        for &n in neighbors {
            let ext = csr.to_external(n);
            assert!(leaves.contains(&ext));
        }
    }

    #[test]
    fn test_csr_roundtrip_ids() {
        let g = ConcurrentGraph::new();
        let n1 = g.insert_node(TypeId(1));
        let n2 = g.insert_node(TypeId(1));
        g.insert_edge(n1, n2, TypeId(10)).unwrap();

        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let i1 = csr.to_internal(n1).unwrap();
        let i2 = csr.to_internal(n2).unwrap();
        assert_eq!(csr.to_external(i1), n1);
        assert_eq!(csr.to_external(i2), n2);
    }

    #[test]
    fn test_csr_empty_graph() {
        let g = ConcurrentGraph::new();
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        assert_eq!(csr.num_nodes(), 0);
        assert_eq!(csr.num_edges(), 0);
    }

    #[test]
    fn test_csr_bidirectional() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        g.insert_edge(a, b, TypeId(10)).unwrap();

        // Out: solo a→b
        let csr_out = CsrGraph::from_concurrent(&g, Direction::Out);
        let ai = csr_out.to_internal(a).unwrap();
        let bi = csr_out.to_internal(b).unwrap();
        assert_eq!(csr_out.degree(ai), 1);
        assert_eq!(csr_out.degree(bi), 0);

        // In: solo b←a
        let csr_in = CsrGraph::from_concurrent(&g, Direction::In);
        let ai = csr_in.to_internal(a).unwrap();
        let bi = csr_in.to_internal(b).unwrap();
        assert_eq!(csr_in.degree(ai), 0);
        assert_eq!(csr_in.degree(bi), 1);
    }
}
