// =============================================================================
// bikodb-graph::lcc — Local Clustering Coefficient paralelo sobre CSR
// =============================================================================
//
// ## Algoritmo: LDBC Graphalytics LCC
//
// El Local Clustering Coefficient (LCC) mide cuán conectados están los
// vecinos de un nodo entre sí. Para un nodo v con grado d(v):
//
//   LCC(v) = 2 × T(v) / (d(v) × (d(v) - 1))
//
// Donde T(v) es el número de triángulos que contienen a v (es decir,
// pares de vecinos de v que están conectados entre sí).
//
// Para nodos con grado 0 o 1, LCC = 0.0 (no hay posibles triángulos).
//
// ## Implementación
//
// Para grafos no dirigidos (LDBC standard):
// - Se construye CSR bidireccional (undirected)
// - Para cada nodo v, se itera sobre pares de vecinos (u, w) y se verifica
//   si existe edge u→w usando búsqueda binaria en el CSR
// - Optimización: los vecinos en CSR están ordenados → binary search O(log d)
//
// ## Paralelización
//
// - rayon par_iter sobre nodos
// - Cada nodo calcula su LCC independientemente → zero contention
// - Chunk size adaptativo según número de nodos
//
// ## Complejidad
//
// O(Σ d(v)² / P) donde d(v) es el grado del nodo v.
// En grafos con grado máximo acotado: O(V × d_max² / P).
// =============================================================================

use crate::csr::CsrGraph;
use crate::graph::ConcurrentGraph;
use bikodb_core::record::Direction;
use bikodb_core::types::NodeId;
use rayon::prelude::*;

// =============================================================================
// Resultado
// =============================================================================

/// Resultado de LCC: coeficientes por nodo.
#[derive(Debug)]
pub struct LccResult {
    /// Coeficiente de clustering por nodo (indexado por internal id).
    pub coefficients: Vec<f64>,
    /// Promedio global (Global Clustering Coefficient).
    pub average: f64,
    /// Número de nodos procesados.
    pub num_nodes: usize,
}

// =============================================================================
// API pública
// =============================================================================

/// Calcula LCC sobre un `ConcurrentGraph` (construye CSR internamente).
///
/// Para LDBC Graphalytics, el grafo se trata como no dirigido:
/// ambas direcciones se incluyen en el CSR.
///
/// # Ejemplo
/// ```
/// use bikodb_graph::ConcurrentGraph;
/// use bikodb_graph::lcc::lcc;
/// use bikodb_core::types::{NodeId, TypeId};
///
/// let g = ConcurrentGraph::new();
/// let a = g.insert_node(TypeId(1));
/// let b = g.insert_node(TypeId(1));
/// let c = g.insert_node(TypeId(1));
/// g.insert_edge(a, b, TypeId(10)).unwrap();
/// g.insert_edge(b, c, TypeId(10)).unwrap();
/// g.insert_edge(c, a, TypeId(10)).unwrap();
///
/// let result = lcc(&g);
/// // Triángulo completo: cada nodo tiene LCC = 1.0
/// assert!((result.average - 1.0).abs() < 1e-9);
/// ```
pub fn lcc(graph: &ConcurrentGraph) -> LccResult {
    // Para undirected LCC, necesitamos un CSR simétrico.
    // Construimos Out-CSR; como insert_edge es dirigido, el usuario
    // debe insertar edges en ambas direcciones para undirected, o
    // usamos ambos Out e In para simular.
    let out_csr = CsrGraph::from_concurrent(graph, Direction::Out);
    let in_csr = CsrGraph::from_concurrent(graph, Direction::In);
    lcc_on_csr_undirected(&out_csr, &in_csr)
}

/// Calcula LCC sobre CSRs pre-construidos (undirected: unión de Out + In).
///
/// Combina los vecinos de `out_csr` y `in_csr` para cada nodo para
/// simular un grafo no dirigido, luego cuenta triángulos.
pub fn lcc_on_csr_undirected(out_csr: &CsrGraph, in_csr: &CsrGraph) -> LccResult {
    let n = out_csr.num_nodes();
    if n == 0 {
        return LccResult {
            coefficients: Vec::new(),
            average: 0.0,
            num_nodes: 0,
        };
    }

    // Para cada nodo, construimos su vecindario no dirigido (unión de out + in,
    // deduplicado y ordenado). Luego contamos triángulos.
    let coefficients: Vec<f64> = (0..n as u32)
        .into_par_iter()
        .map(|v| {
            // Vecindario undirected: unión de out-neighbors y in-neighbors
            let out_neigh = out_csr.neighbors(v);
            let in_neigh = in_csr.neighbors(v);

            // Merge + dedup (ambos están ordenados por internal id)
            let mut neighbors = merge_sorted_unique(out_neigh, in_neigh);
            // Remover self-loops
            neighbors.retain(|&n| n != v);
            let d = neighbors.len();

            if d < 2 {
                return 0.0;
            }

            // Contar triángulos: para cada par (u, w) de vecinos, verificar
            // si u y w son vecinos entre sí.
            let mut triangles: u64 = 0;
            for i in 0..d {
                let u = neighbors[i];
                let u_out = out_csr.neighbors(u);
                let u_in = in_csr.neighbors(u);
                for j in (i + 1)..d {
                    let w = neighbors[j];
                    // Check if u-w edge exists (in either direction)
                    if binary_search_contains(u_out, w) || binary_search_contains(u_in, w) {
                        triangles += 1;
                    }
                }
            }

            // LCC(v) = 2 * T(v) / (d * (d-1))
            let possible = (d as u64) * (d as u64 - 1) / 2;
            if possible == 0 {
                0.0
            } else {
                triangles as f64 / possible as f64
            }
        })
        .collect();

    let sum: f64 = coefficients.iter().sum();
    let average = if n > 0 { sum / n as f64 } else { 0.0 };

    LccResult {
        coefficients,
        average,
        num_nodes: n,
    }
}

/// Merge two sorted slices into a sorted, deduplicated Vec.
fn merge_sorted_unique(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            result.push(a[i]);
            i += 1;
        } else if a[i] > b[j] {
            result.push(b[j]);
            j += 1;
        } else {
            result.push(a[i]);
            i += 1;
            j += 1;
        }
    }
    while i < a.len() {
        result.push(a[i]);
        i += 1;
    }
    while j < b.len() {
        result.push(b[j]);
        j += 1;
    }
    result
}

/// Binary search on sorted slice.
#[inline]
fn binary_search_contains(sorted: &[u32], target: u32) -> bool {
    sorted.binary_search(&target).is_ok()
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use bikodb_core::types::TypeId;

    fn make_triangle() -> ConcurrentGraph {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));
        // Bidirectional edges for undirected triangle
        g.insert_edge(a, b, TypeId(10)).unwrap();
        g.insert_edge(b, a, TypeId(10)).unwrap();
        g.insert_edge(b, c, TypeId(10)).unwrap();
        g.insert_edge(c, b, TypeId(10)).unwrap();
        g.insert_edge(a, c, TypeId(10)).unwrap();
        g.insert_edge(c, a, TypeId(10)).unwrap();
        g
    }

    #[test]
    fn test_triangle_lcc() {
        let g = make_triangle();
        let result = lcc(&g);
        assert_eq!(result.num_nodes, 3);
        // Every node has 2 neighbors that are connected → LCC = 1.0
        for &c in &result.coefficients {
            assert!((c - 1.0).abs() < 1e-9, "Expected 1.0, got {}", c);
        }
        assert!((result.average - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_star_lcc() {
        // Star: center connected to 4 leaves, no connections between leaves
        let g = ConcurrentGraph::new();
        let center = g.insert_node(TypeId(1));
        let mut leaves = Vec::new();
        for _ in 0..4 {
            let leaf = g.insert_node(TypeId(1));
            g.insert_edge(center, leaf, TypeId(10)).unwrap();
            g.insert_edge(leaf, center, TypeId(10)).unwrap();
            leaves.push(leaf);
        }

        let result = lcc(&g);
        assert_eq!(result.num_nodes, 5);
        // Center has 4 neighbors, 0 connections between them → LCC = 0
        let center_internal = CsrGraph::from_concurrent(&g, Direction::Out)
            .to_internal(center)
            .unwrap();
        assert!((result.coefficients[center_internal as usize]).abs() < 1e-9);
        // Leaves have 1 neighbor (center) → LCC = 0 (degree < 2)
    }

    #[test]
    fn test_complete_graph_k4() {
        let g = ConcurrentGraph::new();
        let nodes: Vec<NodeId> = (0..4).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    g.insert_edge(nodes[i], nodes[j], TypeId(10)).unwrap();
                }
            }
        }

        let result = lcc(&g);
        // Complete graph: every node's LCC = 1.0
        for &c in &result.coefficients {
            assert!((c - 1.0).abs() < 1e-9, "Expected 1.0, got {}", c);
        }
    }

    #[test]
    fn test_empty_graph() {
        let g = ConcurrentGraph::new();
        let result = lcc(&g);
        assert_eq!(result.num_nodes, 0);
        assert!((result.average - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_single_node() {
        let g = ConcurrentGraph::new();
        g.insert_node(TypeId(1));
        let result = lcc(&g);
        assert_eq!(result.num_nodes, 1);
        assert!((result.coefficients[0] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_path_graph() {
        // A - B - C (path, no triangles)
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));
        g.insert_edge(a, b, TypeId(10)).unwrap();
        g.insert_edge(b, a, TypeId(10)).unwrap();
        g.insert_edge(b, c, TypeId(10)).unwrap();
        g.insert_edge(c, b, TypeId(10)).unwrap();

        let result = lcc(&g);
        // No triangles: all LCC = 0
        for &c in &result.coefficients {
            assert!((c - 0.0).abs() < 1e-9, "Expected 0.0, got {}", c);
        }
    }

    #[test]
    fn test_merge_sorted_unique() {
        assert_eq!(merge_sorted_unique(&[1, 3, 5], &[2, 3, 6]), vec![1, 2, 3, 5, 6]);
        assert_eq!(merge_sorted_unique(&[], &[1, 2]), vec![1, 2]);
        assert_eq!(merge_sorted_unique(&[1, 2], &[]), vec![1, 2]);
        assert_eq!(merge_sorted_unique(&[1, 2, 3], &[1, 2, 3]), vec![1, 2, 3]);
    }
}
