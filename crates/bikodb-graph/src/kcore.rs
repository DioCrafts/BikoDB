// =============================================================================
// bikodb-graph::kcore — K-Core Decomposition
// =============================================================================
//
// ## Algoritmo
//
// El k-core de un grafo es el subgrafo maximal en el que todos los nodos
// tienen degree ≥ k. La descomposición k-core asigna a cada nodo un valor
// de "coreness" = el máximo k para el cual el nodo pertenece al k-core.
//
// ### Peeling Algorithm (Batagelj & Zaversnik, 2003)
// 1. Calcular degree de cada nodo
// 2. Procesar nodos en orden de degree creciente
// 3. Al "pelar" un nodo, decrementar degree de vecinos
// 4. El degree al momento de ser pelado = coreness
//
// Complejidad: O(V + E) tiempo, O(V) espacio extra.
//
// ## Casos de uso
// - Identificar nodos influyentes (hub detection)
// - Simplificación de grafos (podar periferia)
// - Network resilience analysis
// - Community finding (dense subgraphs)
//
// ## Referencia
// - Kuzu: implementa K-core en su GDS extension
// - ArcadeDB / Neo4j: no incluyen K-core
// =============================================================================

use crate::csr::CsrGraph;
use bikodb_core::record::Direction;
use bikodb_core::types::NodeId;

use crate::graph::ConcurrentGraph;

/// Resultado de K-core decomposition.
#[derive(Debug)]
pub struct KcoreResult {
    /// Coreness de cada nodo (indexado por internal id).
    pub coreness: Vec<u32>,
    /// Máximo valor de coreness en el grafo.
    pub max_coreness: u32,
    /// Número de nodos.
    pub num_nodes: usize,
}

impl KcoreResult {
    /// Coreness de un nodo externo.
    pub fn coreness_of(&self, csr: &CsrGraph, node: NodeId) -> Option<u32> {
        let internal = csr.to_internal(node)?;
        Some(self.coreness[internal as usize])
    }

    /// Nodos con coreness ≥ k (retorna NodeIds externos).
    pub fn k_core_members(&self, csr: &CsrGraph, k: u32) -> Vec<NodeId> {
        self.coreness
            .iter()
            .enumerate()
            .filter(|(_, &c)| c >= k)
            .map(|(i, _)| csr.to_external(i as u32))
            .collect()
    }

    /// Distribución de coreness: Vec<(coreness, count)> ordenado asc.
    pub fn distribution(&self) -> Vec<(u32, usize)> {
        let mut counts = std::collections::HashMap::new();
        for &c in &self.coreness {
            *counts.entry(c).or_insert(0usize) += 1;
        }
        let mut dist: Vec<(u32, usize)> = counts.into_iter().collect();
        dist.sort_unstable_by_key(|&(k, _)| k);
        dist
    }
}

// =============================================================================
// K-core — sobre ConcurrentGraph (undirected)
// =============================================================================

/// K-core decomposition sobre ConcurrentGraph (trata como no dirigido).
pub fn kcore(graph: &ConcurrentGraph) -> KcoreResult {
    let out_csr = CsrGraph::from_concurrent(graph, Direction::Out);
    let in_csr = CsrGraph::from_concurrent(graph, Direction::In);
    kcore_on_csr_undirected(&out_csr, &in_csr)
}

/// K-core decomposition sobre CSR bidireccional (undirected).
///
/// Peeling algorithm: O(V + E).
pub fn kcore_on_csr_undirected(out_csr: &CsrGraph, in_csr: &CsrGraph) -> KcoreResult {
    let n = out_csr.num_nodes();
    if n == 0 {
        return KcoreResult {
            coreness: Vec::new(),
            max_coreness: 0,
            num_nodes: 0,
        };
    }

    // Compute undirected degree
    let mut degree = vec![0u32; n];
    for v in 0..n as u32 {
        // Merge out+in neighbors, deduplicated
        let mut neighbors = std::collections::HashSet::new();
        for &u in out_csr.neighbors(v) {
            neighbors.insert(u);
        }
        for &u in in_csr.neighbors(v) {
            neighbors.insert(u);
        }
        degree[v as usize] = neighbors.len() as u32;
    }

    let max_degree = *degree.iter().max().unwrap_or(&0);

    // Bin-sort by degree (counting sort)
    let mut bin = vec![0u32; max_degree as usize + 1];
    for &d in &degree {
        bin[d as usize] += 1;
    }
    // Prefix sum: bin[d] = starting position for degree d
    let mut start = 0u32;
    for b in bin.iter_mut() {
        let count = *b;
        *b = start;
        start += count;
    }

    // Sort nodes by degree into `order`, and maintain `pos` = position in order
    let mut order = vec![0u32; n];
    let mut pos = vec![0u32; n];
    let mut bin_copy = bin.clone(); // working copy for placement
    for v in 0..n {
        let d = degree[v] as usize;
        order[bin_copy[d] as usize] = v as u32;
        pos[v] = bin_copy[d];
        bin_copy[d] += 1;
    }

    // Peeling
    let mut coreness = degree.clone();

    for i in 0..n {
        let v = order[i] as usize;
        // v's coreness is its current degree value
        // Process neighbors: decrement their effective degree
        // We need unique undirected neighbors
        let mut neighbors = Vec::new();
        for &u in out_csr.neighbors(v as u32) {
            neighbors.push(u);
        }
        for &u in in_csr.neighbors(v as u32) {
            neighbors.push(u);
        }
        neighbors.sort_unstable();
        neighbors.dedup();

        for &u in &neighbors {
            let u_idx = u as usize;
            if coreness[u_idx] > coreness[v] {
                // Move u left in the bin-sort order
                let du = coreness[u_idx] as usize;
                let pu = pos[u_idx] as usize;
                let pw = bin[du] as usize;
                let w = order[pw] as usize;

                // Swap u and w in the order
                if pu != pw {
                    order[pu] = w as u32;
                    order[pw] = u;
                    pos[w] = pu as u32;
                    pos[u_idx] = pw as u32;
                }

                bin[du] += 1;
                coreness[u_idx] -= 1;
            }
        }
    }

    let max_coreness = *coreness.iter().max().unwrap_or(&0);

    KcoreResult {
        coreness,
        max_coreness,
        num_nodes: n,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bikodb_core::types::TypeId;

    fn make_undirected(edges: &[(usize, usize)]) -> ConcurrentGraph {
        let g = ConcurrentGraph::new();
        let max_node = edges.iter().flat_map(|&(a, b)| [a, b]).max().unwrap_or(0) + 1;
        let nodes: Vec<NodeId> = (0..max_node).map(|_| g.insert_node(TypeId(1))).collect();
        for &(a, b) in edges {
            let _ = g.insert_edge(nodes[a], nodes[b], TypeId(10));
            let _ = g.insert_edge(nodes[b], nodes[a], TypeId(10));
        }
        g
    }

    #[test]
    fn test_kcore_triangle() {
        // Triangle: 0-1, 1-2, 0-2 → all have coreness = 2
        let g = make_undirected(&[(0, 1), (1, 2), (0, 2)]);
        let result = kcore(&g);
        assert_eq!(result.max_coreness, 2);
        for &c in &result.coreness {
            assert_eq!(c, 2);
        }
    }

    #[test]
    fn test_kcore_star() {
        // Star: 0-1, 0-2, 0-3, 0-4 → center=1, leaves=1
        let g = make_undirected(&[(0, 1), (0, 2), (0, 3), (0, 4)]);
        let result = kcore(&g);
        assert_eq!(result.max_coreness, 1);
    }

    #[test]
    fn test_kcore_k4() {
        // K4 complete: all nodes have degree 3, coreness = 3
        let g = make_undirected(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
        let result = kcore(&g);
        assert_eq!(result.max_coreness, 3);
        for &c in &result.coreness {
            assert_eq!(c, 3);
        }
    }

    #[test]
    fn test_kcore_path() {
        // Path: 0-1-2-3 → all have coreness = 1
        let g = make_undirected(&[(0, 1), (1, 2), (2, 3)]);
        let result = kcore(&g);
        assert_eq!(result.max_coreness, 1);
    }

    #[test]
    fn test_kcore_isolated() {
        let g = ConcurrentGraph::new();
        for _ in 0..3 {
            g.insert_node(TypeId(1));
        }
        let result = kcore(&g);
        assert_eq!(result.max_coreness, 0);
        for &c in &result.coreness {
            assert_eq!(c, 0);
        }
    }

    #[test]
    fn test_kcore_empty() {
        let g = ConcurrentGraph::new();
        let result = kcore(&g);
        assert_eq!(result.max_coreness, 0);
        assert_eq!(result.num_nodes, 0);
    }

    #[test]
    fn test_kcore_clique_with_pendant() {
        // K4 (0,1,2,3) + pendant 4 connected to 0
        // K4 nodes: coreness=3, pendant: coreness=1
        let g = make_undirected(&[
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 4),
        ]);
        let result = kcore(&g);
        assert_eq!(result.max_coreness, 3);
        // Node 4 should have coreness 1
        // Internal ids may differ, check via CSR
        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let k_core_3 = result.k_core_members(&out_csr, 3);
        assert_eq!(k_core_3.len(), 4);
    }

    #[test]
    fn test_kcore_distribution() {
        let g = make_undirected(&[
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 4),
        ]);
        let result = kcore(&g);
        let dist = result.distribution();
        assert!(!dist.is_empty());
        // Should have coreness 1 and 3
        let keys: Vec<u32> = dist.iter().map(|&(k, _)| k).collect();
        assert!(keys.contains(&1));
        assert!(keys.contains(&3));
    }
}
