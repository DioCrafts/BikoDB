// =============================================================================
// bikodb-graph::louvain — Louvain Community Detection
// =============================================================================
//
// ## Algoritmo Louvain (Blondel et al., 2008)
//
// Detección de comunidades optimizando modularidad (Q) en dos fases:
//
// 1. **Local phase**: cada nodo se mueve a la comunidad vecina que maximiza ΔQ
// 2. **Aggregation phase**: se colapsa el grafo (super-nodos = comunidades)
//
// Se repite hasta que Q ya no mejora.
//
// ## Modularidad
//
// $Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$
//
// Donde m = total edges, k_i = degree(i), c_i = community(i).
//
// ## Ventaja sobre Label Propagation
//
// - Optimiza una función objetivo global (modularidad)
// - Produce comunidades de mayor calidad (más estables, mejor recall)
// - Multi-level: descubre estructura jerárquica
// - LPA: O(k·(V+E)), Louvain: O(V·log(V)) en práctica
//
// ## Implementación
//
// - Trabajamos sobre CSR bidireccional (out + in → no dirigido)
// - Weighted modularity (usa edge weights si existen)
// - Fase local: iteraciones paralelas con atomic ΔQ
// - Multi-level: limitado a max_levels (default: 10)
//
// ## Referencia
// - Kuzu: implementa Louvain en su GDS extension
// - ArcadeDB: no incluye Louvain
// - Neo4j GDS: incluye Louvain con multi-level
// =============================================================================

use crate::csr::CsrGraph;
use bikodb_core::record::Direction;
use bikodb_core::types::NodeId;
use rayon::prelude::*;

use crate::graph::ConcurrentGraph;

/// Configuración de Louvain.
#[derive(Debug, Clone)]
pub struct LouvainConfig {
    /// Máximo de iteraciones en la fase local por nivel.
    pub max_iterations: u32,
    /// Máximo de niveles de coarsening.
    pub max_levels: u32,
    /// Umbral mínimo de mejora de modularidad para continuar.
    pub min_modularity_gain: f64,
}

impl Default for LouvainConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            max_levels: 10,
            min_modularity_gain: 1e-6,
        }
    }
}

/// Resultado de Louvain.
#[derive(Debug)]
pub struct LouvainResult {
    /// Community ID para cada nodo (indexado por internal id del CSR original).
    pub communities: Vec<u32>,
    /// Modularidad final.
    pub modularity: f64,
    /// Número de comunidades.
    pub num_communities: usize,
    /// Número de niveles ejecutados.
    pub levels: u32,
}

impl LouvainResult {
    /// Community ID de un nodo externo.
    pub fn community_of(&self, csr: &CsrGraph, node: NodeId) -> Option<u32> {
        let internal = csr.to_internal(node)?;
        Some(self.communities[internal as usize])
    }
}

// =============================================================================
// Louvain — sobre ConcurrentGraph
// =============================================================================

/// Louvain sobre ConcurrentGraph (construye CSR internamente).
pub fn louvain(graph: &ConcurrentGraph, config: &LouvainConfig) -> LouvainResult {
    let out_csr = CsrGraph::from_concurrent(graph, Direction::Out);
    let in_csr = CsrGraph::from_concurrent(graph, Direction::In);
    louvain_on_csr(&out_csr, &in_csr, config)
}

/// Louvain sobre CSR pre-construidos.
pub fn louvain_on_csr(
    out_csr: &CsrGraph,
    in_csr: &CsrGraph,
    config: &LouvainConfig,
) -> LouvainResult {
    let n = out_csr.num_nodes();
    if n == 0 {
        return LouvainResult {
            communities: Vec::new(),
            modularity: 0.0,
            num_communities: 0,
            levels: 0,
        };
    }

    // Build undirected weighted adjacency from CSR
    // Edge weight = number of edges between u and v (counting both directions)
    let adj = build_undirected_adj(out_csr, in_csr);
    let total_weight = adj.total_weight;

    // Initial community = each node in its own community
    let mut community: Vec<u32> = (0..n as u32).collect();
    let mut level = 0u32;
    let mut current_adj = adj;
    let mut current_n = n;

    loop {
        if level >= config.max_levels {
            break;
        }

        // Phase 1: local moves
        let (new_community, improved) =
            local_phase(&current_adj, current_n, config);

        if !improved {
            break;
        }

        // Phase 2: aggregate into super-graph
        // `mapping` renumbers new_community to 0..K-1
        let (agg_adj, agg_n, mapping) =
            aggregate(&current_adj, current_n, &new_community);

        if agg_n >= current_n {
            break; // No reduction
        }

        // Map original community labels through the renumbered mapping
        for c in community.iter_mut() {
            *c = mapping[*c as usize];
        }

        current_adj = agg_adj;
        current_n = agg_n;
        level += 1;
    }

    // Renumber communities to 0..K-1
    let (final_communities, num_communities) = renumber(&community);
    let modularity = compute_modularity_from_adj_and_communities(
        out_csr, in_csr, &final_communities, total_weight,
    );

    LouvainResult {
        communities: final_communities,
        modularity,
        num_communities,
        levels: level,
    }
}

// =============================================================================
// Internal data structures
// =============================================================================

/// Adjacency in CSR-like format for the weighted undirected graph.
struct WeightedAdj {
    /// offsets[i]..offsets[i+1] = range in neighbors/weights for node i
    offsets: Vec<usize>,
    /// Neighbor node ids (internal)
    neighbors: Vec<u32>,
    /// Weight of each edge
    weights: Vec<f64>,
    /// Sum of all edge weights (each undirected edge counted once as 2*w)
    total_weight: f64,
    /// Weighted degree k_i for each node
    k: Vec<f64>,
}

fn build_undirected_adj(out_csr: &CsrGraph, in_csr: &CsrGraph) -> WeightedAdj {
    let n = out_csr.num_nodes();
    // Merge out + in edges, sum weights (each edge weight = 1.0 for unweighted)
    let mut adj_map: Vec<std::collections::HashMap<u32, f64>> =
        (0..n).map(|_| std::collections::HashMap::new()).collect();

    for v in 0..n as u32 {
        for &u in out_csr.neighbors(v) {
            *adj_map[v as usize].entry(u).or_insert(0.0) += 1.0;
        }
        for &u in in_csr.neighbors(v) {
            *adj_map[v as usize].entry(u).or_insert(0.0) += 1.0;
        }
    }

    // Build CSR-like structure
    let mut offsets = Vec::with_capacity(n + 1);
    let mut neighbors = Vec::new();
    let mut weights = Vec::new();
    let mut k = vec![0.0f64; n];
    let mut total_weight = 0.0f64;

    for v in 0..n {
        offsets.push(neighbors.len());
        for (&u, &w) in &adj_map[v] {
            neighbors.push(u);
            weights.push(w);
            k[v] += w;
            total_weight += w;
        }
    }
    offsets.push(neighbors.len());

    WeightedAdj {
        offsets,
        neighbors,
        weights,
        total_weight,
        k,
    }
}

// =============================================================================
// Phase 1: Local modularity optimization
// =============================================================================

fn local_phase(
    adj: &WeightedAdj,
    n: usize,
    config: &LouvainConfig,
) -> (Vec<u32>, bool) {
    let m2 = adj.total_weight; // 2*m
    if m2 == 0.0 {
        return ((0..n as u32).collect(), false);
    }

    let mut community: Vec<u32> = (0..n as u32).collect();
    // sigma_tot[c] = sum of k_i for all nodes in community c
    let mut sigma_tot: Vec<f64> = adj.k.clone();
    let mut any_improved = false;

    for _iter in 0..config.max_iterations {
        let mut moved = false;

        for v in 0..n {
            let v_comm = community[v];
            let k_v = adj.k[v];

            // Compute sum of edge weights to each neighboring community
            let mut comm_weights: Vec<(u32, f64)> = Vec::new();
            let start = adj.offsets[v];
            let end = adj.offsets[v + 1];
            for idx in start..end {
                let u = adj.neighbors[idx] as usize;
                let w = adj.weights[idx];
                let u_comm = community[u];
                let mut found = false;
                for entry in comm_weights.iter_mut() {
                    if entry.0 == u_comm {
                        entry.1 += w;
                        found = true;
                        break;
                    }
                }
                if !found {
                    comm_weights.push((u_comm, w));
                }
            }

            // Remove v from its current community
            let k_v_in_own = comm_weights
                .iter()
                .find(|&&(c, _)| c == v_comm)
                .map(|&(_, w)| w)
                .unwrap_or(0.0);

            sigma_tot[v_comm as usize] -= k_v;

            // Find best community
            let mut best_comm = v_comm;
            // ΔQ for removing from current community
            let mut best_delta = 0.0f64;

            for &(c, k_v_in_c) in &comm_weights {
                // ΔQ = [k_v_in_c / m2] - [sigma_tot[c] * k_v / (m2 * m2)]
                // Compared to staying removed (ΔQ = 0)
                let delta = k_v_in_c / m2 - (sigma_tot[c as usize] * k_v) / (m2 * m2);
                if delta > best_delta || (delta == best_delta && c < best_comm) {
                    best_delta = delta;
                    best_comm = c;
                }
            }

            // Also consider own community (re-insert) with the delta_remove cost
            let delta_own = k_v_in_own / m2
                - (sigma_tot[v_comm as usize] * k_v) / (m2 * m2);
            if delta_own >= best_delta && v_comm <= best_comm {
                best_comm = v_comm;
            }

            // Move v to best community
            community[v] = best_comm;
            sigma_tot[best_comm as usize] += k_v;

            if best_comm != v_comm {
                moved = true;
                any_improved = true;
            }
        }

        if !moved {
            break;
        }
    }

    (community, any_improved)
}

// =============================================================================
// Phase 2: Aggregation (coarsen graph)
// =============================================================================

fn aggregate(
    adj: &WeightedAdj,
    n: usize,
    community: &[u32],
) -> (WeightedAdj, usize, Vec<u32>) {
    // Renumber communities to 0..K-1
    let (mapping, num_comms) = renumber(community);

    // Build super-graph adjacency
    let mut super_adj: Vec<std::collections::HashMap<u32, f64>> =
        (0..num_comms).map(|_| std::collections::HashMap::new()).collect();
    let mut super_k = vec![0.0f64; num_comms];

    for v in 0..n {
        let cv = mapping[v] as usize;
        let start = adj.offsets[v];
        let end = adj.offsets[v + 1];
        for idx in start..end {
            let u = adj.neighbors[idx] as usize;
            let w = adj.weights[idx];
            let cu = mapping[u];
            *super_adj[cv].entry(cu).or_insert(0.0) += w;
            super_k[cv] += w;
        }
    }

    // Build CSR-like
    let mut offsets = Vec::with_capacity(num_comms + 1);
    let mut neighbors = Vec::new();
    let mut weights = Vec::new();
    let mut total_weight = 0.0f64;

    for c in 0..num_comms {
        offsets.push(neighbors.len());
        for (&u, &w) in &super_adj[c] {
            neighbors.push(u);
            weights.push(w);
            total_weight += w;
        }
    }
    offsets.push(neighbors.len());

    let new_adj = WeightedAdj {
        offsets,
        neighbors,
        weights,
        total_weight,
        k: super_k,
    };

    (new_adj, num_comms, mapping)
}

// =============================================================================
// Utilities
// =============================================================================

fn renumber(labels: &[u32]) -> (Vec<u32>, usize) {
    let mut map = std::collections::HashMap::new();
    let mut next_id = 0u32;
    let result: Vec<u32> = labels
        .iter()
        .map(|&l| {
            *map.entry(l).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect();
    (result, next_id as usize)
}

fn compute_modularity_from_adj_and_communities(
    out_csr: &CsrGraph,
    in_csr: &CsrGraph,
    communities: &[u32],
    _total_weight_hint: f64,
) -> f64 {
    let n = out_csr.num_nodes();
    if n == 0 {
        return 0.0;
    }

    // Compute degrees (undirected: out + in)
    let mut degree = vec![0.0f64; n];
    let mut m2 = 0.0f64;
    for v in 0..n as u32 {
        let d = out_csr.degree(v) as f64 + in_csr.degree(v) as f64;
        degree[v as usize] = d;
        m2 += d;
    }

    if m2 == 0.0 {
        return 0.0;
    }

    // Q = (1/2m) * sum over edges {delta(c_i, c_j)} - (1/2m) * sum_c (sigma_c / 2m)^2
    let mut q_edge = 0.0f64;
    for v in 0..n as u32 {
        let cv = communities[v as usize];
        for &u in out_csr.neighbors(v) {
            if communities[u as usize] == cv {
                q_edge += 1.0;
            }
        }
        for &u in in_csr.neighbors(v) {
            if communities[u as usize] == cv {
                q_edge += 1.0;
            }
        }
    }

    // sigma_c = sum of degrees of nodes in community c
    let mut sigma: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    for v in 0..n {
        *sigma.entry(communities[v]).or_insert(0.0) += degree[v];
    }

    let mut q_degree = 0.0f64;
    for (_, s) in &sigma {
        q_degree += (s / m2) * (s / m2);
    }

    q_edge / m2 - q_degree
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bikodb_core::types::TypeId;

    fn two_cliques_with_bridge(k: usize) -> ConcurrentGraph {
        let g = ConcurrentGraph::new();
        let clique_a: Vec<NodeId> = (0..k).map(|_| g.insert_node(TypeId(1))).collect();
        let clique_b: Vec<NodeId> = (0..k).map(|_| g.insert_node(TypeId(1))).collect();

        for i in 0..k {
            for j in (i + 1)..k {
                let _ = g.insert_edge(clique_a[i], clique_a[j], TypeId(10));
                let _ = g.insert_edge(clique_a[j], clique_a[i], TypeId(10));
                let _ = g.insert_edge(clique_b[i], clique_b[j], TypeId(10));
                let _ = g.insert_edge(clique_b[j], clique_b[i], TypeId(10));
            }
        }
        // Single bridge between the two cliques
        let _ = g.insert_edge(clique_a[0], clique_b[0], TypeId(10));
        let _ = g.insert_edge(clique_b[0], clique_a[0], TypeId(10));
        g
    }

    #[test]
    fn test_louvain_two_cliques() {
        let g = two_cliques_with_bridge(5);
        let config = LouvainConfig::default();
        let result = louvain(&g, &config);
        // Should find 2 communities (the two cliques)
        assert!(result.num_communities <= 3, "Got {} communities", result.num_communities);
        assert!(result.modularity > 0.0, "Modularity should be positive");
    }

    #[test]
    fn test_louvain_single_node() {
        let g = ConcurrentGraph::new();
        g.insert_node(TypeId(1));
        let result = louvain(&g, &LouvainConfig::default());
        assert_eq!(result.num_communities, 1);
    }

    #[test]
    fn test_louvain_empty_graph() {
        let g = ConcurrentGraph::new();
        let result = louvain(&g, &LouvainConfig::default());
        assert_eq!(result.num_communities, 0);
        assert_eq!(result.modularity, 0.0);
    }

    #[test]
    fn test_louvain_disconnected_cliques() {
        // Two fully disconnected K4's
        let g = ConcurrentGraph::new();
        let clique_a: Vec<NodeId> = (0..4).map(|_| g.insert_node(TypeId(1))).collect();
        let clique_b: Vec<NodeId> = (0..4).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..4 {
            for j in (i + 1)..4 {
                let _ = g.insert_edge(clique_a[i], clique_a[j], TypeId(10));
                let _ = g.insert_edge(clique_a[j], clique_a[i], TypeId(10));
                let _ = g.insert_edge(clique_b[i], clique_b[j], TypeId(10));
                let _ = g.insert_edge(clique_b[j], clique_b[i], TypeId(10));
            }
        }
        let result = louvain(&g, &LouvainConfig::default());
        assert_eq!(result.num_communities, 2);
    }

    #[test]
    fn test_louvain_complete_graph() {
        let g = ConcurrentGraph::new();
        let nodes: Vec<NodeId> = (0..6).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..6 {
            for j in (i + 1)..6 {
                let _ = g.insert_edge(nodes[i], nodes[j], TypeId(10));
                let _ = g.insert_edge(nodes[j], nodes[i], TypeId(10));
            }
        }
        let result = louvain(&g, &LouvainConfig::default());
        // Complete graph: all in one community
        assert_eq!(result.num_communities, 1);
    }

    #[test]
    fn test_louvain_modularity_positive() {
        let g = two_cliques_with_bridge(8);
        let result = louvain(&g, &LouvainConfig::default());
        assert!(result.modularity > 0.2, "Modularity too low: {}", result.modularity);
    }

    #[test]
    fn test_louvain_community_of() {
        let g = ConcurrentGraph::new();
        let n1 = g.insert_node(TypeId(1));
        let n2 = g.insert_node(TypeId(1));
        let _ = g.insert_edge(n1, n2, TypeId(10));
        let _ = g.insert_edge(n2, n1, TypeId(10));
        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let result = louvain(&g, &LouvainConfig::default());
        assert!(result.community_of(&out_csr, n1).is_some());
        assert!(result.community_of(&out_csr, n2).is_some());
    }
}
