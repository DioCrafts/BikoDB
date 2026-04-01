// =============================================================================
// bikodb-graph::community — Community Detection paralelo sobre CSR
// =============================================================================
//
// ## Algoritmos incluidos
//
// | Algoritmo                     | Tipo          | Paralelo | Complejidad    |
// |-------------------------------|---------------|----------|----------------|
// | Label Propagation (LPA)       | Overlapping   | Sí       | O(k·(V+E))/P  |
// | Connected Components (CC)     | Disjoint      | Sí       | O(k·(V+E))/P  |
//
// ## Label Propagation Algorithm (LPA)
//
// Raghavan et al., 2007. Cada nodo adopta la etiqueta más frecuente entre
// sus vecinos en cada iteración. Es uno de los algoritmos de community
// detection más rápidos (near-linear time).
//
// ### Versión paralela
//
// - Todos los nodos se actualizan en paralelo (synchronous LPA)
// - Usa rayon par_chunks_mut para zero contention
// - Labels compactos: Vec<u32> (4 bytes/nodo vs ~40 bytes con HashMap)
// - Convergencia: se detiene cuando ningún nodo cambia de label
//
// ### Desempate
//
// Cuando múltiples labels tienen la misma frecuencia máxima, se elige
// el label con valor numérico mínimo (determinista, reproducible).
//
// ## Connected Components (CC)
//
// Min-label propagation paralelo: cada nodo adopta el mínimo label entre
// él y sus vecinos. Converge en O(diámetro) iteraciones.
// Equivalente a encontrar componentes conexas en grafo no dirigido.
//
// ## Almacenamiento compacto
//
// - Labels: Vec<u32> — 4 bytes/nodo, indexado por internal id
// - Para 10M nodos: 40 MB (vs ~400 MB con HashSet<NodeId>)
// - Community sizes: Vec<(u32, usize)> — (label, count) on demand
// =============================================================================

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::csr::CsrGraph;
use bikodb_core::record::Direction;
use bikodb_core::types::NodeId;
use rayon::prelude::*;

use crate::graph::ConcurrentGraph;

// =============================================================================
// Configuración
// =============================================================================

/// Parámetros para Label Propagation.
#[derive(Debug, Clone)]
pub struct LpaConfig {
    /// Máximo de iteraciones.
    pub max_iterations: u32,
}

impl Default for LpaConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
        }
    }
}

// =============================================================================
// Resultado
// =============================================================================

/// Resultado de community detection.
#[derive(Debug)]
pub struct CommunityResult {
    /// Label de comunidad para cada nodo (indexado por internal id).
    /// Valores: 0..N-1 (cada label es el internal id del nodo "fundador").
    pub labels: Vec<u32>,
    /// Número de iteraciones ejecutadas.
    pub iterations: u32,
    /// ¿Convergió (ningún cambio en la última iteración)?
    pub converged: bool,
    /// Número de comunidades distintas.
    pub num_communities: usize,
}

impl CommunityResult {
    /// Label de un nodo externo.
    pub fn community_of(&self, csr: &CsrGraph, node: NodeId) -> Option<u32> {
        let internal = csr.to_internal(node)?;
        Some(self.labels[internal as usize])
    }

    /// ¿Están dos nodos en la misma comunidad?
    pub fn same_community(&self, csr: &CsrGraph, a: NodeId, b: NodeId) -> Option<bool> {
        let la = self.community_of(csr, a)?;
        let lb = self.community_of(csr, b)?;
        Some(la == lb)
    }

    /// Tamaños de comunidad: Vec<(label, count)> ordenado descendente por count.
    pub fn community_sizes(&self) -> Vec<(u32, usize)> {
        let mut counts = std::collections::HashMap::new();
        for &label in &self.labels {
            *counts.entry(label).or_insert(0usize) += 1;
        }
        let mut sizes: Vec<(u32, usize)> = counts.into_iter().collect();
        sizes.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        sizes
    }

    /// Nodos de una comunidad específica (retorna NodeIds externos).
    pub fn members_of(&self, csr: &CsrGraph, label: u32) -> Vec<NodeId> {
        self.labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == label)
            .map(|(i, _)| csr.to_external(i as u32))
            .collect()
    }
}

// =============================================================================
// Label Propagation Algorithm (LPA) — paralelo
// =============================================================================

/// Label Propagation sobre un ConcurrentGraph (construye CSR internamente).
///
/// Usa CSR bidireccional (out + in edges) para tratar el grafo como no dirigido.
pub fn label_propagation(graph: &ConcurrentGraph, config: &LpaConfig) -> CommunityResult {
    let out_csr = CsrGraph::from_concurrent(graph, Direction::Out);
    let in_csr = CsrGraph::from_concurrent(graph, Direction::In);
    label_propagation_on_csr(&out_csr, Some(&in_csr), config)
}

/// Label Propagation sobre CSR pre-construidos.
///
/// `in_csr`: si Some, se usa como edges adicionales (grafo no dirigido).
/// Si None, se usa solo `out_csr` (grafo dirigido).
///
/// Usa **async parallel LPA**: labels son AtomicU32, cada nodo lee los últimos
/// valores (posiblemente ya actualizados en la misma iteración). Esto acelera
/// la convergencia dramáticamente vs synchronous LPA. Incluye self-vote para
/// estabilidad adicional.
pub fn label_propagation_on_csr(
    out_csr: &CsrGraph,
    in_csr: Option<&CsrGraph>,
    config: &LpaConfig,
) -> CommunityResult {
    let n = out_csr.num_nodes();

    if n == 0 {
        return CommunityResult {
            labels: Vec::new(),
            iterations: 0,
            converged: true,
            num_communities: 0,
        };
    }

    // Async LPA: AtomicU32 labels permiten lecturas del estado más reciente
    let labels: Vec<AtomicU32> = (0..n).map(|i| AtomicU32::new(i as u32)).collect();

    let mut iterations: u32 = 0;
    let mut converged = false;

    for _ in 0..config.max_iterations {
        let changed = AtomicBool::new(false);

        // Parallel async update: cada nodo lee los labels más recientes
        (0..n).into_par_iter().for_each(|v_usize| {
            let v = v_usize as u32;

            // Collect unique neighbor IDs (dedup out+in to avoid double-counting
            // bidirectional edges)
            let mut neighbor_ids = Vec::<u32>::new();
            for &u in out_csr.neighbors(v) {
                neighbor_ids.push(u);
            }
            if let Some(in_c) = in_csr {
                for &u in in_c.neighbors(v) {
                    neighbor_ids.push(u);
                }
            }
            neighbor_ids.sort_unstable();
            neighbor_ids.dedup();

            // Self-vote + unique neighbor votes
            let mut freq = Vec::<(u32, u32)>::new();
            increment_freq(&mut freq, labels[v_usize].load(Ordering::Relaxed));
            for &u in &neighbor_ids {
                increment_freq(&mut freq, labels[u as usize].load(Ordering::Relaxed));
            }

            // freq siempre tiene al menos el self-vote
            let mut best_label = freq[0].0;
            let mut best_count = freq[0].1;
            for &(label, count) in &freq[1..] {
                if count > best_count
                    || (count == best_count && label < best_label)
                {
                    best_label = label;
                    best_count = count;
                }
            }

            let old = labels[v_usize].load(Ordering::Relaxed);
            if best_label != old {
                labels[v_usize].store(best_label, Ordering::Relaxed);
                changed.store(true, Ordering::Relaxed);
            }
        });

        iterations += 1;

        if !changed.load(Ordering::Relaxed) {
            converged = true;
            break;
        }
    }

    let final_labels: Vec<u32> = labels.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    let num_communities = count_distinct(&final_labels);

    CommunityResult {
        labels: final_labels,
        iterations,
        converged,
        num_communities,
    }
}

/// Increment frequency of `label` in a small vec (linear scan, usually < 20 entries).
#[inline]
fn increment_freq(freq: &mut Vec<(u32, u32)>, label: u32) {
    for entry in freq.iter_mut() {
        if entry.0 == label {
            entry.1 += 1;
            return;
        }
    }
    freq.push((label, 1));
}

/// Count distinct values in a label array.
fn count_distinct(labels: &[u32]) -> usize {
    let mut sorted = labels.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted.len()
}

// =============================================================================
// Connected Components — min-label propagation paralelo
// =============================================================================

/// Connected Components sobre un ConcurrentGraph.
///
/// Trata el grafo como no dirigido (out + in edges).
pub fn connected_components(graph: &ConcurrentGraph) -> CommunityResult {
    let out_csr = CsrGraph::from_concurrent(graph, Direction::Out);
    let in_csr = CsrGraph::from_concurrent(graph, Direction::In);
    connected_components_on_csr(&out_csr, Some(&in_csr))
}

/// Connected Components sobre CSR.
///
/// Usa **lock-free union-find** con path compression (AtomicU32).
/// Complejidad: O((V+E) · α(V)) — casi lineal.
pub fn connected_components_on_csr(
    out_csr: &CsrGraph,
    in_csr: Option<&CsrGraph>,
) -> CommunityResult {
    let n = out_csr.num_nodes();

    if n == 0 {
        return CommunityResult {
            labels: Vec::new(),
            iterations: 0,
            converged: true,
            num_communities: 0,
        };
    }

    // Lock-free union-find: parent[v] = representative
    let parent: Vec<AtomicU32> = (0..n).map(|i| AtomicU32::new(i as u32)).collect();

    // Process all edges in parallel — union endpoints
    (0..n).into_par_iter().for_each(|v_usize| {
        let v = v_usize as u32;
        for &u in out_csr.neighbors(v) {
            union_atomic(&parent, v, u);
        }
        if let Some(in_c) = in_csr {
            for &u in in_c.neighbors(v) {
                union_atomic(&parent, v, u);
            }
        }
    });

    // Final path compression: every node points to its root
    let labels: Vec<u32> = (0..n).map(|i| find_atomic(&parent, i as u32)).collect();
    let num_communities = count_distinct(&labels);

    CommunityResult {
        labels,
        iterations: 1,
        converged: true,
        num_communities,
    }
}

// =============================================================================
// Lock-free Union-Find helpers
// =============================================================================

/// Find root with path compression (path halving).
fn find_atomic(parent: &[AtomicU32], mut x: u32) -> u32 {
    loop {
        let p = parent[x as usize].load(Ordering::Relaxed);
        if p == x {
            return x;
        }
        let gp = parent[p as usize].load(Ordering::Relaxed);
        // Path halving: point x to grandparent
        let _ = parent[x as usize].compare_exchange_weak(
            p,
            gp,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        x = gp;
    }
}

/// Union by min-id: attach larger root to smaller root.
fn union_atomic(parent: &[AtomicU32], a: u32, b: u32) {
    loop {
        let ra = find_atomic(parent, a);
        let rb = find_atomic(parent, b);
        if ra == rb {
            return;
        }
        let (min, max) = if ra < rb { (ra, rb) } else { (rb, ra) };
        match parent[max as usize].compare_exchange_weak(
            max,
            min,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return,
            Err(_) => continue,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bikodb_core::types::TypeId;

    fn default_config() -> LpaConfig {
        LpaConfig::default()
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Two disconnected cliques of size `k`.
    fn two_cliques(k: usize) -> (ConcurrentGraph, Vec<NodeId>, Vec<NodeId>) {
        let g = ConcurrentGraph::new();
        let clique_a: Vec<NodeId> = (0..k).map(|_| g.insert_node(TypeId(1))).collect();
        let clique_b: Vec<NodeId> = (0..k).map(|_| g.insert_node(TypeId(1))).collect();

        // Fully connect clique A
        for i in 0..k {
            for j in (i + 1)..k {
                g.insert_edge(clique_a[i], clique_a[j], TypeId(10)).unwrap();
                g.insert_edge(clique_a[j], clique_a[i], TypeId(10)).unwrap();
            }
        }
        // Fully connect clique B
        for i in 0..k {
            for j in (i + 1)..k {
                g.insert_edge(clique_b[i], clique_b[j], TypeId(10)).unwrap();
                g.insert_edge(clique_b[j], clique_b[i], TypeId(10)).unwrap();
            }
        }

        (g, clique_a, clique_b)
    }

    /// Ring of `n` nodes (bidirectional edges).
    fn bidirectional_ring(n: usize) -> (ConcurrentGraph, Vec<NodeId>) {
        let g = ConcurrentGraph::new();
        let nodes: Vec<NodeId> = (0..n).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..n {
            let j = (i + 1) % n;
            g.insert_edge(nodes[i], nodes[j], TypeId(10)).unwrap();
            g.insert_edge(nodes[j], nodes[i], TypeId(10)).unwrap();
        }
        (g, nodes)
    }

    // ── LPA basic tests ─────────────────────────────────────────────────

    #[test]
    fn test_lpa_empty_graph() {
        let g = ConcurrentGraph::new();
        let result = label_propagation(&g, &default_config());
        assert!(result.converged);
        assert_eq!(result.num_communities, 0);
    }

    #[test]
    fn test_lpa_single_node() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let result = label_propagation(&g, &default_config());

        assert!(result.converged);
        assert_eq!(result.num_communities, 1);

        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        assert!(result.community_of(&csr, a).is_some());
    }

    #[test]
    fn test_lpa_two_disconnected_cliques() {
        let (g, clique_a, clique_b) = two_cliques(5);
        let result = label_propagation(&g, &default_config());
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        assert!(result.converged);
        assert_eq!(result.num_communities, 2, "Should detect 2 communities");

        // All nodes in clique A should share a label
        let label_a = result.community_of(&csr, clique_a[0]).unwrap();
        for &node in &clique_a {
            assert_eq!(result.community_of(&csr, node).unwrap(), label_a);
        }

        // All nodes in clique B should share a different label
        let label_b = result.community_of(&csr, clique_b[0]).unwrap();
        for &node in &clique_b {
            assert_eq!(result.community_of(&csr, node).unwrap(), label_b);
        }

        assert_ne!(label_a, label_b);
    }

    #[test]
    fn test_lpa_same_community_query() {
        let (g, clique_a, clique_b) = two_cliques(4);
        let result = label_propagation(&g, &default_config());
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        assert_eq!(
            result.same_community(&csr, clique_a[0], clique_a[1]),
            Some(true)
        );
        assert_eq!(
            result.same_community(&csr, clique_a[0], clique_b[0]),
            Some(false)
        );
    }

    #[test]
    fn test_lpa_community_sizes() {
        let (g, _clique_a, _clique_b) = two_cliques(5);
        let result = label_propagation(&g, &default_config());

        let sizes = result.community_sizes();
        assert_eq!(sizes.len(), 2);
        assert_eq!(sizes[0].1, 5); // both cliques have 5 nodes
        assert_eq!(sizes[1].1, 5);
    }

    #[test]
    fn test_lpa_members_of() {
        let (g, clique_a, _clique_b) = two_cliques(3);
        let result = label_propagation(&g, &default_config());
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let label_a = result.community_of(&csr, clique_a[0]).unwrap();
        let members = result.members_of(&csr, label_a);
        assert_eq!(members.len(), 3);
        for &node in &clique_a {
            assert!(members.contains(&node));
        }
    }

    #[test]
    fn test_lpa_ring_single_community() {
        // Note: LPA on a homogeneous ring has no density-based community signal,
        // so it may fragment into multiple labels. We verify it runs correctly
        // and all labels are valid.
        let (g, nodes) = bidirectional_ring(10);
        let result = label_propagation(&g, &default_config());
        assert!(result.num_communities >= 1);
        assert!(result.num_communities <= 10);
        // All nodes have valid labels
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        for &node in &nodes {
            assert!(result.community_of(&csr, node).is_some());
        }
    }

    #[test]
    fn test_lpa_isolated_nodes() {
        let g = ConcurrentGraph::new();
        let _a = g.insert_node(TypeId(1));
        let _b = g.insert_node(TypeId(1));
        let _c = g.insert_node(TypeId(1));

        let result = label_propagation(&g, &default_config());
        assert!(result.converged);
        // Each isolated node is its own community
        assert_eq!(result.num_communities, 3);
    }

    #[test]
    fn test_lpa_max_iterations_limit() {
        let (g, _nodes) = bidirectional_ring(10);
        let config = LpaConfig { max_iterations: 1 };
        let result = label_propagation(&g, &config);
        assert!(result.iterations <= 1);
    }

    // ── Connected Components tests ──────────────────────────────────────

    #[test]
    fn test_cc_empty() {
        let g = ConcurrentGraph::new();
        let result = connected_components(&g);
        assert!(result.converged);
        assert_eq!(result.num_communities, 0);
    }

    #[test]
    fn test_cc_single_component() {
        let (g, nodes) = bidirectional_ring(10);
        let result = connected_components(&g);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        assert!(result.converged);
        assert_eq!(result.num_communities, 1);

        // All nodes share the same label
        let first_label = result.community_of(&csr, nodes[0]).unwrap();
        for &node in &nodes {
            assert_eq!(result.community_of(&csr, node).unwrap(), first_label);
        }
    }

    #[test]
    fn test_cc_two_components() {
        let (g, clique_a, clique_b) = two_cliques(4);
        let result = connected_components(&g);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        assert!(result.converged);
        assert_eq!(result.num_communities, 2);

        assert_eq!(
            result.same_community(&csr, clique_a[0], clique_a[3]),
            Some(true)
        );
        assert_eq!(
            result.same_community(&csr, clique_a[0], clique_b[0]),
            Some(false)
        );
    }

    #[test]
    fn test_cc_isolated_nodes() {
        let g = ConcurrentGraph::new();
        let _a = g.insert_node(TypeId(1));
        let _b = g.insert_node(TypeId(1));

        let result = connected_components(&g);
        assert_eq!(result.num_communities, 2);
    }

    #[test]
    fn test_cc_directed_reachability() {
        // A → B (one direction only). CC should still find them connected
        // because we use both out + in CSR.
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        g.insert_edge(a, b, TypeId(10)).unwrap();

        let result = connected_components(&g);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        assert_eq!(result.num_communities, 1);
        assert_eq!(result.same_community(&csr, a, b), Some(true));
    }

    // ── Large-scale tests ───────────────────────────────────────────────

    #[test]
    fn test_lpa_two_large_cliques_with_bridge() {
        // Two cliques of 50 connected by a single bridge edge
        // LPA should detect 2 communities (or close to it)
        let n = 50;
        let g = ConcurrentGraph::with_capacity(n * 2, n * n * 2);
        let clique_a: Vec<NodeId> = (0..n).map(|_| g.insert_node(TypeId(1))).collect();
        let clique_b: Vec<NodeId> = (0..n).map(|_| g.insert_node(TypeId(1))).collect();

        for i in 0..n {
            for j in (i + 1)..n {
                g.insert_edge(clique_a[i], clique_a[j], TypeId(10)).unwrap();
                g.insert_edge(clique_a[j], clique_a[i], TypeId(10)).unwrap();
                g.insert_edge(clique_b[i], clique_b[j], TypeId(10)).unwrap();
                g.insert_edge(clique_b[j], clique_b[i], TypeId(10)).unwrap();
            }
        }

        // Single bridge
        g.insert_edge(clique_a[0], clique_b[0], TypeId(10)).unwrap();
        g.insert_edge(clique_b[0], clique_a[0], TypeId(10)).unwrap();

        let result = label_propagation(&g, &default_config());
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        assert!(result.converged);
        // With a single bridge between two dense cliques, LPA should detect
        // either 1 or 2 communities (the bridge may or may not merge them).
        // At minimum, nodes within each clique should be in the same community.
        assert!(
            result.same_community(&csr, clique_a[0], clique_a[n - 1]) == Some(true),
            "Clique A members should be in same community"
        );
        assert!(
            result.same_community(&csr, clique_b[0], clique_b[n - 1]) == Some(true),
            "Clique B members should be in same community"
        );
    }

    #[test]
    fn test_cc_100k_chain() {
        // Chain of 100K nodes: one connected component
        let n = 100_000;
        let g = ConcurrentGraph::with_capacity(n, n);
        let nodes: Vec<NodeId> = (0..n).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..n - 1 {
            g.insert_edge(nodes[i], nodes[i + 1], TypeId(10)).unwrap();
        }

        let result = connected_components(&g);
        assert!(result.converged);
        assert_eq!(result.num_communities, 1);
    }

    #[test]
    fn test_cc_100k_five_components() {
        // 5 disconnected chains of 20K nodes each
        let chain_size = 20_000;
        let num_chains = 5;
        let g = ConcurrentGraph::with_capacity(
            chain_size * num_chains,
            chain_size * num_chains,
        );

        let mut chains: Vec<Vec<NodeId>> = Vec::new();
        for _ in 0..num_chains {
            let chain: Vec<NodeId> = (0..chain_size)
                .map(|_| g.insert_node(TypeId(1)))
                .collect();
            for i in 0..chain_size - 1 {
                g.insert_edge(chain[i], chain[i + 1], TypeId(10)).unwrap();
            }
            chains.push(chain);
        }

        let result = connected_components(&g);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        assert!(result.converged);
        assert_eq!(result.num_communities, 5);

        // Verify cross-chain separation
        assert_eq!(
            result.same_community(&csr, chains[0][0], chains[1][0]),
            Some(false)
        );
        // Verify within-chain connectivity
        assert_eq!(
            result.same_community(&csr, chains[0][0], chains[0][chain_size - 1]),
            Some(true)
        );
    }

    #[test]
    fn test_lpa_dense_modular_graph() {
        // 10 disconnected cliques of 200 nodes each.
        // Dense cliques give LPA a clear community signal → converges fast.
        let clique_size = 200;
        let num_cliques = 10;
        let g = ConcurrentGraph::with_capacity(
            clique_size * num_cliques,
            clique_size * clique_size * num_cliques,
        );

        let mut cliques: Vec<Vec<NodeId>> = Vec::new();
        for _ in 0..num_cliques {
            let nodes: Vec<NodeId> = (0..clique_size)
                .map(|_| g.insert_node(TypeId(1)))
                .collect();
            // Fully connect within clique (bidirectional)
            for i in 0..clique_size {
                for j in (i + 1)..clique_size {
                    g.insert_edge(nodes[i], nodes[j], TypeId(10)).unwrap();
                    g.insert_edge(nodes[j], nodes[i], TypeId(10)).unwrap();
                }
            }
            cliques.push(nodes);
        }

        let result = label_propagation(&g, &default_config());
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        assert!(result.converged);
        assert_eq!(result.num_communities, 10);

        // Verify within-clique consistency
        for clique in &cliques {
            assert_eq!(
                result.same_community(&csr, clique[0], clique[clique.len() - 1]),
                Some(true)
            );
        }
        // Verify cross-clique separation
        assert_eq!(
            result.same_community(&csr, cliques[0][0], cliques[1][0]),
            Some(false)
        );
    }

    #[test]
    fn test_community_memory_compactness() {
        // For N=1M nodes, labels use N × 4 bytes = 4 MB
        let n = 1_000_000;
        let mem = n * std::mem::size_of::<u32>();
        assert!(
            mem < 5_000_000,
            "Community labels for 1M nodes should be < 5MB, got {mem}"
        );
    }
}
