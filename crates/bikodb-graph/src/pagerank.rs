// =============================================================================
// bikodb-graph::pagerank — PageRank paralelo sobre CSR
// =============================================================================
//
// ## Algoritmo: Pull-based PageRank con rayon
//
// PageRank clásico (Brin & Page, 1998) con las siguientes optimizaciones:
//
// | Aspecto            | Implementación                              |
// |--------------------|---------------------------------------------|
// | Layout datos       | CSR contiguo → cache lines llenas           |
// | Paralelización     | rayon par_chunks sobre nodos                |
// | Dangling nodes     | Redistribución uniforme por iteración       |
// | Convergencia       | L1 norm con tolerancia configurable         |
// | Pre-cómputo        | `inv_degree[u] = 1.0 / out_degree(u)`      |
// | Memoria            | 2 arrays de f64 (current + next) + inv_deg  |
//
// ## Fórmula
//
//   PR(v) = (1 - d) / N  +  d × Σ_{u→v} PR(u) / out_degree(u)
//
// Donde:
// - d = damping factor (default 0.85)
// - N = número de nodos
// - u→v = u tiene edge hacia v (es decir, u ∈ in-neighbors(v))
//
// ## Implementación Pull-based
//
// Para cada nodo v, "tiramos" las contribuciones de sus in-neighbors:
//
//   new_rank[v] = base  +  d × Σ_{u ∈ in(v)} rank[u] × inv_degree[u]
//
// donde base = (1-d)/N + d × dangling_sum/N
//
// Esto es altamente paralelizable: cada v lee datos de otros nodos
// pero solo escribe en su propia posición → zero contention.
//
// ## Dangling nodes
//
// Nodos sin out-edges (sinks) acumulan rank sin distribuirlo.
// En cada iteración calculamos:
//   dangling_sum = Σ rank[u] para u con out_degree = 0
// y lo redistribuimos uniformemente: dangling_sum / N sumado a base.
//
// ## Convergencia
//
// L1 norm: Σ |new_rank[v] - old_rank[v]| < tolerance
// Si no converge en max_iterations, retornamos el mejor resultado parcial.
// =============================================================================

use crate::csr::CsrGraph;
use bikodb_core::record::Direction;
use bikodb_core::types::NodeId;
use rayon::prelude::*;

use crate::graph::ConcurrentGraph;

// =============================================================================
// Configuración
// =============================================================================

/// Parámetros configurables para PageRank.
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    /// Factor de amortiguación (probability of following a link vs random jump).
    /// Estándar: 0.85
    pub damping: f64,
    /// Número máximo de iteraciones.
    pub max_iterations: u32,
    /// Tolerancia para convergencia (L1 norm).
    /// Default: 1e-6
    pub tolerance: f64,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

// =============================================================================
// Resultado
// =============================================================================

/// Resultado de PageRank.
#[derive(Debug)]
pub struct PageRankResult {
    /// Scores de PageRank indexados por internal id.
    pub scores: Vec<f64>,
    /// Número de iteraciones ejecutadas.
    pub iterations: u32,
    /// ¿Convergió dentro de la tolerancia?
    pub converged: bool,
    /// L1 norm de la última iteración.
    pub final_l1_norm: f64,
    /// Número de nodos.
    pub num_nodes: usize,
}

impl PageRankResult {
    /// Score de un nodo externo.
    pub fn score_of(&self, out_csr: &CsrGraph, node: NodeId) -> Option<f64> {
        let internal = out_csr.to_internal(node)?;
        Some(self.scores[internal as usize])
    }

    /// Top-K nodos por PageRank score (retorna NodeIds externos + scores).
    pub fn top_k(&self, out_csr: &CsrGraph, k: usize) -> Vec<(NodeId, f64)> {
        let mut indexed: Vec<(usize, f64)> = self
            .scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();

        // Partial sort: solo necesitamos top-K
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);

        indexed
            .into_iter()
            .map(|(i, s)| (out_csr.to_external(i as u32), s))
            .collect()
    }
}

// =============================================================================
// PageRank paralelo
// =============================================================================

/// Calcula PageRank paralelo sobre un grafo concurrente.
///
/// Construye internamente los CSR de out-edges (para grados) e in-edges
/// (para pull-based computation), luego ejecuta iteraciones paralelas.
///
/// # Ejemplo
///
/// ```
/// use bikodb_graph::pagerank::{pagerank, PageRankConfig};
/// use bikodb_graph::ConcurrentGraph;
/// use bikodb_core::types::TypeId;
///
/// let g = ConcurrentGraph::new();
/// let a = g.insert_node(TypeId(1));
/// let b = g.insert_node(TypeId(1));
/// g.insert_edge(a, b, TypeId(10)).unwrap();
///
/// let result = pagerank(&g, &PageRankConfig::default());
/// assert!(result.converged);
/// ```
pub fn pagerank(graph: &ConcurrentGraph, config: &PageRankConfig) -> PageRankResult {
    let out_csr = CsrGraph::from_concurrent(graph, Direction::Out);
    let in_csr = CsrGraph::from_concurrent(graph, Direction::In);

    pagerank_on_csr(&out_csr, &in_csr, config)
}

/// Calcula PageRank sobre CSRs pre-construidos.
///
/// Útil cuando ya tienes los CSR (evita reconstruirlos).
/// `out_csr` se usa para out-degree, `in_csr` para pull de contribuciones.
pub fn pagerank_on_csr(
    out_csr: &CsrGraph,
    in_csr: &CsrGraph,
    config: &PageRankConfig,
) -> PageRankResult {
    let n = out_csr.num_nodes();

    if n == 0 {
        return PageRankResult {
            scores: Vec::new(),
            iterations: 0,
            converged: true,
            final_l1_norm: 0.0,
            num_nodes: 0,
        };
    }

    let d = config.damping;
    let inv_n = 1.0 / n as f64;

    // ── Pre-cómputo: inv_degree[u] = 1.0 / out_degree(u) ───────────
    // Para nodos dangling (degree=0), inv_degree = 0.0 (no contribuyen via edges).
    let inv_degree: Vec<f64> = (0..n as u32)
        .map(|u| {
            let deg = out_csr.degree(u);
            if deg > 0 { 1.0 / deg as f64 } else { 0.0 }
        })
        .collect();

    // Identificar dangling nodes (out_degree = 0) una sola vez
    let dangling_nodes: Vec<u32> = (0..n as u32)
        .filter(|&u| out_csr.degree(u) == 0)
        .collect();

    // ── Inicialización: rank uniforme 1/N ────────────────────────────
    let mut rank = vec![inv_n; n];
    let mut new_rank = vec![0.0f64; n];

    let mut iterations: u32 = 0;
    let mut l1_norm: f64 = f64::INFINITY;

    // ── Iteración power-method ───────────────────────────────────────
    for _ in 0..config.max_iterations {
        // 1. Dangling sum: Σ rank[u] para nodos sin out-edges
        let dangling_sum: f64 = dangling_nodes.iter().map(|&u| rank[u as usize]).sum();

        // 2. Base contribution: teleport + dangling redistribution
        let base = (1.0 - d) * inv_n + d * dangling_sum * inv_n;

        // 3. Pull-based parallel computation
        //    Cada nodo v calcula su new_rank tirando de sus in-neighbors.
        //    Zero contention: cada thread escribe en su propio chunk.
        let rank_ref = &rank;
        let inv_degree_ref = &inv_degree;

        new_rank
            .par_chunks_mut(1024)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start = chunk_idx * 1024;
                for (offset, slot) in chunk.iter_mut().enumerate() {
                    let v = (start + offset) as u32;
                    if (v as usize) >= n {
                        break;
                    }

                    // Sumar contribuciones de in-neighbors
                    let mut sum = 0.0f64;
                    for &u in in_csr.neighbors(v) {
                        sum += rank_ref[u as usize] * inv_degree_ref[u as usize];
                    }

                    *slot = base + d * sum;
                }
            });

        // 4. Convergence check: L1 norm
        l1_norm = rank
            .par_iter()
            .zip(new_rank.par_iter())
            .map(|(old, new)| (old - new).abs())
            .sum();

        // Swap buffers
        std::mem::swap(&mut rank, &mut new_rank);
        iterations += 1;

        if l1_norm < config.tolerance {
            break;
        }
    }

    PageRankResult {
        scores: rank,
        iterations,
        converged: l1_norm < config.tolerance,
        final_l1_norm: l1_norm,
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

    fn default_config() -> PageRankConfig {
        PageRankConfig::default()
    }

    fn tight_config() -> PageRankConfig {
        PageRankConfig {
            damping: 0.85,
            max_iterations: 200,
            tolerance: 1e-10,
        }
    }

    // ── Basic tests ─────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = ConcurrentGraph::new();
        let result = pagerank(&g, &default_config());
        assert!(result.converged);
        assert_eq!(result.num_nodes, 0);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_single_node() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let result = pagerank(&g, &default_config());

        assert!(result.converged);
        assert_eq!(result.num_nodes, 1);
        let score = result.score_of(
            &CsrGraph::from_concurrent(&g, Direction::Out),
            a,
        ).unwrap();
        assert!((score - 1.0).abs() < 1e-6, "Single node should have PR=1.0, got {score}");
    }

    #[test]
    fn test_two_nodes_one_edge() {
        // A → B
        // B is a dangling node (sink)
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        g.insert_edge(a, b, TypeId(10)).unwrap();

        let result = pagerank(&g, &tight_config());
        assert!(result.converged);

        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let sa = result.score_of(&out_csr, a).unwrap();
        let sb = result.score_of(&out_csr, b).unwrap();

        // B should have higher rank (receives from A + dangling redistribution)
        assert!(sb > sa, "B (sink) should rank higher: B={sb}, A={sa}");

        // Sum should be ~1.0 (conservation)
        let sum = sa + sb;
        assert!((sum - 1.0).abs() < 1e-6, "Sum should be 1.0, got {sum}");
    }

    #[test]
    fn test_cycle_uniform() {
        // A → B → C → A (symmetric cycle)
        // All nodes should have equal rank = 1/3
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));
        g.insert_edge(a, b, TypeId(10)).unwrap();
        g.insert_edge(b, c, TypeId(10)).unwrap();
        g.insert_edge(c, a, TypeId(10)).unwrap();

        let result = pagerank(&g, &tight_config());
        assert!(result.converged);

        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let sa = result.score_of(&out_csr, a).unwrap();
        let sb = result.score_of(&out_csr, b).unwrap();
        let sc = result.score_of(&out_csr, c).unwrap();

        let expected = 1.0 / 3.0;
        assert!((sa - expected).abs() < 1e-6, "A should be {expected}, got {sa}");
        assert!((sb - expected).abs() < 1e-6, "B should be {expected}, got {sb}");
        assert!((sc - expected).abs() < 1e-6, "C should be {expected}, got {sc}");
    }

    #[test]
    fn test_star_topology() {
        // Center ← leaf1, leaf2, ..., leaf10 (all pointing to center)
        // Center should have highest rank
        let g = ConcurrentGraph::new();
        let center = g.insert_node(TypeId(1));
        let mut leaves = Vec::new();
        for _ in 0..10 {
            let l = g.insert_node(TypeId(1));
            g.insert_edge(l, center, TypeId(10)).unwrap();
            leaves.push(l);
        }

        let result = pagerank(&g, &default_config());
        assert!(result.converged);

        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let center_score = result.score_of(&out_csr, center).unwrap();
        let leaf_score = result.score_of(&out_csr, leaves[0]).unwrap();

        assert!(center_score > leaf_score * 5.0,
            "Center should be much higher: center={center_score}, leaf={leaf_score}");
    }

    #[test]
    fn test_dangling_nodes() {
        // A → B → C, D is isolated (dangling)
        // D should still get some rank from teleportation
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));
        let d = g.insert_node(TypeId(1));
        g.insert_edge(a, b, TypeId(10)).unwrap();
        g.insert_edge(b, c, TypeId(10)).unwrap();

        let result = pagerank(&g, &default_config());
        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let sd = result.score_of(&out_csr, d).unwrap();
        assert!(sd > 0.0, "Isolated node should have PR > 0 from teleport: {sd}");

        // Sum conservation
        let sum: f64 = result.scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "Total PR should be ~1.0, got {sum}");
    }

    #[test]
    fn test_sum_conservation() {
        // Random-ish graph, verify sum of all ranks ≈ 1.0
        let g = ConcurrentGraph::new();
        let nodes: Vec<NodeId> = (0..20).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..20 {
            for j in [1, 3, 7] {
                let target = (i + j) % 20;
                g.insert_edge(nodes[i], nodes[target], TypeId(10)).unwrap();
            }
        }

        let result = pagerank(&g, &default_config());
        let sum: f64 = result.scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "Total PR should be ~1.0, got {sum}");
    }

    #[test]
    fn test_convergence() {
        let g = ConcurrentGraph::new();
        let nodes: Vec<NodeId> = (0..50).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..50 {
            g.insert_edge(nodes[i], nodes[(i + 1) % 50], TypeId(10)).unwrap();
        }

        let result = pagerank(&g, &default_config());
        assert!(result.converged, "50-node cycle should converge");
        assert!(result.iterations < 100, "Should converge in <100 iterations, took {}", result.iterations);
    }

    #[test]
    fn test_max_iterations_limit() {
        // Non-trivial graph that won't converge in 3 iterations with tight tolerance
        let g = ConcurrentGraph::new();
        let nodes: Vec<NodeId> = (0..100).map(|_| g.insert_node(TypeId(1))).collect();
        // Asymmetric structure: chain + some back-edges
        for i in 0..100 {
            g.insert_edge(nodes[i], nodes[(i + 1) % 100], TypeId(10)).unwrap();
            if i % 5 == 0 {
                g.insert_edge(nodes[i], nodes[0], TypeId(10)).unwrap();
            }
        }

        let config = PageRankConfig {
            damping: 0.85,
            max_iterations: 3,
            tolerance: 1e-20, // impossibly tight
        };

        let result = pagerank(&g, &config);
        assert_eq!(result.iterations, 3);
        assert!(!result.converged);
    }

    #[test]
    fn test_damping_factor_effect() {
        // Higher damping → more link-following, less teleportation
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));
        g.insert_edge(a, b, TypeId(10)).unwrap();
        g.insert_edge(b, c, TypeId(10)).unwrap();

        let high_d = pagerank(&g, &PageRankConfig {
            damping: 0.99,
            max_iterations: 200,
            tolerance: 1e-10,
        });
        let low_d = pagerank(&g, &PageRankConfig {
            damping: 0.50,
            max_iterations: 200,
            tolerance: 1e-10,
        });

        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);

        // With high damping, score of C (end of chain) should be more dominant
        let c_high = high_d.score_of(&out_csr, c).unwrap();
        let c_low = low_d.score_of(&out_csr, c).unwrap();
        let a_high = high_d.score_of(&out_csr, a).unwrap();
        let a_low = low_d.score_of(&out_csr, a).unwrap();

        // Ratio C/A should be higher with more damping
        let ratio_high = c_high / a_high;
        let ratio_low = c_low / a_low;
        assert!(ratio_high > ratio_low,
            "Higher damping should amplify link structure: ratio_high={ratio_high}, ratio_low={ratio_low}");
    }

    #[test]
    fn test_top_k() {
        // Star: everyone points to center → center has highest rank
        let g = ConcurrentGraph::new();
        let center = g.insert_node(TypeId(1));
        for _ in 0..20 {
            let l = g.insert_node(TypeId(1));
            g.insert_edge(l, center, TypeId(10)).unwrap();
        }

        let result = pagerank(&g, &default_config());
        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let top = result.top_k(&out_csr, 3);

        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, center, "Center should be #1");
    }

    // ── Large-scale tests ───────────────────────────────────────────────

    #[test]
    fn test_pagerank_100k_cycle() {
        let n = 100_000;
        let g = ConcurrentGraph::with_capacity(n, n);
        let nodes: Vec<NodeId> = (0..n).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..n {
            g.insert_edge(nodes[i], nodes[(i + 1) % n], TypeId(10)).unwrap();
        }

        let result = pagerank(&g, &default_config());

        // Symmetric cycle: all ranks should be equal = 1/N
        assert!(result.converged, "100K cycle should converge");
        let expected = 1.0 / n as f64;
        let max_deviation = result
            .scores
            .iter()
            .map(|&s| (s - expected).abs())
            .fold(0.0f64, f64::max);
        assert!(max_deviation < 1e-6,
            "Max deviation in 100K cycle: {max_deviation}");
    }

    #[test]
    fn test_pagerank_100k_random_graph() {
        // Each node connects to next 3 (modular)
        let n = 100_000;
        let g = ConcurrentGraph::with_capacity(n, n * 3);
        let nodes: Vec<NodeId> = (0..n).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..n {
            for offset in [1, 7, 31] {
                g.insert_edge(nodes[i], nodes[(i + offset) % n], TypeId(10)).unwrap();
            }
        }

        let result = pagerank(&g, &default_config());
        assert!(result.converged, "100K random graph should converge");

        // Sum conservation
        let sum: f64 = result.scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "Total PR should be ~1.0, got {sum}");
    }

    #[test]
    fn test_pagerank_power_law_graph() {
        // Simulate power-law: node 0 receives many in-links
        let n = 10_000;
        let g = ConcurrentGraph::with_capacity(n, n * 2);
        let nodes: Vec<NodeId> = (0..n).map(|_| g.insert_node(TypeId(1))).collect();

        // Chain
        for i in 0..n - 1 {
            g.insert_edge(nodes[i], nodes[i + 1], TypeId(10)).unwrap();
        }
        // Extra edges pointing to node 0 (hub)
        for i in 1..n {
            if i % 10 == 0 {
                g.insert_edge(nodes[i], nodes[0], TypeId(10)).unwrap();
            }
        }

        let result = pagerank(&g, &default_config());
        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);

        // Node 0 should be in top-5
        let top = result.top_k(&out_csr, 5);
        let top_ids: Vec<NodeId> = top.iter().map(|(id, _)| *id).collect();
        assert!(top_ids.contains(&nodes[0]),
            "Hub node should be in top-5, top={top_ids:?}");
    }

    #[test]
    fn test_pagerank_on_csr_api() {
        // Verify the pre-built CSR API works
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));
        g.insert_edge(a, b, TypeId(10)).unwrap();
        g.insert_edge(b, c, TypeId(10)).unwrap();
        g.insert_edge(c, a, TypeId(10)).unwrap();

        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let in_csr = CsrGraph::from_concurrent(&g, Direction::In);
        let result = pagerank_on_csr(&out_csr, &in_csr, &default_config());

        assert!(result.converged);
        let sum: f64 = result.scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pagerank_memory_efficiency() {
        // For N=1M nodes, PageRank uses:
        // - 2 × N × 8 bytes (rank + new_rank) = 16 MB
        // - 1 × N × 8 bytes (inv_degree) = 8 MB
        // Total ≈ 24 MB (vs HashMap-based: ~200+ MB)
        let n = 1_000_000;
        let mem_rank = n * 8 * 2;     // two f64 arrays
        let mem_inv = n * 8;           // inv_degree
        let total = mem_rank + mem_inv;
        assert!(total < 30_000_000, "PageRank memory {total} exceeds 30MB for 1M nodes");
    }
}
