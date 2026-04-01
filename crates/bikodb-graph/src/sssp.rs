// =============================================================================
// bikodb-graph::sssp — Single-Source Shortest Path adaptativo y multithreading
// =============================================================================
//
// ## Algoritmos incluidos
//
// | Algoritmo        | Pesos           | Paralelo | Complejidad      |
// |------------------|-----------------|----------|------------------|
// | BFS (hop-count)  | Unitarios (1.0) | Sí       | O(V+E)/P         |
// | Dijkstra         | No negativos    | No       | O((V+E) log V)   |
// | Bellman-Ford     | Cualquiera      | No       | O(V·E)           |
// | Δ-Stepping       | No negativos    | Sí       | O(V+E)/P aprox   |
//
// ## Selector adaptativo
//
// `sssp()` inspecciona las propiedades del grafo y elige automáticamente:
//
// 1. Todos pesos = 1.0 → **Parallel BFS** (level-synchronous, rayon)
// 2. Pesos ≥ 0, grafo pequeño (<50K nodos) → **Dijkstra** (BinaryHeap)
// 3. Pesos ≥ 0, grafo grande (≥50K nodos) → **Δ-Stepping paralelo** (rayon)
// 4. Pesos negativos → **Bellman-Ford** (con detección de ciclos negativos)
//
// ## Δ-Stepping (Meyer & Sanders, 2003)
//
// Variante paralela de Dijkstra. Agrupa nodos en "buckets" de ancho Δ:
// - Bucket i contiene nodos con tentative distance en [i·Δ, (i+1)·Δ)
// - Edges "ligeros" (peso < Δ) se relajan dentro del bucket (paralelo)
// - Edges "pesados" (peso ≥ Δ) se relajan al procesar el bucket
// - Avanza al siguiente bucket no vacío
//
// Cuando Δ → 0 se aproxima a Dijkstra; cuando Δ → ∞ se aproxima a Bellman-Ford.
// Buen balance con Δ = max_weight / log(V).
// =============================================================================

use crate::bitset::{AtomicBitSet, BitSet};
use crate::weighted_csr::WeightedCsrGraph;
use bikodb_core::types::NodeId;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

// =============================================================================
// Resultado SSSP
// =============================================================================

/// Resultado de SSSP: distancias y predecesores como arrays densos.
#[derive(Debug)]
pub struct SsspResult {
    /// Distancia desde el nodo inicio. `f64::INFINITY` si no alcanzable.
    pub distances: Vec<f64>,
    /// Predecesor (internal id). `u32::MAX` si raíz o no alcanzable.
    pub predecessors: Vec<u32>,
    /// Número de nodos alcanzados.
    pub nodes_reached: usize,
    /// Algoritmo que se utilizó.
    pub algorithm: SsspAlgorithm,
}

/// Algoritmo seleccionado por el dispatcher adaptativo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsspAlgorithm {
    Bfs,
    Dijkstra,
    BellmanFord,
    DeltaStepping,
}

impl SsspResult {
    /// Reconstruye el path (NodeIds externos) desde start hasta target.
    pub fn path_to(&self, wcsr: &WeightedCsrGraph, target: NodeId) -> Option<Vec<NodeId>> {
        let internal = wcsr.to_internal(target)?;
        if self.distances[internal as usize].is_infinite() {
            return None;
        }

        let mut path = Vec::new();
        let mut current = internal;
        loop {
            path.push(wcsr.to_external(current));
            let pred = self.predecessors[current as usize];
            if pred == u32::MAX {
                break;
            }
            current = pred;
        }
        path.reverse();
        Some(path)
    }

    /// Distancia a un nodo externo. None si no alcanzable.
    pub fn distance_to(&self, wcsr: &WeightedCsrGraph, target: NodeId) -> Option<f64> {
        let internal = wcsr.to_internal(target)?;
        let d = self.distances[internal as usize];
        if d.is_infinite() { None } else { Some(d) }
    }
}

// =============================================================================
// Selector adaptativo
// =============================================================================

/// Threshold para elegir Δ-Stepping paralelo vs Dijkstra secuencial.
const PARALLEL_THRESHOLD: usize = 50_000;

/// SSSP adaptativo: elige el mejor algoritmo según propiedades del grafo.
///
/// Inspecciona `WeightedCsrGraph` y despacha a:
/// - BFS paralelo si todos los pesos son 1.0
/// - Dijkstra si pesos ≥ 0 y grafo < 50K nodos
/// - Δ-Stepping paralelo si pesos ≥ 0 y grafo ≥ 50K nodos
/// - Bellman-Ford si hay pesos negativos
pub fn sssp(
    wcsr: &WeightedCsrGraph,
    start: NodeId,
) -> Option<SsspResult> {
    if wcsr.all_unit_weights() {
        bfs_sssp(wcsr, start)
    } else if wcsr.has_negative_weights() {
        bellman_ford(wcsr, start)
    } else if wcsr.num_nodes() >= PARALLEL_THRESHOLD {
        delta_stepping(wcsr, start, None)
    } else {
        dijkstra(wcsr, start)
    }
}

/// SSSP forzando un algoritmo específico. Útil para benchmarking.
pub fn sssp_with_algorithm(
    wcsr: &WeightedCsrGraph,
    start: NodeId,
    algorithm: SsspAlgorithm,
) -> Option<SsspResult> {
    match algorithm {
        SsspAlgorithm::Bfs => bfs_sssp(wcsr, start),
        SsspAlgorithm::Dijkstra => dijkstra(wcsr, start),
        SsspAlgorithm::BellmanFord => bellman_ford(wcsr, start),
        SsspAlgorithm::DeltaStepping => delta_stepping(wcsr, start, None),
    }
}

// =============================================================================
// BFS SSSP (unweighted / unit weights)
// =============================================================================

/// BFS level-synchronous con rayon para grafos con pesos unitarios.
fn bfs_sssp(
    wcsr: &WeightedCsrGraph,
    start: NodeId,
) -> Option<SsspResult> {
    let start_internal = wcsr.to_internal(start)?;
    let n = wcsr.num_nodes();

    let mut distances = vec![f64::INFINITY; n];
    let mut predecessors = vec![u32::MAX; n];

    if n < 1024 {
        // Small graph: sequential BFS
        let mut visited = BitSet::new(n);
        visited.set_and_check(start_internal);
        distances[start_internal as usize] = 0.0;

        let mut frontier = vec![start_internal];
        let mut depth: u32 = 0;

        while !frontier.is_empty() {
            let mut next = Vec::new();
            for &node in &frontier {
                for &neighbor in wcsr.neighbors(node) {
                    if visited.set_and_check(neighbor) {
                        distances[neighbor as usize] = (depth + 1) as f64;
                        predecessors[neighbor as usize] = node;
                        next.push(neighbor);
                    }
                }
            }
            frontier = next;
            depth += 1;
        }
    } else {
        // Large graph: parallel BFS with AtomicBitSet
        let visited = AtomicBitSet::new(n);
        visited.set_and_check(start_internal);
        distances[start_internal as usize] = 0.0;

        let mut frontier = vec![start_internal];
        let mut depth: u32 = 0;

        while !frontier.is_empty() {
            let next: Vec<(u32, u32)> = frontier
                .par_iter()
                .flat_map(|&node| {
                    let mut local = Vec::new();
                    for &neighbor in wcsr.neighbors(node) {
                        if visited.set_and_check(neighbor) {
                            local.push((neighbor, node));
                        }
                    }
                    local
                })
                .collect();

            for &(neighbor, parent) in &next {
                distances[neighbor as usize] = (depth + 1) as f64;
                predecessors[neighbor as usize] = parent;
            }

            frontier = next.into_iter().map(|(n, _)| n).collect();
            depth += 1;
        }
    }

    let nodes_reached = distances.iter().filter(|d| d.is_finite()).count();
    Some(SsspResult {
        distances,
        predecessors,
        nodes_reached,
        algorithm: SsspAlgorithm::Bfs,
    })
}

// =============================================================================
// Dijkstra (secuencial, BinaryHeap)
// =============================================================================

/// Dijkstra clásico sobre WeightedCsrGraph.
///
/// Usa `BinaryHeap<Reverse<(OrderedFloat, u32)>>` para min-heap.
/// Complejidad: O((V + E) log V).
///
/// # Precondición
/// Todos los pesos deben ser ≥ 0.
pub fn dijkstra(
    wcsr: &WeightedCsrGraph,
    start: NodeId,
) -> Option<SsspResult> {
    let start_internal = wcsr.to_internal(start)?;
    let n = wcsr.num_nodes();

    let mut distances = vec![f64::INFINITY; n];
    let mut predecessors = vec![u32::MAX; n];
    distances[start_internal as usize] = 0.0;

    // Min-heap: (distance, node). Reverse para convertir max-heap en min-heap.
    // Usamos u64 bits de f64 para Ord (válido porque no hay NaN en pesos).
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
    heap.push(Reverse((0u64, start_internal)));

    while let Some(Reverse((dist_bits, u))) = heap.pop() {
        let dist_u = f64::from_bits(dist_bits);

        // Skip stale entries
        if dist_u > distances[u as usize] {
            continue;
        }

        for (v, weight) in wcsr.weighted_neighbors(u) {
            let new_dist = dist_u + weight;
            if new_dist < distances[v as usize] {
                distances[v as usize] = new_dist;
                predecessors[v as usize] = u;
                heap.push(Reverse((new_dist.to_bits(), v)));
            }
        }
    }

    let nodes_reached = distances.iter().filter(|d| d.is_finite()).count();
    Some(SsspResult {
        distances,
        predecessors,
        nodes_reached,
        algorithm: SsspAlgorithm::Dijkstra,
    })
}

// =============================================================================
// Bellman-Ford (pesos negativos + detección de ciclos negativos)
// =============================================================================

/// Error de Bellman-Ford: ciclo de peso negativo detectado.
#[derive(Debug, Clone)]
pub struct NegativeCycleError;

/// Bellman-Ford sobre WeightedCsrGraph.
///
/// Soporta pesos negativos. Detecta ciclos negativos (retorna None vía
/// el wrapper `bellman_ford`, o error vía `bellman_ford_checked`).
///
/// Complejidad: O(V·E). Más lento que Dijkstra pero maneja negativos.
pub fn bellman_ford(
    wcsr: &WeightedCsrGraph,
    start: NodeId,
) -> Option<SsspResult> {
    bellman_ford_checked(wcsr, start).ok()
}

/// Bellman-Ford con resultado explícito para ciclos negativos.
pub fn bellman_ford_checked(
    wcsr: &WeightedCsrGraph,
    start: NodeId,
) -> Result<SsspResult, NegativeCycleError> {
    let start_internal = wcsr.to_internal(start).ok_or(NegativeCycleError)?;
    let n = wcsr.num_nodes();

    let mut distances = vec![f64::INFINITY; n];
    let mut predecessors = vec![u32::MAX; n];
    distances[start_internal as usize] = 0.0;

    // V-1 relaxation rounds
    for _ in 0..n.saturating_sub(1) {
        let mut changed = false;
        for u in 0..n as u32 {
            let dist_u = distances[u as usize];
            if dist_u.is_infinite() {
                continue;
            }
            for (v, weight) in wcsr.weighted_neighbors(u) {
                let new_dist = dist_u + weight;
                if new_dist < distances[v as usize] {
                    distances[v as usize] = new_dist;
                    predecessors[v as usize] = u;
                    changed = true;
                }
            }
        }
        // Early exit: no changes in this round
        if !changed {
            break;
        }
    }

    // Cycle detection: one more round
    for u in 0..n as u32 {
        let dist_u = distances[u as usize];
        if dist_u.is_infinite() {
            continue;
        }
        for (v, weight) in wcsr.weighted_neighbors(u) {
            if dist_u + weight < distances[v as usize] {
                return Err(NegativeCycleError);
            }
        }
    }

    let nodes_reached = distances.iter().filter(|d| d.is_finite()).count();
    Ok(SsspResult {
        distances,
        predecessors,
        nodes_reached,
        algorithm: SsspAlgorithm::BellmanFord,
    })
}

// =============================================================================
// Δ-Stepping (paralelo con rayon)
// =============================================================================

/// Δ-Stepping paralelo: SSSP para grafos grandes con pesos no negativos.
///
/// ## Algoritmo (Meyer & Sanders, 2003)
///
/// 1. Asignar Δ (bucket width). Auto-tuned: max_weight / log2(V).
/// 2. Buckets[i] = nodos con tentative distance en [i·Δ, (i+1)·Δ)
/// 3. Para cada bucket no vacío (de menor a mayor):
///    a. "Light phase": relaja edges con peso < Δ (paralelo, iterar hasta vacío)
///    b. "Heavy phase": relaja edges con peso ≥ Δ (paralelo, una vez)
/// 4. Avanzar al siguiente bucket
///
/// ## Parámetros
/// - `delta`: ancho del bucket. None = auto-tuned.
///
/// ## Precondición
/// Todos los pesos ≥ 0.
pub fn delta_stepping(
    wcsr: &WeightedCsrGraph,
    start: NodeId,
    delta: Option<f64>,
) -> Option<SsspResult> {
    let start_internal = wcsr.to_internal(start)?;
    let n = wcsr.num_nodes();

    if n == 0 {
        return Some(SsspResult {
            distances: Vec::new(),
            predecessors: Vec::new(),
            nodes_reached: 0,
            algorithm: SsspAlgorithm::DeltaStepping,
        });
    }

    // Auto-tune Δ
    let delta = delta.unwrap_or_else(|| {
        let mut max_w: f64 = 1.0;
        for u in 0..n as u32 {
            for (_, w) in wcsr.weighted_neighbors(u) {
                if w > max_w {
                    max_w = w;
                }
            }
        }
        let log_v = (n as f64).log2().max(1.0);
        max_w / log_v
    });

    let mut distances = vec![f64::INFINITY; n];
    let mut predecessors = vec![u32::MAX; n];
    distances[start_internal as usize] = 0.0;

    // Bucket storage: Vec<Vec<u32>>. Grows dynamically as needed.
    let initial_buckets = 1024;
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); initial_buckets];
    buckets[0].push(start_internal);

    let mut current_bucket: usize = 0;

    // Helper: ensure buckets has enough capacity for index `idx`.
    fn ensure_bucket_capacity(buckets: &mut Vec<Vec<u32>>, idx: usize) {
        if idx >= buckets.len() {
            let new_len = (idx + 1).next_power_of_two();
            buckets.resize_with(new_len, Vec::new);
        }
    }

    while current_bucket < buckets.len() {
        // Skip empty buckets
        if buckets[current_bucket].is_empty() {
            current_bucket += 1;
            continue;
        }

        // Light phase: relax light edges (weight < delta) until bucket is stable
        loop {
            let bucket_nodes: Vec<u32> = std::mem::take(&mut buckets[current_bucket]);
            if bucket_nodes.is_empty() {
                break;
            }

            // Parallel relaxation of light edges
            let relaxations: Vec<(u32, f64, u32)> = bucket_nodes
                .par_iter()
                .flat_map(|&u| {
                    let dist_u = distances[u as usize];
                    let mut local = Vec::new();
                    for (v, w) in wcsr.weighted_neighbors(u) {
                        if w < delta {
                            let new_dist = dist_u + w;
                            local.push((v, new_dist, u));
                        }
                    }
                    local
                })
                .collect();

            for (v, new_dist, parent) in relaxations {
                if new_dist < distances[v as usize] {
                    distances[v as usize] = new_dist;
                    predecessors[v as usize] = parent;
                    let bucket_idx = (new_dist / delta) as usize;
                    ensure_bucket_capacity(&mut buckets, bucket_idx);
                    buckets[bucket_idx].push(v);
                }
            }
        }

        // Heavy phase: relax heavy edges (weight >= delta) once
        // Re-collect nodes that were in this bucket (by scanning distances)
        let heavy_nodes: Vec<u32> = (0..n as u32)
            .into_par_iter()
            .filter(|&u| {
                let d = distances[u as usize];
                !d.is_infinite() && (d / delta) as usize == current_bucket
            })
            .collect();

        let heavy_relaxations: Vec<(u32, f64, u32)> = heavy_nodes
            .par_iter()
            .flat_map(|&u| {
                let dist_u = distances[u as usize];
                let mut local = Vec::new();
                for (v, w) in wcsr.weighted_neighbors(u) {
                    if w >= delta {
                        let new_dist = dist_u + w;
                        local.push((v, new_dist, u));
                    }
                }
                local
            })
            .collect();

        for (v, new_dist, parent) in heavy_relaxations {
            if new_dist < distances[v as usize] {
                distances[v as usize] = new_dist;
                predecessors[v as usize] = parent;
                let bucket_idx = (new_dist / delta) as usize;
                ensure_bucket_capacity(&mut buckets, bucket_idx);
                buckets[bucket_idx].push(v);
            }
        }

        current_bucket += 1;
    }

    let nodes_reached = distances.iter().filter(|d| d.is_finite()).count();
    Some(SsspResult {
        distances,
        predecessors,
        nodes_reached,
        algorithm: SsspAlgorithm::DeltaStepping,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ConcurrentGraph;
    use crate::weighted_csr::WeightedCsrGraph;
    use bikodb_core::types::TypeId;
    use bikodb_core::value::Value;

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Crea grafo ponderado: A --2.0--> B --3.0--> C --1.0--> D
    fn weighted_chain() -> (ConcurrentGraph, Vec<NodeId>) {
        let g = ConcurrentGraph::new();
        let nodes: Vec<NodeId> = (0..4).map(|_| g.insert_node(TypeId(1))).collect();
        let weights = [2.0, 3.0, 1.0];
        for i in 0..3 {
            g.insert_edge_with_props(
                nodes[i],
                nodes[i + 1],
                TypeId(10),
                vec![(0, Value::Float(weights[i]))],
            )
            .unwrap();
        }
        (g, nodes)
    }

    /// Grafo diamante ponderado:
    ///   A --1.0--> B --1.0--> D
    ///   A --5.0--> C --0.5--> D
    /// Shortest: A→B→D = 2.0 (not A→C→D = 5.5)
    fn weighted_diamond() -> (ConcurrentGraph, Vec<NodeId>) {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));
        let d = g.insert_node(TypeId(1));

        g.insert_edge_with_props(a, b, TypeId(10), vec![(0, Value::Float(1.0))])
            .unwrap();
        g.insert_edge_with_props(b, d, TypeId(10), vec![(0, Value::Float(1.0))])
            .unwrap();
        g.insert_edge_with_props(a, c, TypeId(10), vec![(0, Value::Float(5.0))])
            .unwrap();
        g.insert_edge_with_props(c, d, TypeId(10), vec![(0, Value::Float(0.5))])
            .unwrap();

        (g, vec![a, b, c, d])
    }

    /// Grafo con peso negativo (sin ciclo negativo).
    fn negative_weight_graph() -> (ConcurrentGraph, Vec<NodeId>) {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));

        // A --3.0--> B, A --5.0--> C, B ---2.0--> C
        // Shortest A→C: A→B→C = 3.0 + (-2.0) = 1.0 (not A→C direct = 5.0)
        g.insert_edge_with_props(a, b, TypeId(10), vec![(0, Value::Float(3.0))])
            .unwrap();
        g.insert_edge_with_props(a, c, TypeId(10), vec![(0, Value::Float(5.0))])
            .unwrap();
        g.insert_edge_with_props(b, c, TypeId(10), vec![(0, Value::Float(-2.0))])
            .unwrap();

        (g, vec![a, b, c])
    }

    /// Grafo unweighted (todos pesos 1.0).
    fn unweighted_chain(n: usize) -> (ConcurrentGraph, Vec<NodeId>) {
        let g = ConcurrentGraph::new();
        let nodes: Vec<NodeId> = (0..n).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..n - 1 {
            g.insert_edge(nodes[i], nodes[i + 1], TypeId(10)).unwrap();
        }
        (g, nodes)
    }

    fn to_wcsr(g: &ConcurrentGraph) -> WeightedCsrGraph {
        WeightedCsrGraph::from_concurrent(g, bikodb_core::record::Direction::Out, 0, 1.0)
    }

    // ── Dijkstra tests ──────────────────────────────────────────────────

    #[test]
    fn test_dijkstra_weighted_chain() {
        let (g, nodes) = weighted_chain();
        let wcsr = to_wcsr(&g);

        let result = dijkstra(&wcsr, nodes[0]).unwrap();
        assert_eq!(result.algorithm, SsspAlgorithm::Dijkstra);
        assert_eq!(result.nodes_reached, 4);

        // A→B=2.0, A→C=5.0, A→D=6.0
        let d_b = result.distance_to(&wcsr, nodes[1]).unwrap();
        let d_c = result.distance_to(&wcsr, nodes[2]).unwrap();
        let d_d = result.distance_to(&wcsr, nodes[3]).unwrap();
        assert!((d_b - 2.0).abs() < f64::EPSILON);
        assert!((d_c - 5.0).abs() < f64::EPSILON);
        assert!((d_d - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dijkstra_diamond_shortest_path() {
        let (g, nodes) = weighted_diamond();
        let wcsr = to_wcsr(&g);

        let result = dijkstra(&wcsr, nodes[0]).unwrap();

        // Shortest to D: A→B→D = 2.0 (not A→C→D = 5.5)
        let dist_d = result.distance_to(&wcsr, nodes[3]).unwrap();
        assert!((dist_d - 2.0).abs() < f64::EPSILON);

        let path = result.path_to(&wcsr, nodes[3]).unwrap();
        assert_eq!(path, vec![nodes[0], nodes[1], nodes[3]]);
    }

    #[test]
    fn test_dijkstra_single_node() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let wcsr = to_wcsr(&g);

        let result = dijkstra(&wcsr, a).unwrap();
        assert_eq!(result.nodes_reached, 1);
        let d = result.distance_to(&wcsr, a).unwrap();
        assert!((d - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dijkstra_unreachable() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let wcsr = to_wcsr(&g);

        let result = dijkstra(&wcsr, a).unwrap();
        assert!(result.distance_to(&wcsr, b).is_none());
    }

    #[test]
    fn test_dijkstra_invalid_start() {
        let g = ConcurrentGraph::new();
        g.insert_node(TypeId(1));
        let wcsr = to_wcsr(&g);
        assert!(dijkstra(&wcsr, NodeId(9999)).is_none());
    }

    // ── Bellman-Ford tests ──────────────────────────────────────────────

    #[test]
    fn test_bellman_ford_negative_weights() {
        let (g, nodes) = negative_weight_graph();
        let wcsr = to_wcsr(&g);

        let result = bellman_ford_checked(&wcsr, nodes[0]).unwrap();
        assert_eq!(result.algorithm, SsspAlgorithm::BellmanFord);

        // A→C via B: 3.0 + (-2.0) = 1.0
        let d_c = result.distance_to(&wcsr, nodes[2]).unwrap();
        assert!((d_c - 1.0).abs() < f64::EPSILON);

        let path = result.path_to(&wcsr, nodes[2]).unwrap();
        assert_eq!(path, vec![nodes[0], nodes[1], nodes[2]]);
    }

    #[test]
    fn test_bellman_ford_negative_cycle_detection() {
        // A --1.0--> B ---2.0--> C --(-5.0)--> A  (cycle weight = 1-2-5 = -6)
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));

        g.insert_edge_with_props(a, b, TypeId(10), vec![(0, Value::Float(1.0))])
            .unwrap();
        g.insert_edge_with_props(b, c, TypeId(10), vec![(0, Value::Float(2.0))])
            .unwrap();
        g.insert_edge_with_props(c, a, TypeId(10), vec![(0, Value::Float(-5.0))])
            .unwrap();

        let wcsr = to_wcsr(&g);
        let result = bellman_ford_checked(&wcsr, a);
        assert!(result.is_err());
    }

    #[test]
    fn test_bellman_ford_positive_weights() {
        // Should work the same as Dijkstra for positive weights
        let (g, nodes) = weighted_diamond();
        let wcsr = to_wcsr(&g);

        let result = bellman_ford(&wcsr, nodes[0]).unwrap();
        let dist_d = result.distance_to(&wcsr, nodes[3]).unwrap();
        assert!((dist_d - 2.0).abs() < f64::EPSILON);
    }

    // ── BFS SSSP tests ──────────────────────────────────────────────────

    #[test]
    fn test_bfs_sssp_unit_weights() {
        let (g, nodes) = unweighted_chain(6);
        let wcsr = to_wcsr(&g);

        let result = bfs_sssp(&wcsr, nodes[0]).unwrap();
        assert_eq!(result.algorithm, SsspAlgorithm::Bfs);
        assert_eq!(result.nodes_reached, 6);

        for i in 0..6 {
            let d = result.distance_to(&wcsr, nodes[i]).unwrap();
            assert!((d - i as f64).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_bfs_sssp_path_reconstruction() {
        let (g, nodes) = unweighted_chain(5);
        let wcsr = to_wcsr(&g);

        let result = bfs_sssp(&wcsr, nodes[0]).unwrap();
        let path = result.path_to(&wcsr, nodes[4]).unwrap();
        assert_eq!(path, nodes);
    }

    // ── Δ-Stepping tests ────────────────────────────────────────────────

    #[test]
    fn test_delta_stepping_weighted_chain() {
        let (g, nodes) = weighted_chain();
        let wcsr = to_wcsr(&g);

        let result = delta_stepping(&wcsr, nodes[0], Some(1.0)).unwrap();
        assert_eq!(result.algorithm, SsspAlgorithm::DeltaStepping);

        let d_d = result.distance_to(&wcsr, nodes[3]).unwrap();
        assert!((d_d - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_delta_stepping_diamond() {
        let (g, nodes) = weighted_diamond();
        let wcsr = to_wcsr(&g);

        let result = delta_stepping(&wcsr, nodes[0], Some(1.0)).unwrap();
        let dist_d = result.distance_to(&wcsr, nodes[3]).unwrap();
        assert!((dist_d - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_delta_stepping_auto_delta() {
        let (g, nodes) = weighted_chain();
        let wcsr = to_wcsr(&g);

        let result = delta_stepping(&wcsr, nodes[0], None).unwrap();
        let d_d = result.distance_to(&wcsr, nodes[3]).unwrap();
        assert!((d_d - 6.0).abs() < f64::EPSILON);
    }

    // ── Adaptive selector tests ─────────────────────────────────────────

    #[test]
    fn test_sssp_selects_bfs_for_unit_weights() {
        let (g, nodes) = unweighted_chain(10);
        let wcsr = to_wcsr(&g);

        let result = sssp(&wcsr, nodes[0]).unwrap();
        assert_eq!(result.algorithm, SsspAlgorithm::Bfs);
    }

    #[test]
    fn test_sssp_selects_dijkstra_for_small_weighted() {
        let (g, nodes) = weighted_chain();
        let wcsr = to_wcsr(&g);

        let result = sssp(&wcsr, nodes[0]).unwrap();
        // < 50K nodos, weighted, no negative → Dijkstra
        assert_eq!(result.algorithm, SsspAlgorithm::Dijkstra);
    }

    #[test]
    fn test_sssp_selects_bellman_ford_for_negative() {
        let (g, nodes) = negative_weight_graph();
        let wcsr = to_wcsr(&g);

        let result = sssp(&wcsr, nodes[0]).unwrap();
        assert_eq!(result.algorithm, SsspAlgorithm::BellmanFord);
    }

    #[test]
    fn test_sssp_with_forced_algorithm() {
        let (g, nodes) = weighted_chain();
        let wcsr = to_wcsr(&g);

        // Force BellmanFord even though Dijkstra would be adaptive choice
        let result =
            sssp_with_algorithm(&wcsr, nodes[0], SsspAlgorithm::BellmanFord).unwrap();
        assert_eq!(result.algorithm, SsspAlgorithm::BellmanFord);

        let d_d = result.distance_to(&wcsr, nodes[3]).unwrap();
        assert!((d_d - 6.0).abs() < f64::EPSILON);
    }

    // ── Consistency: all algorithms agree ───────────────────────────────

    #[test]
    fn test_all_algorithms_agree_on_distances() {
        let (g, nodes) = weighted_diamond();
        let wcsr = to_wcsr(&g);

        let dij = dijkstra(&wcsr, nodes[0]).unwrap();
        let bf = bellman_ford(&wcsr, nodes[0]).unwrap();
        let ds = delta_stepping(&wcsr, nodes[0], Some(0.5)).unwrap();

        for node in &nodes {
            let d_dij = dij.distance_to(&wcsr, *node);
            let d_bf = bf.distance_to(&wcsr, *node);
            let d_ds = ds.distance_to(&wcsr, *node);

            match (d_dij, d_bf, d_ds) {
                (Some(a), Some(b), Some(c)) => {
                    assert!(
                        (a - b).abs() < 1e-9,
                        "Dijkstra vs BF disagree on {:?}: {} vs {}",
                        node,
                        a,
                        b
                    );
                    assert!(
                        (a - c).abs() < 1e-9,
                        "Dijkstra vs ΔStep disagree on {:?}: {} vs {}",
                        node,
                        a,
                        c
                    );
                }
                (None, None, None) => {}
                _ => panic!("Algorithms disagree on reachability of {:?}", node),
            }
        }
    }

    // ── Path reconstruction ─────────────────────────────────────────────

    #[test]
    fn test_dijkstra_path_weighted() {
        // Graph with multiple paths, different weights
        //   A --1--> B --1--> D
        //   A --10-> C --0.1> D
        // Shortest A→D: A→B→D = 2.0
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));
        let d = g.insert_node(TypeId(1));

        g.insert_edge_with_props(a, b, TypeId(10), vec![(0, Value::Float(1.0))])
            .unwrap();
        g.insert_edge_with_props(b, d, TypeId(10), vec![(0, Value::Float(1.0))])
            .unwrap();
        g.insert_edge_with_props(a, c, TypeId(10), vec![(0, Value::Float(10.0))])
            .unwrap();
        g.insert_edge_with_props(c, d, TypeId(10), vec![(0, Value::Float(0.1))])
            .unwrap();

        let wcsr = to_wcsr(&g);
        let result = dijkstra(&wcsr, a).unwrap();

        let path = result.path_to(&wcsr, d).unwrap();
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], a);
        assert_eq!(path[2], d);
    }

    // ── Star topology ───────────────────────────────────────────────────

    #[test]
    fn test_dijkstra_star() {
        let g = ConcurrentGraph::new();
        let center = g.insert_node(TypeId(1));
        let leaves: Vec<NodeId> = (0..100)
            .map(|i| {
                let l = g.insert_node(TypeId(1));
                let w = (i + 1) as f64 * 0.5;
                g.insert_edge_with_props(
                    center,
                    l,
                    TypeId(10),
                    vec![(0, Value::Float(w))],
                )
                .unwrap();
                l
            })
            .collect();

        let wcsr = to_wcsr(&g);
        let result = dijkstra(&wcsr, center).unwrap();
        assert_eq!(result.nodes_reached, 101);

        // Closest leaf has weight 0.5
        let min_dist: f64 = leaves
            .iter()
            .filter_map(|l| result.distance_to(&wcsr, *l))
            .fold(f64::INFINITY, f64::min);
        assert!((min_dist - 0.5).abs() < f64::EPSILON);
    }

    // ── Large-scale tests ───────────────────────────────────────────────

    #[test]
    fn test_dijkstra_large_chain_100k() {
        let n = 100_000;
        let g = ConcurrentGraph::with_capacity(n, n);
        let nodes: Vec<NodeId> = (0..n).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..n - 1 {
            g.insert_edge_with_props(
                nodes[i],
                nodes[i + 1],
                TypeId(10),
                vec![(0, Value::Float(1.0))],
            )
            .unwrap();
        }

        let wcsr = to_wcsr(&g);
        let result = dijkstra(&wcsr, nodes[0]).unwrap();
        assert_eq!(result.nodes_reached, n);

        let dist_last = result.distance_to(&wcsr, nodes[n - 1]).unwrap();
        assert!((dist_last - (n - 1) as f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_delta_stepping_large_random_graph() {
        // Random-ish graph: each node connects to next 3 with varying weights
        let n = 10_000;
        let g = ConcurrentGraph::with_capacity(n, n * 3);
        let nodes: Vec<NodeId> = (0..n).map(|_| g.insert_node(TypeId(1))).collect();

        for i in 0..n {
            for offset in 1..=3 {
                let j = (i + offset) % n;
                let w = ((i * 7 + offset * 13) % 100 + 1) as f64 / 10.0;
                g.insert_edge_with_props(
                    nodes[i],
                    nodes[j],
                    TypeId(10),
                    vec![(0, Value::Float(w))],
                )
                .unwrap();
            }
        }

        let wcsr = to_wcsr(&g);

        // Run both Dijkstra and Delta-Stepping and verify consistency
        let dij_result = dijkstra(&wcsr, nodes[0]).unwrap();
        let ds_result = delta_stepping(&wcsr, nodes[0], None).unwrap();

        assert_eq!(dij_result.nodes_reached, ds_result.nodes_reached);

        // Check a sample of distances match
        for &i in &[0, 1, 100, 500, 1000, 5000, n - 1] {
            let d_dij = dij_result.distance_to(&wcsr, nodes[i]);
            let d_ds = ds_result.distance_to(&wcsr, nodes[i]);
            match (d_dij, d_ds) {
                (Some(a), Some(b)) => {
                    assert!(
                        (a - b).abs() < 1e-6,
                        "Mismatch at node {i}: dijkstra={a}, delta_stepping={b}"
                    );
                }
                (None, None) => {}
                _ => panic!("Reachability mismatch at node {i}"),
            }
        }
    }

    #[test]
    fn test_adaptive_selects_delta_stepping_for_large_graph() {
        // Build graph with >50K nodes to trigger delta-stepping
        let n = 51_000;
        let g = ConcurrentGraph::with_capacity(n, n);
        let nodes: Vec<NodeId> = (0..n).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..n - 1 {
            g.insert_edge_with_props(
                nodes[i],
                nodes[i + 1],
                TypeId(10),
                vec![(0, Value::Float(1.5))],
            )
            .unwrap();
        }

        let wcsr = to_wcsr(&g);
        let result = sssp(&wcsr, nodes[0]).unwrap();
        assert_eq!(result.algorithm, SsspAlgorithm::DeltaStepping);
        assert_eq!(result.nodes_reached, n);
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn test_zero_weight_edges() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));

        g.insert_edge_with_props(a, b, TypeId(10), vec![(0, Value::Float(0.0))])
            .unwrap();
        g.insert_edge_with_props(b, c, TypeId(10), vec![(0, Value::Float(0.0))])
            .unwrap();

        let wcsr = to_wcsr(&g);
        let result = dijkstra(&wcsr, a).unwrap();
        let d_c = result.distance_to(&wcsr, c).unwrap();
        assert!((d_c - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_self_loop_weighted() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));

        g.insert_edge_with_props(a, a, TypeId(10), vec![(0, Value::Float(5.0))])
            .unwrap();
        g.insert_edge_with_props(a, b, TypeId(10), vec![(0, Value::Float(1.0))])
            .unwrap();

        let wcsr = to_wcsr(&g);
        let result = dijkstra(&wcsr, a).unwrap();
        // Self-loop doesn't affect shortest distance to self (0.0)
        let d_a = result.distance_to(&wcsr, a).unwrap();
        assert!((d_a - 0.0).abs() < f64::EPSILON);
        let d_b = result.distance_to(&wcsr, b).unwrap();
        assert!((d_b - 1.0).abs() < f64::EPSILON);
    }
}
