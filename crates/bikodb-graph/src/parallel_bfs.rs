// =============================================================================
// bikodb-graph::parallel_bfs — BFS paralelo level-synchronous sobre CSR
// =============================================================================
// BFS por niveles con paralelización de la frontera mediante rayon.
//
// ## Algoritmo Level-Synchronous BFS
//
//   frontier = {start}
//   while frontier is not empty:
//       next_frontier = frontier.par_iter()
//           .flat_map(|node| csr.neighbors(node))
//           .filter(|n| visited.set_and_check(n))  // atómico
//           .collect()
//       frontier = next_frontier
//
// Cada nivel se paraleliza completamente: los nodos de la frontera actual
// se expanden en paralelo, y los nuevos descubiertos forman la próxima frontera.
//
// ## Complejidad
// - Tiempo: O(V + E) / P  donde P = número de threads
// - Espacio: O(V) para bitset + O(frontera_max) para buffers
//
// ## Inspiración
// - Ligra: direction-optimizing BFS (push/pull switching)
// - PBFS (MIT): level-synchronous parallelism
// - GAP benchmark suite: referencia BFS paralelo
// =============================================================================

use crate::bitset::{AtomicBitSet, BitSet};
use crate::csr::CsrGraph;
use bikodb_core::types::NodeId;
use rayon::prelude::*;

/// Resultado de BFS paralelo.
#[derive(Debug)]
pub struct ParallelBfsResult {
    /// Distancia desde el nodo inicio para cada nodo (internal id).
    /// distances[i] = u32::MAX si no alcanzable.
    pub distances: Vec<u32>,
    /// Predecesor para cada nodo (internal id). u32::MAX si no tiene.
    pub predecessors: Vec<u32>,
    /// Número de nodos alcanzados.
    pub nodes_reached: usize,
    /// Número de niveles (profundidad máxima).
    pub max_depth: u32,
}

impl ParallelBfsResult {
    /// Reconstruye el path (en NodeIds externos) desde start hasta target.
    pub fn path_to(&self, csr: &CsrGraph, target: NodeId) -> Option<Vec<NodeId>> {
        let internal = csr.to_internal(target)?;
        if self.distances[internal as usize] == u32::MAX {
            return None;
        }

        let mut path = Vec::new();
        let mut current = internal;
        loop {
            path.push(csr.to_external(current));
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
    pub fn distance_to(&self, csr: &CsrGraph, target: NodeId) -> Option<u32> {
        let internal = csr.to_internal(target)?;
        let dist = self.distances[internal as usize];
        if dist == u32::MAX {
            None
        } else {
            Some(dist)
        }
    }
}

/// BFS paralelo level-synchronous sobre CSR.
///
/// Usa rayon para paralelizar la expansión de cada nivel de la frontera.
/// El bitset atómico evita locks: múltiples threads marcan nodos descubiertos
/// simultáneamente con fetch_or.
///
/// # Parámetros
/// - `csr`: grafo en formato CSR (inmutable, contiguo)
/// - `start`: nodo inicio (NodeId externo)
/// - `max_depth`: límite de profundidad (None = sin límite)
///
/// # Rendimiento
/// Para grafos con millones de nodos, el speedup es casi lineal en el
/// número de cores cuando la frontera es amplia (grafos con diámetro bajo).
pub fn parallel_bfs(
    csr: &CsrGraph,
    start: NodeId,
    max_depth: Option<u32>,
) -> Option<ParallelBfsResult> {
    let start_internal = csr.to_internal(start)?;
    let n = csr.num_nodes();

    let mut distances = vec![u32::MAX; n];
    let mut predecessors = vec![u32::MAX; n];
    let visited = AtomicBitSet::new(n);

    // Inicializar nodo start
    distances[start_internal as usize] = 0;
    visited.set_and_check(start_internal);

    let mut frontier: Vec<u32> = vec![start_internal];
    let mut depth: u32 = 0;
    let mut nodes_reached: usize = 1;

    while !frontier.is_empty() {
        if let Some(max) = max_depth {
            if depth >= max {
                break;
            }
        }

        let next_depth = depth + 1;

        // ── Expansión paralela de la frontera ──────────────────────────
        //
        // Para fronteras pequeñas (<1024 nodos), usamos single-thread
        // para evitar overhead de scheduling de rayon.
        // Para fronteras grandes, paralelizamos con rayon.
        let next_frontier: Vec<(u32, u32)> = if frontier.len() < 1024 {
            // Single-thread: evitar overhead de rayon para fronteras pequeñas
            let visited_ref = &visited;
            frontier
                .iter()
                .flat_map(|&node| {
                    csr.neighbors(node)
                        .iter()
                        .filter_map(move |&neighbor| {
                            if visited_ref.set_and_check(neighbor) {
                                Some((neighbor, node))
                            } else {
                                None
                            }
                        })
                })
                .collect()
        } else {
            // Paralelo: cada nodo de la frontera se expande concurrentemente
            let visited_ref = &visited;
            frontier
                .par_iter()
                .flat_map(|&node| {
                    csr.neighbors(node)
                        .par_iter()
                        .filter_map(move |&neighbor| {
                            if visited_ref.set_and_check(neighbor) {
                                Some((neighbor, node))
                            } else {
                                None
                            }
                        })
                })
                .collect()
        };

        // Registrar distancias y predecesores
        frontier = Vec::with_capacity(next_frontier.len());
        for (neighbor, parent) in next_frontier {
            distances[neighbor as usize] = next_depth;
            predecessors[neighbor as usize] = parent;
            frontier.push(neighbor);
            nodes_reached += 1;
        }

        if !frontier.is_empty() {
            depth = next_depth;
        }
    }

    Some(ParallelBfsResult {
        distances,
        predecessors,
        nodes_reached,
        max_depth: depth,
    })
}

// =============================================================================
// Direction-Optimizing BFS (Beamer et al., SC 2012)
// =============================================================================
// Alterna entre expansión push (top-down) y pull (bottom-up) según el tamaño
// de la frontera relativa a los edges sin visitar.
//
// Push: itera nodos de la frontera → expande vecinos (out_csr)
//       Bueno cuando la frontera es pequeña.
// Pull: itera nodos NO visitados → chequea si algún in-vecino está en frontera
//       Bueno cuando la frontera es enorme (evita explorar edges redundantes).
//
// Criterio de switch: push → pull cuando frontier_edges > remaining_edges / α
//                      pull → push cuando |frontier| < n / β
//
// En grafos power-law (redes sociales, web), esto da 2-10x speedup vs push-only.
// =============================================================================

/// Configuración para direction-optimizing BFS.
pub struct DoBfsConfig {
    /// Factor alpha para switch push→pull. Default: 15.
    /// Switch a pull cuando frontier_edges > remaining_edges / alpha.
    pub alpha: u32,
    /// Factor beta para switch pull→push. Default: 24.
    /// Switch a push cuando |frontier| < num_nodes / beta.
    pub beta: u32,
    /// Límite de profundidad (None = sin límite).
    pub max_depth: Option<u32>,
}

impl Default for DoBfsConfig {
    fn default() -> Self {
        Self {
            alpha: 15,
            beta: 24,
            max_depth: None,
        }
    }
}

/// BFS direction-optimizing sobre CSR (Beamer's algorithm).
///
/// Requiere ambos CSR (out e in) para las fases push y pull.
/// En grafos power-law grandes, da 2-10x speedup vs BFS push-only.
pub fn direction_optimizing_bfs(
    out_csr: &CsrGraph,
    in_csr: &CsrGraph,
    start: NodeId,
    config: &DoBfsConfig,
) -> Option<ParallelBfsResult> {
    let start_internal = out_csr.to_internal(start)?;
    let n = out_csr.num_nodes();
    if n == 0 {
        return Some(ParallelBfsResult {
            distances: vec![],
            predecessors: vec![],
            nodes_reached: 0,
            max_depth: 0,
        });
    }

    let mut distances = vec![u32::MAX; n];
    let mut predecessors = vec![u32::MAX; n];
    let visited = AtomicBitSet::new(n);

    distances[start_internal as usize] = 0;
    visited.set_and_check(start_internal);

    let mut frontier: Vec<u32> = vec![start_internal];
    let mut depth: u32 = 0;
    let mut nodes_reached: usize = 1;

    let total_edges = out_csr.num_edges() as u64;
    let mut visited_edges: u64 = 0;
    let mut use_pull = false;

    while !frontier.is_empty() {
        if let Some(max) = config.max_depth {
            if depth >= max {
                break;
            }
        }

        let next_depth = depth + 1;

        // Calcular frontier_edges (sum out-degrees de nodos en frontera)
        let frontier_edges: u64 = if frontier.len() < 1024 {
            frontier.iter().map(|&v| out_csr.degree(v) as u64).sum()
        } else {
            frontier.par_iter().map(|&v| out_csr.degree(v) as u64).sum()
        };
        let remaining_edges = total_edges.saturating_sub(visited_edges);

        // Decidir dirección
        if !use_pull
            && remaining_edges > 0
            && frontier_edges > remaining_edges / config.alpha as u64
        {
            use_pull = true;
        } else if use_pull && (frontier.len() as u64) < (n as u64) / config.beta as u64 {
            use_pull = false;
        }

        if use_pull {
            // ── Pull phase (bottom-up) ─────────────────────────────
            // Construir bitset de frontera fresco para O(1) membership check
            let frontier_set = AtomicBitSet::new(n);
            for &v in &frontier {
                frontier_set.set_and_check(v);
            }

            // Escanear nodos no visitados; para cada uno, buscar primer
            // in-vecino en la frontera.
            let visited_ref = &visited;
            let frontier_ref = &frontier_set;
            let discovered: Vec<(u32, u32)> = (0..n as u32)
                .into_par_iter()
                .filter_map(|v| {
                    if visited_ref.get(v) {
                        return None;
                    }
                    for &parent in in_csr.neighbors(v) {
                        if frontier_ref.get(parent) {
                            return Some((v, parent));
                        }
                    }
                    None
                })
                .collect();

            // Registrar descubiertos (filter duplicates via visited CAS)
            frontier = Vec::with_capacity(discovered.len());
            for (v, parent) in discovered {
                if visited.set_and_check(v) {
                    distances[v as usize] = next_depth;
                    predecessors[v as usize] = parent;
                    frontier.push(v);
                }
            }
        } else {
            // ── Push phase (top-down) ──────────────────────────────
            let visited_ref = &visited;
            let next_frontier: Vec<(u32, u32)> = if frontier.len() < 1024 {
                frontier
                    .iter()
                    .flat_map(|&node| {
                        out_csr.neighbors(node).iter().filter_map(move |&neighbor| {
                            if visited_ref.set_and_check(neighbor) {
                                Some((neighbor, node))
                            } else {
                                None
                            }
                        })
                    })
                    .collect()
            } else {
                frontier
                    .par_iter()
                    .flat_map(|&node| {
                        out_csr
                            .neighbors(node)
                            .par_iter()
                            .filter_map(move |&neighbor| {
                                if visited_ref.set_and_check(neighbor) {
                                    Some((neighbor, node))
                                } else {
                                    None
                                }
                            })
                    })
                    .collect()
            };

            frontier = Vec::with_capacity(next_frontier.len());
            for (neighbor, parent) in next_frontier {
                distances[neighbor as usize] = next_depth;
                predecessors[neighbor as usize] = parent;
                frontier.push(neighbor);
            }
        }

        // Actualizar tracking de edges visitados
        let new_edges: u64 = frontier
            .iter()
            .map(|&v| out_csr.degree(v) as u64)
            .sum();
        nodes_reached += frontier.len();
        visited_edges += new_edges;

        if !frontier.is_empty() {
            depth = next_depth;
        }
    }

    Some(ParallelBfsResult {
        distances,
        predecessors,
        nodes_reached,
        max_depth: depth,
    })
}

/// BFS secuencial optimizado sobre CSR con BitSet.
///
/// Para cuando el grafo es pequeño o se necesita determinismo puro.
/// Usa bitset en vez de HashSet para máxima eficiencia de caché.
pub fn sequential_bfs(
    csr: &CsrGraph,
    start: NodeId,
    max_depth: Option<u32>,
) -> Option<ParallelBfsResult> {
    let start_internal = csr.to_internal(start)?;
    let n = csr.num_nodes();

    let mut distances = vec![u32::MAX; n];
    let mut predecessors = vec![u32::MAX; n];
    let mut visited = BitSet::new(n);

    distances[start_internal as usize] = 0;
    visited.set_and_check(start_internal);

    // Double-buffer: frontier actual y next se intercambian
    let mut frontier = Vec::new();
    let mut next_frontier = Vec::new();
    frontier.push(start_internal);

    let mut depth: u32 = 0;
    let mut nodes_reached: usize = 1;

    while !frontier.is_empty() {
        if let Some(max) = max_depth {
            if depth >= max {
                break;
            }
        }

        let next_depth = depth + 1;

        for &node in &frontier {
            for &neighbor in csr.neighbors(node) {
                if visited.set_and_check(neighbor) {
                    distances[neighbor as usize] = next_depth;
                    predecessors[neighbor as usize] = node;
                    next_frontier.push(neighbor);
                    nodes_reached += 1;
                }
            }
        }

        std::mem::swap(&mut frontier, &mut next_frontier);
        next_frontier.clear();
        depth = next_depth;
    }

    Some(ParallelBfsResult {
        distances,
        predecessors,
        nodes_reached,
        max_depth: depth,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ConcurrentGraph;
    use bikodb_core::record::Direction;
    use bikodb_core::types::TypeId;

    /// Helper: crea cadena A→B→C→D→E
    fn chain_graph(n: usize) -> (ConcurrentGraph, Vec<NodeId>) {
        let g = ConcurrentGraph::new();
        let nodes: Vec<NodeId> = (0..n).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..n - 1 {
            g.insert_edge(nodes[i], nodes[i + 1], TypeId(10)).unwrap();
        }
        (g, nodes)
    }

    /// Helper: estrella con centro y n hojas
    fn star_graph(n: usize) -> (ConcurrentGraph, NodeId, Vec<NodeId>) {
        let g = ConcurrentGraph::new();
        let center = g.insert_node(TypeId(1));
        let leaves: Vec<NodeId> = (0..n)
            .map(|_| {
                let leaf = g.insert_node(TypeId(1));
                g.insert_edge(center, leaf, TypeId(10)).unwrap();
                leaf
            })
            .collect();
        (g, center, leaves)
    }

    #[test]
    fn test_parallel_bfs_chain() {
        let (g, nodes) = chain_graph(5);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let result = parallel_bfs(&csr, nodes[0], None).unwrap();
        assert_eq!(result.nodes_reached, 5);
        assert_eq!(result.distance_to(&csr, nodes[4]), Some(4));
        assert_eq!(result.distance_to(&csr, nodes[0]), Some(0));

        // Verificar path
        let path = result.path_to(&csr, nodes[4]).unwrap();
        assert_eq!(path, nodes);
    }

    #[test]
    fn test_sequential_bfs_chain() {
        let (g, nodes) = chain_graph(5);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let result = sequential_bfs(&csr, nodes[0], None).unwrap();
        assert_eq!(result.nodes_reached, 5);
        assert_eq!(result.distance_to(&csr, nodes[4]), Some(4));
    }

    #[test]
    fn test_parallel_bfs_star() {
        let (g, center, leaves) = star_graph(100);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let result = parallel_bfs(&csr, center, None).unwrap();
        assert_eq!(result.nodes_reached, 101); // center + 100 leaves
        for leaf in &leaves {
            assert_eq!(result.distance_to(&csr, *leaf), Some(1));
        }
    }

    #[test]
    fn test_bfs_max_depth() {
        let (g, nodes) = chain_graph(10);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let result = parallel_bfs(&csr, nodes[0], Some(3)).unwrap();
        // Con max_depth=3, solo alcanza nodos 0,1,2,3
        assert_eq!(result.distance_to(&csr, nodes[3]), Some(3));
        assert_eq!(result.distance_to(&csr, nodes[4]), None);
    }

    #[test]
    fn test_bfs_unreachable() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1)); // aislado
        g.insert_node(TypeId(1));

        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let result = parallel_bfs(&csr, a, None).unwrap();

        assert_eq!(result.nodes_reached, 1); // solo a
        assert_eq!(result.distance_to(&csr, b), None);
    }

    #[test]
    fn test_parallel_vs_sequential_consistency() {
        // Grafo tipo grid 20x20
        let g = ConcurrentGraph::new();
        let width = 20;
        let mut grid = Vec::new();
        for _ in 0..width * width {
            grid.push(g.insert_node(TypeId(1)));
        }
        for row in 0..width {
            for col in 0..width {
                let idx = row * width + col;
                if col + 1 < width {
                    g.insert_edge(grid[idx], grid[idx + 1], TypeId(10)).unwrap();
                }
                if row + 1 < width {
                    g.insert_edge(grid[idx], grid[idx + width], TypeId(10)).unwrap();
                }
            }
        }

        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let par = parallel_bfs(&csr, grid[0], None).unwrap();
        let seq = sequential_bfs(&csr, grid[0], None).unwrap();

        // Ambos deben alcanzar los mismos nodos con las mismas distancias
        assert_eq!(par.nodes_reached, seq.nodes_reached);
        assert_eq!(par.nodes_reached, width * width);

        for i in 0..csr.num_nodes() {
            assert_eq!(par.distances[i], seq.distances[i],
                "Mismatch en distancia para nodo interno {i}");
        }
    }

    #[test]
    fn test_bfs_invalid_start() {
        let g = ConcurrentGraph::new();
        g.insert_node(TypeId(1));
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        // NodeId que no existe
        let result = parallel_bfs(&csr, NodeId(9999), None);
        assert!(result.is_none());
    }

    #[test]
    fn test_bfs_single_node() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let result = parallel_bfs(&csr, a, None).unwrap();
        assert_eq!(result.nodes_reached, 1);
        assert_eq!(result.max_depth, 0);
    }

    #[test]
    fn test_bfs_diamond() {
        //    A
        //   / \
        //  B   C
        //   \ /
        //    D
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));
        let d = g.insert_node(TypeId(1));
        g.insert_edge(a, b, TypeId(10)).unwrap();
        g.insert_edge(a, c, TypeId(10)).unwrap();
        g.insert_edge(b, d, TypeId(10)).unwrap();
        g.insert_edge(c, d, TypeId(10)).unwrap();

        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let result = parallel_bfs(&csr, a, None).unwrap();

        assert_eq!(result.nodes_reached, 4);
        assert_eq!(result.distance_to(&csr, d), Some(2));
        assert_eq!(result.distance_to(&csr, b), Some(1));
    }

    /// Test con 1M+ nodos: verifica que BFS paralelo escala.
    /// Grafo: cadena de clusters conectados (estilo social network).
    #[test]
    fn test_parallel_bfs_one_million_nodes() {
        let num_nodes = 1_000_000;
        let avg_edges = 6;
        let g = ConcurrentGraph::with_capacity(num_nodes, num_nodes * avg_edges);

        // Insertar nodos
        let nodes: Vec<NodeId> = (0..num_nodes)
            .map(|_| g.insert_node(TypeId(1)))
            .collect();

        // Crear estructura tipo small-world:
        // - Cadena principal (diámetro bajo)
        // - Conexiones locales (clusters)
        for i in 0..num_nodes - 1 {
            // Cadena principal
            g.insert_edge(nodes[i], nodes[i + 1], TypeId(10)).unwrap();
            // Conexiones locales (window de ~5)
            for offset in 2..=5 {
                if i + offset < num_nodes {
                    g.insert_edge(nodes[i], nodes[i + offset], TypeId(10)).unwrap();
                }
            }
        }

        // Construir CSR y ejecutar BFS paralelo
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        assert_eq!(csr.num_nodes(), num_nodes);

        let result = parallel_bfs(&csr, nodes[0], None).unwrap();

        // Todos los nodos deben ser alcanzables
        assert_eq!(result.nodes_reached, num_nodes);

        // El último nodo debe estar a distancia razonable (no lineal gracias a shortcuts)
        let dist_last = result.distance_to(&csr, nodes[num_nodes - 1]).unwrap();
        // Con shortcuts de 5, la distancia máxima es ~N/5
        assert!(dist_last <= (num_nodes as u32) / 4,
            "Distancia {dist_last} demasiado alta para grafo small-world");

        // Verificar consistencia par vs seq
        let seq_result = sequential_bfs(&csr, nodes[0], None).unwrap();
        assert_eq!(result.nodes_reached, seq_result.nodes_reached);

        // Distancias deben coincidir
        for i in 0..csr.num_nodes() {
            assert_eq!(result.distances[i], seq_result.distances[i],
                "Distancia inconsistente en nodo interno {i}");
        }
    }

    // --------------------------------------------------------
    // Tests para Direction-Optimizing BFS
    // --------------------------------------------------------

    #[test]
    fn test_do_bfs_chain() {
        let (g, nodes) = chain_graph(5);
        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let in_csr = CsrGraph::from_concurrent(&g, Direction::In);
        let config = DoBfsConfig::default();

        let result = direction_optimizing_bfs(&out_csr, &in_csr, nodes[0], &config).unwrap();
        assert_eq!(result.nodes_reached, 5);
        assert_eq!(result.distance_to(&out_csr, nodes[4]), Some(4));
        assert_eq!(result.distance_to(&out_csr, nodes[0]), Some(0));

        let path = result.path_to(&out_csr, nodes[4]).unwrap();
        assert_eq!(path, nodes);
    }

    #[test]
    fn test_do_bfs_star() {
        let (g, center, leaves) = star_graph(100);
        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let in_csr = CsrGraph::from_concurrent(&g, Direction::In);
        let config = DoBfsConfig::default();

        let result = direction_optimizing_bfs(&out_csr, &in_csr, center, &config).unwrap();
        assert_eq!(result.nodes_reached, 101);
        for leaf in &leaves {
            assert_eq!(result.distance_to(&out_csr, *leaf), Some(1));
        }
    }

    #[test]
    fn test_do_bfs_max_depth() {
        let (g, nodes) = chain_graph(10);
        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let in_csr = CsrGraph::from_concurrent(&g, Direction::In);
        let config = DoBfsConfig {
            max_depth: Some(3),
            ..DoBfsConfig::default()
        };

        let result = direction_optimizing_bfs(&out_csr, &in_csr, nodes[0], &config).unwrap();
        assert_eq!(result.distance_to(&out_csr, nodes[3]), Some(3));
        assert_eq!(result.distance_to(&out_csr, nodes[4]), None);
    }

    #[test]
    fn test_do_bfs_unreachable() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let in_csr = CsrGraph::from_concurrent(&g, Direction::In);
        let config = DoBfsConfig::default();

        let result = direction_optimizing_bfs(&out_csr, &in_csr, a, &config).unwrap();
        assert_eq!(result.nodes_reached, 1);
        assert_eq!(result.distance_to(&out_csr, b), None);
    }

    #[test]
    fn test_do_bfs_invalid_start() {
        let g = ConcurrentGraph::new();
        g.insert_node(TypeId(1));
        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let in_csr = CsrGraph::from_concurrent(&g, Direction::In);
        let config = DoBfsConfig::default();

        let result = direction_optimizing_bfs(&out_csr, &in_csr, NodeId(9999), &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_do_bfs_vs_push_only_consistency() {
        // Grid 20x20: DO-BFS debe dar mismas distancias que push-only BFS
        let g = ConcurrentGraph::new();
        let width = 20;
        let mut grid = Vec::new();
        for _ in 0..width * width {
            grid.push(g.insert_node(TypeId(1)));
        }
        for row in 0..width {
            for col in 0..width {
                let idx = row * width + col;
                if col + 1 < width {
                    g.insert_edge(grid[idx], grid[idx + 1], TypeId(10)).unwrap();
                }
                if row + 1 < width {
                    g.insert_edge(grid[idx], grid[idx + width], TypeId(10)).unwrap();
                }
            }
        }

        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let in_csr = CsrGraph::from_concurrent(&g, Direction::In);
        let config = DoBfsConfig::default();

        let do_result = direction_optimizing_bfs(&out_csr, &in_csr, grid[0], &config).unwrap();
        let push_result = parallel_bfs(&out_csr, grid[0], None).unwrap();

        assert_eq!(do_result.nodes_reached, push_result.nodes_reached);
        assert_eq!(do_result.nodes_reached, width * width);
        for i in 0..out_csr.num_nodes() {
            assert_eq!(
                do_result.distances[i], push_result.distances[i],
                "Distancia mismatch en nodo interno {i}"
            );
        }
    }

    #[test]
    fn test_do_bfs_power_law_graph() {
        // Grafo power-law: pocos hubs con muchas conexiones, muchos nodos con pocas
        // Esto debería activar la fase pull en DO-BFS
        let g = ConcurrentGraph::new();
        let n = 5000;
        let nodes: Vec<NodeId> = (0..n).map(|_| g.insert_node(TypeId(1))).collect();

        // Hub central conectado a todos
        let hub = nodes[0];
        for i in 1..n {
            g.insert_edge(hub, nodes[i], TypeId(10)).unwrap();
            g.insert_edge(nodes[i], hub, TypeId(10)).unwrap();
        }
        // Algunos sub-hubs
        for h in [1, 2, 3, 4, 5] {
            for i in (h * 100)..((h + 1) * 100).min(n) {
                if i != h {
                    g.insert_edge(nodes[h], nodes[i], TypeId(10)).unwrap();
                }
            }
        }

        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let in_csr = CsrGraph::from_concurrent(&g, Direction::In);

        // Force low alpha to trigger pull phase on this graph
        let config = DoBfsConfig {
            alpha: 2,
            beta: 24,
            max_depth: None,
        };

        let do_result = direction_optimizing_bfs(&out_csr, &in_csr, hub, &config).unwrap();
        let push_result = parallel_bfs(&out_csr, hub, None).unwrap();

        assert_eq!(do_result.nodes_reached, push_result.nodes_reached);
        for i in 0..out_csr.num_nodes() {
            assert_eq!(
                do_result.distances[i], push_result.distances[i],
                "Distancia mismatch en nodo interno {i} (do={}, push={})",
                do_result.distances[i], push_result.distances[i]
            );
        }
    }

    #[test]
    fn test_do_bfs_single_node() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let in_csr = CsrGraph::from_concurrent(&g, Direction::In);
        let config = DoBfsConfig::default();

        let result = direction_optimizing_bfs(&out_csr, &in_csr, a, &config).unwrap();
        assert_eq!(result.nodes_reached, 1);
        assert_eq!(result.max_depth, 0);
    }
}
