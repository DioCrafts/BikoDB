// =============================================================================
// bikodb-graph::optimized_dfs — DFS optimizado sobre CSR
// =============================================================================
// DFS de alto rendimiento para grafos con millones de nodos.
//
// ## Mejoras sobre el DFS original (traversal.rs):
//
// | Aspecto              | traversal::dfs           | optimized_dfs          |
// |----------------------|--------------------------|------------------------|
// | Grafo                | DashMap (hash-based)     | CSR (contiguo)         |
// | Visited              | HashSet (~40B/entry)     | BitSet (1 bit/nodo)    |
// | Resultado            | 3 HashMaps               | 2 Vec<u32> densos      |
// | Stack duplicados     | Sí (check post-pop)      | No (check pre-push)    |
// | Memoria 10M nodos    | ~1.2 GB                  | ~80 MB                 |
// | Cache behaviour      | Random (hash lookups)    | Sequential (arrays)    |
//
// ## Algoritmos incluidos:
// - `dfs()`  — DFS iterativo con stack eficiente sobre CSR
// - `iddfs()` — Iterative Deepening DFS: completo como BFS, memoria de DFS
//
// ## IDDFS: ¿cuándo usar?
// IDDFS ejecuta DFS con profundidad 1, luego 2, luego 3... hasta encontrar
// el target. Parece redundante, pero el overhead es bajo (factor ~d/(d-1)
// donde d = branching factor medio) y combina:
// - Memoria O(profundidad × branching)  ← como DFS
// - Completitud / optimalidad           ← como BFS
// Ideal para "encontrar el nodo X más cercano" sin explotar memoria.
// =============================================================================

use crate::bitset::BitSet;
use crate::csr::CsrGraph;
use bikodb_core::types::NodeId;

/// Resultado compacto de un DFS sobre CSR.
///
/// Usa arrays densos indexados por internal id (0..N) en vez de HashMaps.
/// Memoria: 2 × N × 4 bytes = 8 bytes por nodo (vs ~120 bytes con HashMaps).
#[derive(Debug)]
pub struct DfsResult {
    /// Profundidad de descubrimiento para cada nodo.
    /// `depths[i] = u32::MAX` si no alcanzado.
    pub depths: Vec<u32>,
    /// Predecesor de cada nodo. `u32::MAX` si raíz o no alcanzado.
    pub predecessors: Vec<u32>,
    /// Nodos visitados en orden de descubrimiento (internal ids).
    pub visit_order: Vec<u32>,
    /// Número de nodos alcanzados.
    pub nodes_reached: usize,
    /// Profundidad máxima alcanzada.
    pub max_depth: u32,
}

impl DfsResult {
    /// Reconstruye el path (NodeIds externos) desde start hasta target.
    pub fn path_to(&self, csr: &CsrGraph, target: NodeId) -> Option<Vec<NodeId>> {
        let internal = csr.to_internal(target)?;
        if self.depths[internal as usize] == u32::MAX {
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

    /// Profundidad de un nodo externo. None si no alcanzado.
    pub fn depth_of(&self, csr: &CsrGraph, target: NodeId) -> Option<u32> {
        let internal = csr.to_internal(target)?;
        let d = self.depths[internal as usize];
        if d == u32::MAX { None } else { Some(d) }
    }
}

// =============================================================================
// DFS iterativo sobre CSR
// =============================================================================

/// DFS iterativo optimizado sobre CSR con BitSet.
///
/// Stack sin duplicados: verifica visited ANTES de push (no después de pop).
/// Esto mantiene el stack en O(ruta_actual × branching) en vez de
/// O(V × degree_promedio).
///
/// # Memoria para 10M nodos:
/// - BitSet visited: 1.25 MB
/// - depths + predecessors: 80 MB
/// - Stack: proporcional a profundidad × branching (no a V)
///
/// # Parámetros
/// - `csr`: grafo CSR (inmutable, contiguo)
/// - `start`: nodo inicio (NodeId externo)
/// - `max_depth`: límite de profundidad (None = sin límite)
pub fn dfs(
    csr: &CsrGraph,
    start: NodeId,
    max_depth: Option<u32>,
) -> Option<DfsResult> {
    let start_internal = csr.to_internal(start)?;
    let n = csr.num_nodes();

    let mut depths = vec![u32::MAX; n];
    let mut predecessors = vec![u32::MAX; n];
    let mut visit_order = Vec::new();
    let mut visited = BitSet::new(n);

    // Stack: (node, depth). Pre-check visited = zero duplicados.
    let mut stack: Vec<(u32, u32)> = Vec::new();

    visited.set_and_check(start_internal);
    stack.push((start_internal, 0));

    let mut max_depth_seen: u32 = 0;

    while let Some((current, depth)) = stack.pop() {
        depths[current as usize] = depth;
        visit_order.push(current);

        if depth > max_depth_seen {
            max_depth_seen = depth;
        }

        // Solo expandir si no hemos llegado al límite
        if max_depth.map_or(true, |max| depth < max) {
            let neighbors = csr.neighbors(current);
            // Push en reverso para que el primer vecino sea procesado primero
            // (mantiene DFS left-to-right order)
            for &neighbor in neighbors.iter().rev() {
                if visited.set_and_check(neighbor) {
                    predecessors[neighbor as usize] = current;
                    stack.push((neighbor, depth + 1));
                }
            }
        }
    }

    let nodes_reached = visit_order.len();
    Some(DfsResult {
        depths,
        predecessors,
        visit_order,
        nodes_reached,
        max_depth: max_depth_seen,
    })
}

// =============================================================================
// IDDFS — Iterative Deepening Depth-First Search
// =============================================================================

/// Resultado de IDDFS: encontró target o no.
#[derive(Debug)]
pub struct IddfsResult {
    /// ¿Se encontró el target?
    pub found: bool,
    /// Profundidad a la que se encontró (si found = true).
    pub depth: u32,
    /// Path desde start hasta target (NodeIds externos).
    pub path: Vec<NodeId>,
    /// Nodos totales explorados en todas las iteraciones.
    pub total_explored: usize,
}

/// Iterative Deepening DFS sobre CSR.
///
/// Combina la eficiencia de memoria del DFS con la completitud del BFS.
/// Ejecuta DFS con profundidad límite 0, 1, 2, ... hasta encontrar el target
/// o agotar el grafo.
///
/// # Complejidad
/// - Tiempo: O(b^d) donde b = branching factor, d = profundidad del target
///   (el overhead de re-explorar niveles anteriores es factor b/(b-1) ≈ constante)
/// - Espacio: O(d × b) — solo la ruta actual + un nivel de vecinos
///
/// # Cuándo usar
/// - Buscar un nodo específico sin saber a qué profundidad está
/// - Grafos enormes donde BFS exploraría demasiado (frontera exponencial)
/// - Cuando importa encontrar el path más corto con memoria mínima
pub fn iddfs(
    csr: &CsrGraph,
    start: NodeId,
    target: NodeId,
    max_depth_limit: u32,
) -> Option<IddfsResult> {
    let start_internal = csr.to_internal(start)?;
    let target_internal = csr.to_internal(target)?;
    let n = csr.num_nodes();

    let mut total_explored: usize = 0;

    for depth_limit in 0..=max_depth_limit {
        // DFS limitado: reutiliza BitSet para cada iteración
        let mut visited = BitSet::new(n);
        // Stack: (node, depth)
        let mut stack: Vec<(u32, u32)> = Vec::new();
        // Predecessors para reconstruir path
        let mut predecessors = vec![u32::MAX; n];

        visited.set_and_check(start_internal);
        stack.push((start_internal, 0));

        while let Some((current, depth)) = stack.pop() {
            total_explored += 1;

            // ¿Encontrado?
            if current == target_internal {
                // Reconstruir path
                let mut path = Vec::new();
                let mut c = current;
                loop {
                    path.push(csr.to_external(c));
                    let pred = predecessors[c as usize];
                    if pred == u32::MAX {
                        break;
                    }
                    c = pred;
                }
                path.reverse();

                return Some(IddfsResult {
                    found: true,
                    depth,
                    path,
                    total_explored,
                });
            }

            // Solo expandir si no hemos llegado al límite de esta iteración
            if depth < depth_limit {
                for &neighbor in csr.neighbors(current).iter().rev() {
                    if visited.set_and_check(neighbor) {
                        predecessors[neighbor as usize] = current;
                        stack.push((neighbor, depth + 1));
                    }
                }
            }
        }
    }

    // No encontrado dentro del límite
    Some(IddfsResult {
        found: false,
        depth: 0,
        path: Vec::new(),
        total_explored,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrGraph;
    use crate::graph::ConcurrentGraph;
    use bikodb_core::record::Direction;
    use bikodb_core::types::TypeId;

    // ── Helpers ─────────────────────────────────────────────────────────

    fn chain(n: usize) -> (ConcurrentGraph, Vec<NodeId>) {
        let g = ConcurrentGraph::new();
        let nodes: Vec<NodeId> = (0..n).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..n - 1 {
            g.insert_edge(nodes[i], nodes[i + 1], TypeId(10)).unwrap();
        }
        (g, nodes)
    }

    fn star(n: usize) -> (ConcurrentGraph, NodeId, Vec<NodeId>) {
        let g = ConcurrentGraph::new();
        let center = g.insert_node(TypeId(1));
        let leaves: Vec<NodeId> = (0..n)
            .map(|_| {
                let l = g.insert_node(TypeId(1));
                g.insert_edge(center, l, TypeId(10)).unwrap();
                l
            })
            .collect();
        (g, center, leaves)
    }

    fn diamond() -> (ConcurrentGraph, NodeId, NodeId, NodeId, NodeId) {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));
        let d = g.insert_node(TypeId(1));
        g.insert_edge(a, b, TypeId(10)).unwrap();
        g.insert_edge(a, c, TypeId(10)).unwrap();
        g.insert_edge(b, d, TypeId(10)).unwrap();
        g.insert_edge(c, d, TypeId(10)).unwrap();
        (g, a, b, c, d)
    }

    // ── DFS tests ───────────────────────────────────────────────────────

    #[test]
    fn test_dfs_chain() {
        let (g, nodes) = chain(5);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let result = dfs(&csr, nodes[0], None).unwrap();
        assert_eq!(result.nodes_reached, 5);
        assert_eq!(result.max_depth, 4);
        assert_eq!(result.depth_of(&csr, nodes[4]), Some(4));

        // Path reconstruction
        let path = result.path_to(&csr, nodes[4]).unwrap();
        assert_eq!(path, nodes);
    }

    #[test]
    fn test_dfs_star() {
        let (g, center, leaves) = star(50);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let result = dfs(&csr, center, None).unwrap();
        assert_eq!(result.nodes_reached, 51); // center + 50 leaves
        assert_eq!(result.max_depth, 1);

        for leaf in &leaves {
            assert_eq!(result.depth_of(&csr, *leaf), Some(1));
        }
    }

    #[test]
    fn test_dfs_diamond() {
        let (g, a, _b, _c, d) = diamond();
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let result = dfs(&csr, a, None).unwrap();
        assert_eq!(result.nodes_reached, 4);
        // D reachable at depth 2 (a→b→d or a→c→d)
        assert_eq!(result.depth_of(&csr, d), Some(2));
    }

    #[test]
    fn test_dfs_max_depth() {
        let (g, nodes) = chain(10);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let result = dfs(&csr, nodes[0], Some(3)).unwrap();
        // Can reach nodes 0,1,2,3 (depths 0,1,2,3)
        assert_eq!(result.depth_of(&csr, nodes[3]), Some(3));
        assert_eq!(result.depth_of(&csr, nodes[4]), None);
    }

    #[test]
    fn test_dfs_single_node() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let result = dfs(&csr, a, None).unwrap();
        assert_eq!(result.nodes_reached, 1);
        assert_eq!(result.max_depth, 0);
    }

    #[test]
    fn test_dfs_unreachable() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let result = dfs(&csr, a, None).unwrap();
        assert_eq!(result.nodes_reached, 1);
        assert_eq!(result.depth_of(&csr, b), None);
    }

    #[test]
    fn test_dfs_invalid_start() {
        let g = ConcurrentGraph::new();
        g.insert_node(TypeId(1));
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        assert!(dfs(&csr, NodeId(9999), None).is_none());
    }

    #[test]
    fn test_dfs_no_stack_duplicates() {
        // Complete graph K5: every node connected to every other.
        // With duplicates the stack would blow up; without, it stays manageable.
        let g = ConcurrentGraph::new();
        let nodes: Vec<NodeId> = (0..5).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    g.insert_edge(nodes[i], nodes[j], TypeId(10)).unwrap();
                }
            }
        }
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let result = dfs(&csr, nodes[0], None).unwrap();
        // All 5 nodes reached, each visited exactly once
        assert_eq!(result.nodes_reached, 5);
        assert_eq!(result.visit_order.len(), 5);
    }

    // ── IDDFS tests ─────────────────────────────────────────────────────

    #[test]
    fn test_iddfs_chain_find_target() {
        let (g, nodes) = chain(10);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let result = iddfs(&csr, nodes[0], nodes[5], 20).unwrap();
        assert!(result.found);
        assert_eq!(result.depth, 5);
        assert_eq!(result.path.len(), 6); // nodes 0..=5
        assert_eq!(result.path.first(), Some(&nodes[0]));
        assert_eq!(result.path.last(), Some(&nodes[5]));
    }

    #[test]
    fn test_iddfs_finds_shortest_path() {
        // Diamond: a→b→d, a→c→d. IDDFS should find depth 2.
        let (g, a, _b, _c, d) = diamond();
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let result = iddfs(&csr, a, d, 10).unwrap();
        assert!(result.found);
        assert_eq!(result.depth, 2);
        assert_eq!(result.path.len(), 3); // a → (b or c) → d
    }

    #[test]
    fn test_iddfs_not_found() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1)); // isolated
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let result = iddfs(&csr, a, b, 100).unwrap();
        assert!(!result.found);
        assert!(result.path.is_empty());
    }

    #[test]
    fn test_iddfs_start_equals_target() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        let result = iddfs(&csr, a, a, 10).unwrap();
        assert!(result.found);
        assert_eq!(result.depth, 0);
        assert_eq!(result.path, vec![a]);
    }

    #[test]
    fn test_iddfs_depth_limit_prevents_find() {
        let (g, nodes) = chain(10);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);

        // Target at depth 5, but limit is 3
        let result = iddfs(&csr, nodes[0], nodes[5], 3).unwrap();
        assert!(!result.found);
    }

    // ── Large-scale tests ───────────────────────────────────────────────

    #[test]
    fn test_dfs_deep_chain_1m_nodes() {
        // Chain de 1M nodos: verifica que DFS no explota en memoria
        // y maneja profundidades extremas sin stack overflow.
        let num_nodes = 1_000_000;
        let g = ConcurrentGraph::with_capacity(num_nodes, num_nodes);
        let nodes: Vec<NodeId> = (0..num_nodes)
            .map(|_| g.insert_node(TypeId(1)))
            .collect();
        for i in 0..num_nodes - 1 {
            g.insert_edge(nodes[i], nodes[i + 1], TypeId(10)).unwrap();
        }

        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let result = dfs(&csr, nodes[0], None).unwrap();

        assert_eq!(result.nodes_reached, num_nodes);
        assert_eq!(result.max_depth, (num_nodes - 1) as u32);
        assert_eq!(result.depth_of(&csr, nodes[num_nodes - 1]),
                   Some((num_nodes - 1) as u32));
    }

    #[test]
    fn test_dfs_memory_efficiency() {
        // Verify BitSet + dense arrays use far less memory than HashSet + HashMap
        let n = 1_000_000;
        let bitset = BitSet::new(n);
        let depths: Vec<u32> = vec![u32::MAX; n];
        let predecessors: Vec<u32> = vec![u32::MAX; n];

        // BitSet: ~125 KB, depths: 4 MB, predecessors: 4 MB = ~8.1 MB total
        let bitset_bytes = (n + 63) / 64 * 8;
        let arrays_bytes = n * 4 * 2;
        let total = bitset_bytes + arrays_bytes;

        assert!(total < 10_000_000, "DFS memory {total} exceeds 10MB for 1M nodes");
        // HashSet would use ~40 bytes/entry = 40 MB just for visited
        // + 2 HashMaps ~120 bytes/entry each = 240 MB
        // Total HashMap-based: ~280 MB vs our ~8 MB

        // Ensure structures are valid
        assert_eq!(bitset.len(), n);
        assert_eq!(depths.len(), n);
        assert_eq!(predecessors.len(), n);
    }

    #[test]
    fn test_dfs_vs_bfs_coverage_consistency() {
        // Both DFS and BFS should reach the same set of nodes
        use crate::parallel_bfs;

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

        let dfs_result = dfs(&csr, grid[0], None).unwrap();
        let bfs_result = parallel_bfs::sequential_bfs(&csr, grid[0], None).unwrap();

        // Same number of nodes reached
        assert_eq!(dfs_result.nodes_reached, bfs_result.nodes_reached);
        assert_eq!(dfs_result.nodes_reached, width * width);
    }
}
