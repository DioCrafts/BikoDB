// =============================================================================
// bikodb-graph::scc — Strongly Connected Components (Tarjan + Kosaraju)
// =============================================================================
//
// ## Algoritmos
//
// | Algoritmo | Complejidad | Paralelo | Uso                              |
// |-----------|-------------|----------|----------------------------------|
// | Tarjan    | O(V + E)    | No       | Single-threaded, óptimo serial   |
// | Kosaraju  | O(V + E)    | Parcial  | Parallelizable en cada BFS/DFS   |
//
// ## SCC (Strongly Connected Components)
//
// Encuentra todos los conjuntos máximos de nodos tales que existe un camino
// dirigido de cualquier nodo a cualquier otro nodo dentro del mismo componente.
//
// ### Tarjan's Algorithm
// - DFS iterativo (no recursivo, evita stack overflow en grafos grandes)
// - Un solo pass: O(V + E) tiempo, O(V) espacio extra
// - Genera SCCs en orden topológico inverso
//
// ### Kosaraju's Algorithm
// - Dos passes: (1) DFS forward → finish order, (2) DFS reverse → SCCs
// - Más simple conceptualmente, BFS paralelo posible en fase 2
//
// ## Referencia
// - LDBC Graphalytics no incluye SCC, pero Kuzu sí (SCC + Kosaraju)
// - ArcadeDB no incluye SCC en su capa OLAP
// - BikoDB incluye ambos para máxima cobertura
// =============================================================================

use crate::csr::CsrGraph;
use bikodb_core::record::Direction;
use bikodb_core::types::NodeId;

use crate::graph::ConcurrentGraph;

/// Resultado de SCC.
#[derive(Debug)]
pub struct SccResult {
    /// Component ID para cada nodo (indexado por internal id).
    pub components: Vec<u32>,
    /// Número total de SCCs encontrados.
    pub num_components: usize,
}

impl SccResult {
    /// Component ID de un nodo externo.
    pub fn component_of(&self, csr: &CsrGraph, node: NodeId) -> Option<u32> {
        let internal = csr.to_internal(node)?;
        Some(self.components[internal as usize])
    }

    /// ¿Están dos nodos en el mismo SCC?
    pub fn same_component(&self, csr: &CsrGraph, a: NodeId, b: NodeId) -> Option<bool> {
        let ca = self.component_of(csr, a)?;
        let cb = self.component_of(csr, b)?;
        Some(ca == cb)
    }

    /// Tamaños de componentes: Vec<(component_id, count)> ordenado desc por count.
    pub fn component_sizes(&self) -> Vec<(u32, usize)> {
        let mut counts = std::collections::HashMap::new();
        for &c in &self.components {
            *counts.entry(c).or_insert(0usize) += 1;
        }
        let mut sizes: Vec<(u32, usize)> = counts.into_iter().collect();
        sizes.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        sizes
    }
}

// =============================================================================
// SCC — Tarjan's Algorithm (iterativo)
// =============================================================================

/// SCC via Tarjan's algorithm sobre ConcurrentGraph.
pub fn tarjan_scc(graph: &ConcurrentGraph) -> SccResult {
    let out_csr = CsrGraph::from_concurrent(graph, Direction::Out);
    tarjan_scc_on_csr(&out_csr)
}

/// SCC via Tarjan's algorithm sobre CSR (iterativo, sin recursión).
///
/// Usa una pila explícita para evitar stack overflow en grafos con millones de nodos.
pub fn tarjan_scc_on_csr(csr: &CsrGraph) -> SccResult {
    let n = csr.num_nodes();
    if n == 0 {
        return SccResult {
            components: Vec::new(),
            num_components: 0,
        };
    }

    let mut index_counter: u32 = 0;
    let mut indices = vec![u32::MAX; n]; // u32::MAX = undefined
    let mut lowlinks = vec![0u32; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<u32> = Vec::new();
    let mut components = vec![u32::MAX; n];
    let mut comp_id: u32 = 0;

    // Iterative Tarjan: explicit DFS stack
    // Each frame: (node, neighbor_index)
    let mut dfs_stack: Vec<(u32, usize)> = Vec::new();

    for root in 0..n as u32 {
        if indices[root as usize] != u32::MAX {
            continue;
        }

        // Push root
        indices[root as usize] = index_counter;
        lowlinks[root as usize] = index_counter;
        index_counter += 1;
        on_stack[root as usize] = true;
        stack.push(root);
        dfs_stack.push((root, 0));

        while let Some(&mut (v, ref mut ni)) = dfs_stack.last_mut() {
            let neighbors = csr.neighbors(v);
            if *ni < neighbors.len() {
                let w = neighbors[*ni];
                *ni += 1;

                if indices[w as usize] == u32::MAX {
                    // Tree edge: push w
                    indices[w as usize] = index_counter;
                    lowlinks[w as usize] = index_counter;
                    index_counter += 1;
                    on_stack[w as usize] = true;
                    stack.push(w);
                    dfs_stack.push((w, 0));
                } else if on_stack[w as usize] {
                    // Back edge
                    lowlinks[v as usize] =
                        lowlinks[v as usize].min(indices[w as usize]);
                }
            } else {
                // All neighbors processed: check if root of SCC
                if lowlinks[v as usize] == indices[v as usize] {
                    // Pop SCC from stack
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack[w as usize] = false;
                        components[w as usize] = comp_id;
                        if w == v {
                            break;
                        }
                    }
                    comp_id += 1;
                }

                dfs_stack.pop();
                // Update parent's lowlink
                if let Some(&mut (parent, _)) = dfs_stack.last_mut() {
                    lowlinks[parent as usize] =
                        lowlinks[parent as usize].min(lowlinks[v as usize]);
                }
            }
        }
    }

    SccResult {
        components,
        num_components: comp_id as usize,
    }
}

// =============================================================================
// SCC — Kosaraju's Algorithm
// =============================================================================

/// SCC via Kosaraju's algorithm (requiere CSR in + out).
///
/// Fase 1: DFS forward → finish order
/// Fase 2: DFS reverse en finish order → SCCs
pub fn kosaraju_scc(graph: &ConcurrentGraph) -> SccResult {
    let out_csr = CsrGraph::from_concurrent(graph, Direction::Out);
    let in_csr = CsrGraph::from_concurrent(graph, Direction::In);
    kosaraju_scc_on_csr(&out_csr, &in_csr)
}

/// SCC via Kosaraju sobre CSR pre-construidos.
pub fn kosaraju_scc_on_csr(out_csr: &CsrGraph, in_csr: &CsrGraph) -> SccResult {
    let n = out_csr.num_nodes();
    if n == 0 {
        return SccResult {
            components: Vec::new(),
            num_components: 0,
        };
    }

    // Phase 1: Forward DFS (iterative) → finish order
    let mut visited = vec![false; n];
    let mut finish_order: Vec<u32> = Vec::with_capacity(n);

    for root in 0..n as u32 {
        if visited[root as usize] {
            continue;
        }
        // Iterative DFS
        let mut dfs_stack: Vec<(u32, usize)> = vec![(root, 0)];
        visited[root as usize] = true;

        while let Some(&mut (v, ref mut ni)) = dfs_stack.last_mut() {
            let neighbors = out_csr.neighbors(v);
            if *ni < neighbors.len() {
                let w = neighbors[*ni];
                *ni += 1;
                if !visited[w as usize] {
                    visited[w as usize] = true;
                    dfs_stack.push((w, 0));
                }
            } else {
                finish_order.push(v);
                dfs_stack.pop();
            }
        }
    }

    // Phase 2: Reverse DFS in reverse finish order → SCCs
    let mut components = vec![u32::MAX; n];
    let mut comp_id: u32 = 0;

    for &root in finish_order.iter().rev() {
        if components[root as usize] != u32::MAX {
            continue;
        }
        // BFS/DFS on reverse graph
        let mut queue: Vec<u32> = vec![root];
        components[root as usize] = comp_id;
        while let Some(v) = queue.pop() {
            for &w in in_csr.neighbors(v) {
                if components[w as usize] == u32::MAX {
                    components[w as usize] = comp_id;
                    queue.push(w);
                }
            }
        }
        comp_id += 1;
    }

    SccResult {
        components,
        num_components: comp_id as usize,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bikodb_core::types::TypeId;

    fn make_directed_graph(edges: &[(usize, usize)]) -> ConcurrentGraph {
        let g = ConcurrentGraph::new();
        let max_node = edges.iter().flat_map(|&(a, b)| [a, b]).max().unwrap_or(0) + 1;
        let nodes: Vec<NodeId> = (0..max_node).map(|_| g.insert_node(TypeId(1))).collect();
        for &(a, b) in edges {
            let _ = g.insert_edge(nodes[a], nodes[b], TypeId(10));
        }
        g
    }

    fn make_directed_graph_with_nodes(
        edges: &[(usize, usize)],
    ) -> (ConcurrentGraph, Vec<NodeId>) {
        let g = ConcurrentGraph::new();
        let max_node = edges.iter().flat_map(|&(a, b)| [a, b]).max().unwrap_or(0) + 1;
        let nodes: Vec<NodeId> = (0..max_node).map(|_| g.insert_node(TypeId(1))).collect();
        for &(a, b) in edges {
            let _ = g.insert_edge(nodes[a], nodes[b], TypeId(10));
        }
        (g, nodes)
    }

    // ── Tarjan ──────────────────────────────────────────────────────────

    #[test]
    fn test_tarjan_single_cycle() {
        // 0 → 1 → 2 → 0 (one SCC of size 3)
        let g = make_directed_graph(&[(0, 1), (1, 2), (2, 0)]);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let result = tarjan_scc_on_csr(&csr);
        assert_eq!(result.num_components, 1);
    }

    #[test]
    fn test_tarjan_two_sccs() {
        // SCC1: 0→1→2→0, SCC2: 3→4→3, bridge: 2→3
        let g = make_directed_graph(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3)]);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let result = tarjan_scc_on_csr(&csr);
        assert_eq!(result.num_components, 2);
        // 0,1,2 same component
        assert_eq!(result.components[0], result.components[1]);
        assert_eq!(result.components[1], result.components[2]);
        // 3,4 same component
        assert_eq!(result.components[3], result.components[4]);
        // Different SCCs
        assert_ne!(result.components[0], result.components[3]);
    }

    #[test]
    fn test_tarjan_dag_all_singletons() {
        // 0→1→2→3 (DAG, each node is its own SCC)
        let g = make_directed_graph(&[(0, 1), (1, 2), (2, 3)]);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let result = tarjan_scc_on_csr(&csr);
        assert_eq!(result.num_components, 4);
    }

    #[test]
    fn test_tarjan_self_loop() {
        // 0→0 (self-loop = SCC of size 1)
        let g = ConcurrentGraph::new();
        let n = g.insert_node(TypeId(1));
        let _ = g.insert_edge(n, n, TypeId(10));
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let result = tarjan_scc_on_csr(&csr);
        assert_eq!(result.num_components, 1);
    }

    #[test]
    fn test_tarjan_empty_graph() {
        let g = ConcurrentGraph::new();
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let result = tarjan_scc_on_csr(&csr);
        assert_eq!(result.num_components, 0);
    }

    #[test]
    fn test_tarjan_isolated_nodes() {
        let g = ConcurrentGraph::new();
        for _ in 0..5 {
            g.insert_node(TypeId(1));
        }
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let result = tarjan_scc_on_csr(&csr);
        assert_eq!(result.num_components, 5);
    }

    #[test]
    fn test_tarjan_complex_graph() {
        // Wikipedia Tarjan example:
        //   SCC1: {0,1,2}, SCC2: {3,4,5,6}, SCC3: {7}
        let edges = &[
            (0, 1), (1, 2), (2, 0),     // SCC1
            (3, 4), (4, 5), (5, 6), (6, 3), // SCC2
            (2, 3),                       // bridge SCC1→SCC2
            (5, 7),                       // bridge SCC2→SCC3
        ];
        let g = make_directed_graph(edges);
        let csr = CsrGraph::from_concurrent(&g, Direction::Out);
        let result = tarjan_scc_on_csr(&csr);
        assert_eq!(result.num_components, 3);
    }

    // ── Kosaraju ────────────────────────────────────────────────────────

    #[test]
    fn test_kosaraju_single_cycle() {
        let g = make_directed_graph(&[(0, 1), (1, 2), (2, 0)]);
        let result = kosaraju_scc(&g);
        assert_eq!(result.num_components, 1);
    }

    #[test]
    fn test_kosaraju_two_sccs() {
        let g = make_directed_graph(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3)]);
        let result = kosaraju_scc(&g);
        assert_eq!(result.num_components, 2);
    }

    #[test]
    fn test_kosaraju_dag() {
        let g = make_directed_graph(&[(0, 1), (1, 2), (2, 3)]);
        let result = kosaraju_scc(&g);
        assert_eq!(result.num_components, 4);
    }

    #[test]
    fn test_kosaraju_matches_tarjan() {
        // Both algorithms should find the same number of SCCs
        let edges = &[
            (0, 1), (1, 2), (2, 0),
            (3, 4), (4, 5), (5, 6), (6, 3),
            (2, 3), (5, 7),
        ];
        let (g, nodes) = make_directed_graph_with_nodes(edges);
        let tarjan = tarjan_scc(&g);
        let kosaraju = kosaraju_scc(&g);
        assert_eq!(tarjan.num_components, kosaraju.num_components);

        // Same groupings: nodes in same SCC in Tarjan should be in same SCC in Kosaraju
        let out_csr = CsrGraph::from_concurrent(&g, Direction::Out);
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let t_same = tarjan.same_component(&out_csr, nodes[i], nodes[j]);
                let k_same = kosaraju.same_component(&out_csr, nodes[i], nodes[j]);
                assert_eq!(t_same, k_same, "Mismatch for nodes {} and {}", i, j);
            }
        }
    }

    #[test]
    fn test_scc_component_sizes() {
        let g = make_directed_graph(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3)]);
        let result = tarjan_scc(&g);
        let sizes = result.component_sizes();
        assert_eq!(sizes.len(), 2);
        // One SCC of size 3, one of size 2
        let mut size_list: Vec<usize> = sizes.iter().map(|(_, s)| *s).collect();
        size_list.sort_unstable();
        assert_eq!(size_list, vec![2, 3]);
    }
}
