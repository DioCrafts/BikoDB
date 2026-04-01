// =============================================================================
// bikodb-graph::traversal — Algoritmos de recorrido de grafos
// =============================================================================
// BFS, DFS, shortest path y traversals filtrados.
//
// Diseño:
// - Visitor pattern para traversals genéricos
// - Resultado como Iterator lazy (cuando sea posible)
// - Soporte para profundidad máxima, filtros de tipo, early termination
//
// Inspirado en:
// - ArcadeDB: SQL TRAVERSE, Gremlin steps
// - Neo4j: traversal framework con Evaluator/BranchSelector
// =============================================================================

use crate::graph::ConcurrentGraph;
use bikodb_core::error::{BikoError, BikoResult};
use bikodb_core::record::Direction;
use bikodb_core::types::{NodeId, TypeId};
use std::collections::{HashMap, HashSet, VecDeque};

/// Resultado de un traversal: nodos visitados con su profundidad.
#[derive(Debug, Clone)]
pub struct TraversalResult {
    /// Nodos visitados en orden de descubrimiento
    pub visited_order: Vec<NodeId>,
    /// Mapa nodo → profundidad (distancia desde el inicio)
    pub depths: HashMap<NodeId, u32>,
    /// Mapa nodo → predecesor (para reconstruir paths)
    pub predecessors: HashMap<NodeId, NodeId>,
}

impl TraversalResult {
    fn new() -> Self {
        Self {
            visited_order: Vec::new(),
            depths: HashMap::new(),
            predecessors: HashMap::new(),
        }
    }

    /// Reconstruye el path desde el inicio hasta un nodo target.
    pub fn path_to(&self, target: NodeId) -> Option<Vec<NodeId>> {
        if !self.depths.contains_key(&target) {
            return None;
        }

        let mut path = vec![target];
        let mut current = target;

        while let Some(&pred) = self.predecessors.get(&current) {
            path.push(pred);
            current = pred;
        }

        path.reverse();
        Some(path)
    }
}

/// BFS (Breadth-First Search) desde un nodo inicio.
///
/// Recorre el grafo por niveles, expandiendo todos los vecinos de un nivel
/// antes de pasar al siguiente. Ideal para shortest path en grafos no ponderados.
///
/// # Parámetros
/// - `graph`: grafo a recorrer
/// - `start`: nodo inicial
/// - `direction`: dirección de los edges a seguir
/// - `max_depth`: profundidad máxima (None = sin límite)
///
/// # Ejemplo
/// ```
/// use bikodb_graph::{ConcurrentGraph, traversal};
/// use bikodb_core::types::{NodeId, TypeId};
/// use bikodb_core::record::Direction;
///
/// let g = ConcurrentGraph::new();
/// let a = g.insert_node(TypeId(1));
/// let b = g.insert_node(TypeId(1));
/// let c = g.insert_node(TypeId(1));
/// g.insert_edge(a, b, TypeId(10)).unwrap();
/// g.insert_edge(b, c, TypeId(10)).unwrap();
///
/// let result = traversal::bfs(&g, a, Direction::Out, Some(10)).unwrap();
/// assert_eq!(result.visited_order, vec![a, b, c]);
/// assert_eq!(result.depths[&c], 2);
/// ```
pub fn bfs(
    graph: &ConcurrentGraph,
    start: NodeId,
    direction: Direction,
    max_depth: Option<u32>,
) -> BikoResult<TraversalResult> {
    if !graph.contains_node(start) {
        return Err(BikoError::NodeNotFound(start));
    }

    let mut result = TraversalResult::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    visited.insert(start);
    queue.push_back((start, 0u32));
    result.visited_order.push(start);
    result.depths.insert(start, 0);

    while let Some((current, depth)) = queue.pop_front() {
        if let Some(max) = max_depth {
            if depth >= max {
                continue;
            }
        }

        let neighbors = graph.neighbors(current, direction)?;
        for neighbor in neighbors {
            if visited.insert(neighbor) {
                let next_depth = depth + 1;
                result.visited_order.push(neighbor);
                result.depths.insert(neighbor, next_depth);
                result.predecessors.insert(neighbor, current);
                queue.push_back((neighbor, next_depth));
            }
        }
    }

    Ok(result)
}

/// DFS (Depth-First Search) desde un nodo inicio.
///
/// Recorre el grafo en profundidad, explorando lo más lejos posible por cada
/// rama antes de retroceder. Útil para detección de ciclos y componentes.
///
/// # Parámetros
/// - `graph`: grafo a recorrer
/// - `start`: nodo inicial
/// - `direction`: dirección de los edges a seguir
/// - `max_depth`: profundidad máxima (None = sin límite)
pub fn dfs(
    graph: &ConcurrentGraph,
    start: NodeId,
    direction: Direction,
    max_depth: Option<u32>,
) -> BikoResult<TraversalResult> {
    if !graph.contains_node(start) {
        return Err(BikoError::NodeNotFound(start));
    }

    let mut result = TraversalResult::new();
    let mut visited = HashSet::new();
    let mut stack = Vec::new();

    stack.push((start, 0u32));

    while let Some((current, depth)) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }

        result.visited_order.push(current);
        result.depths.insert(current, depth);

        if let Some(max) = max_depth {
            if depth >= max {
                continue;
            }
        }

        let neighbors = graph.neighbors(current, direction)?;
        // Push in reverse order so first neighbor is processed first
        for neighbor in neighbors.into_iter().rev() {
            if !visited.contains(&neighbor) {
                result.predecessors.insert(neighbor, current);
                stack.push((neighbor, depth + 1));
            }
        }
    }

    Ok(result)
}

/// BFS filtrado por tipo de edge.
///
/// Solo sigue edges del tipo indicado. Útil para queries como
/// "amigos de amigos" (solo edges KNOWS).
pub fn bfs_by_edge_type(
    graph: &ConcurrentGraph,
    start: NodeId,
    direction: Direction,
    edge_type: TypeId,
    max_depth: Option<u32>,
) -> BikoResult<TraversalResult> {
    if !graph.contains_node(start) {
        return Err(BikoError::NodeNotFound(start));
    }

    let mut result = TraversalResult::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    visited.insert(start);
    queue.push_back((start, 0u32));
    result.visited_order.push(start);
    result.depths.insert(start, 0);

    while let Some((current, depth)) = queue.pop_front() {
        if let Some(max) = max_depth {
            if depth >= max {
                continue;
            }
        }

        let neighbors = graph.neighbors_by_type(current, direction, edge_type)?;
        for neighbor in neighbors {
            if visited.insert(neighbor) {
                let next_depth = depth + 1;
                result.visited_order.push(neighbor);
                result.depths.insert(neighbor, next_depth);
                result.predecessors.insert(neighbor, current);
                queue.push_back((neighbor, next_depth));
            }
        }
    }

    Ok(result)
}

/// Shortest path (BFS) entre dos nodos.
///
/// Retorna el camino más corto en un grafo no ponderado,
/// o None si no existe camino.
pub fn shortest_path(
    graph: &ConcurrentGraph,
    start: NodeId,
    end: NodeId,
    direction: Direction,
) -> BikoResult<Option<Vec<NodeId>>> {
    if !graph.contains_node(start) {
        return Err(BikoError::NodeNotFound(start));
    }
    if !graph.contains_node(end) {
        return Err(BikoError::NodeNotFound(end));
    }

    if start == end {
        return Ok(Some(vec![start]));
    }

    let mut visited = HashSet::new();
    let mut predecessors: HashMap<NodeId, NodeId> = HashMap::new();
    let mut queue = VecDeque::new();

    visited.insert(start);
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        let neighbors = graph.neighbors(current, direction)?;
        for neighbor in neighbors {
            if visited.insert(neighbor) {
                predecessors.insert(neighbor, current);

                if neighbor == end {
                    // Reconstruir path
                    let mut path = vec![end];
                    let mut curr = end;
                    while let Some(&pred) = predecessors.get(&curr) {
                        path.push(pred);
                        curr = pred;
                    }
                    path.reverse();
                    return Ok(Some(path));
                }

                queue.push_back(neighbor);
            }
        }
    }

    Ok(None) // No path found
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn build_chain(n: usize) -> (ConcurrentGraph, Vec<NodeId>) {
        let g = ConcurrentGraph::new();
        let nodes: Vec<NodeId> = (0..n).map(|_| g.insert_node(TypeId(1))).collect();
        for i in 0..n - 1 {
            g.insert_edge(nodes[i], nodes[i + 1], TypeId(10)).unwrap();
        }
        (g, nodes)
    }

    fn build_diamond() -> (ConcurrentGraph, [NodeId; 4]) {
        //   A
        //  / \
        // B   C
        //  \ /
        //   D
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(1));
        let d = g.insert_node(TypeId(1));

        g.insert_edge(a, b, TypeId(10)).unwrap();
        g.insert_edge(a, c, TypeId(10)).unwrap();
        g.insert_edge(b, d, TypeId(10)).unwrap();
        g.insert_edge(c, d, TypeId(10)).unwrap();

        (g, [a, b, c, d])
    }

    #[test]
    fn test_bfs_chain() {
        let (g, nodes) = build_chain(5);
        let result = bfs(&g, nodes[0], Direction::Out, None).unwrap();

        assert_eq!(result.visited_order, nodes);
        assert_eq!(result.depths[&nodes[0]], 0);
        assert_eq!(result.depths[&nodes[4]], 4);
    }

    #[test]
    fn test_bfs_max_depth() {
        let (g, nodes) = build_chain(10);
        let result = bfs(&g, nodes[0], Direction::Out, Some(3)).unwrap();

        assert_eq!(result.visited_order.len(), 4); // depth 0,1,2,3
        assert!(!result.depths.contains_key(&nodes[4]));
    }

    #[test]
    fn test_bfs_diamond() {
        let (g, [a, b, c, d]) = build_diamond();
        let result = bfs(&g, a, Direction::Out, None).unwrap();

        assert_eq!(result.depths[&a], 0);
        assert_eq!(result.depths[&b], 1);
        assert_eq!(result.depths[&c], 1);
        assert_eq!(result.depths[&d], 2);
    }

    #[test]
    fn test_dfs_chain() {
        let (g, nodes) = build_chain(5);
        let result = dfs(&g, nodes[0], Direction::Out, None).unwrap();

        assert_eq!(result.visited_order.len(), 5);
        assert_eq!(result.visited_order[0], nodes[0]);
    }

    #[test]
    fn test_shortest_path() {
        let (g, [a, _b, _c, d]) = build_diamond();
        let path = shortest_path(&g, a, d, Direction::Out).unwrap().unwrap();

        assert_eq!(path.len(), 3); // A → B|C → D
        assert_eq!(path[0], a);
        assert_eq!(*path.last().unwrap(), d);
    }

    #[test]
    fn test_shortest_path_same_node() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let path = shortest_path(&g, a, a, Direction::Out).unwrap().unwrap();
        assert_eq!(path, vec![a]);
    }

    #[test]
    fn test_shortest_path_unreachable() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        // No edge between them

        let path = shortest_path(&g, a, b, Direction::Out).unwrap();
        assert!(path.is_none());
    }

    #[test]
    fn test_path_reconstruction() {
        let (g, nodes) = build_chain(5);
        let result = bfs(&g, nodes[0], Direction::Out, None).unwrap();

        let path = result.path_to(nodes[4]).unwrap();
        assert_eq!(path, nodes);
    }

    #[test]
    fn test_bfs_by_edge_type() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(TypeId(1));
        let b = g.insert_node(TypeId(1));
        let c = g.insert_node(TypeId(2));

        let knows = TypeId(10);
        let works_at = TypeId(11);

        g.insert_edge(a, b, knows).unwrap();
        g.insert_edge(a, c, works_at).unwrap();

        let result = bfs_by_edge_type(&g, a, Direction::Out, knows, None).unwrap();
        assert_eq!(result.visited_order.len(), 2); // a, b (not c)
        assert!(result.depths.contains_key(&b));
        assert!(!result.depths.contains_key(&c));
    }

    #[test]
    fn test_bfs_invalid_start() {
        let g = ConcurrentGraph::new();
        let result = bfs(&g, NodeId(999), Direction::Out, None);
        assert!(result.is_err());
    }
}
