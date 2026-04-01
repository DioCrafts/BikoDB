// =============================================================================
// Generadores de datos sintéticos para benchmarks
// =============================================================================

use bikodb_core::types::{NodeId, TypeId};
use bikodb_core::value::Value;
use bikodb_graph::ConcurrentGraph;
use rand::Rng;

/// Genera un grafo tipo social network con N nodos y ~M edges por nodo.
pub fn generate_social_graph(num_nodes: usize, avg_edges_per_node: usize) -> ConcurrentGraph {
    let graph = ConcurrentGraph::with_capacity(num_nodes, num_nodes * avg_edges_per_node);
    let mut rng = rand::thread_rng();

    let person = TypeId(1);
    let knows = TypeId(10);

    // Insert nodes
    let nodes: Vec<NodeId> = (0..num_nodes)
        .map(|i| {
            graph.insert_node_with_props(
                person,
                vec![
                    (0, Value::from(format!("user_{i}"))),
                    (1, Value::Int(rng.gen_range(18..80))),
                ],
            )
        })
        .collect();

    // Insert random edges (power-law approximation)
    for &node in &nodes {
        let num_edges = rng.gen_range(1..=avg_edges_per_node * 2);
        for _ in 0..num_edges {
            let target_idx = rng.gen_range(0..num_nodes);
            let target = nodes[target_idx];
            if node != target {
                let _ = graph.insert_edge(node, target, knows);
            }
        }
    }

    graph
}

/// Genera vectores aleatorios de dimensión dada.
pub fn generate_random_vectors(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..dimensions).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}
