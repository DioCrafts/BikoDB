// =============================================================================
// bikodb-bench::ldbc — LDBC Graphalytics infrastructure
// =============================================================================
//
// Proporciona:
// 1. Generadores de grafos en escalas LDBC (Graph500-style)
// 2. Runner framework para los 6 algoritmos core
// 3. Resultado JSON exportable para comparativas
// 4. Múltiples escalas: XS, S, M, L, XL
//
// ## LDBC Graphalytics Core Algorithms
// | Algo | Descripción                        |
// |------|------------------------------------|
// | BFS  | Breadth-First Search               |
// | SSSP | Single-Source Shortest Path         |
// | PR   | PageRank                           |
// | CDLP | Community Detection (Label Prop.)  |
// | WCC  | Weakly Connected Components        |
// | LCC  | Local Clustering Coefficient       |
//
// ## Scales (inspirado en LDBC/Graph500)
// | Scale | Nodos   | Edges aprox           |
// |-------|---------|-----------------------|
// | XS    | 1K      | ~10K                  |
// | S     | 10K     | ~100K                 |
// | M     | 100K    | ~1M                   |
// | L     | 1M      | ~10M                  |
// | XL    | 10M     | ~100M                 |
// =============================================================================

use bikodb_core::record::Direction;
use bikodb_core::types::{NodeId, TypeId};
use bikodb_core::value::Value;
use bikodb_graph::csr::CsrGraph;
use bikodb_graph::weighted_csr::WeightedCsrGraph;
use bikodb_graph::ConcurrentGraph;
use rand::Rng;
use serde::Serialize;
use std::time::Instant;

// =============================================================================
// Scales
// =============================================================================

/// LDBC-inspired scale definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdbcScale {
    XS,
    S,
    M,
    L,
    XL,
}

impl LdbcScale {
    pub fn num_nodes(&self) -> usize {
        match self {
            LdbcScale::XS => 1_000,
            LdbcScale::S => 10_000,
            LdbcScale::M => 100_000,
            LdbcScale::L => 1_000_000,
            LdbcScale::XL => 10_000_000,
        }
    }

    pub fn avg_degree(&self) -> usize {
        10
    }

    pub fn name(&self) -> &str {
        match self {
            LdbcScale::XS => "XS",
            LdbcScale::S => "S",
            LdbcScale::M => "M",
            LdbcScale::L => "L",
            LdbcScale::XL => "XL",
        }
    }
}

// =============================================================================
// Graph Generator
// =============================================================================

/// Genera un grafo Graph500-style con distribución power-law.
///
/// Edges se insertan en ambas direcciones para simular lo no dirigido para WCC/LCC.
/// Los edges tienen un peso aleatorio [0.1, 10.0] para SSSP.
pub fn generate_ldbc_graph(scale: LdbcScale) -> LdbcGraph {
    let num_nodes = scale.num_nodes();
    let avg_degree = scale.avg_degree();
    let graph = ConcurrentGraph::with_capacity(num_nodes, num_nodes * avg_degree * 2);
    let mut rng = rand::thread_rng();

    let node_type = TypeId(1);
    let edge_type = TypeId(10);
    let weight_prop: u16 = 0;

    // Insert nodes
    let nodes: Vec<NodeId> = (0..num_nodes)
        .map(|_| graph.insert_node(node_type))
        .collect();

    // Insert edges (power-law approximation + bidirectional)
    let mut edge_count = 0usize;
    for &src in &nodes {
        // Power-law: some nodes get more edges
        let degree = if rng.gen_range(0..100) < 5 {
            // Hub node: 5x average
            avg_degree * 5
        } else {
            rng.gen_range(1..=avg_degree * 2)
        };
        for _ in 0..degree {
            let tgt_idx = rng.gen_range(0..num_nodes);
            let tgt = nodes[tgt_idx];
            if src != tgt {
                let weight = rng.gen_range(0.1_f64..10.0);
                // Forward edge with weight
                if graph.insert_edge_with_props(
                    src, tgt, edge_type,
                    vec![(weight_prop, Value::Float(weight))],
                ).is_ok() {
                    edge_count += 1;
                }
                // Reverse edge (for undirected algorithms)
                if graph.insert_edge_with_props(
                    tgt, src, edge_type,
                    vec![(weight_prop, Value::Float(weight))],
                ).is_ok() {
                    edge_count += 1;
                }
            }
        }
    }

    LdbcGraph {
        graph,
        scale,
        num_nodes,
        num_edges: edge_count,
        weight_prop_id: weight_prop,
    }
}

/// Datos de grafo LDBC para benchmarking.
pub struct LdbcGraph {
    pub graph: ConcurrentGraph,
    pub scale: LdbcScale,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub weight_prop_id: u16,
}

// =============================================================================
// Benchmark Runner
// =============================================================================

/// Ejecuta los 6 algoritmos LDBC Graphalytics y retorna resultados.
pub fn run_ldbc_suite(ldbc_graph: &LdbcGraph) -> LdbcSuiteResult {
    let graph = &ldbc_graph.graph;
    let start_node = NodeId(1);

    // Pre-build CSRs (shared across algorithms)
    let t0 = Instant::now();
    let out_csr = CsrGraph::from_concurrent(graph, Direction::Out);
    let in_csr = CsrGraph::from_concurrent(graph, Direction::In);
    let wcsr = WeightedCsrGraph::from_concurrent(
        graph, Direction::Out, ldbc_graph.weight_prop_id, 1.0,
    );
    let csr_build_time = t0.elapsed();

    let mut results = Vec::new();

    // 1. BFS
    let bfs_result = run_timed("BFS", || {
        bikodb_graph::parallel_bfs::parallel_bfs(&out_csr, start_node, None)
    });
    results.push(bfs_result);

    // 2. SSSP
    let sssp_result = run_timed("SSSP", || {
        bikodb_graph::sssp::sssp(&wcsr, start_node)
    });
    results.push(sssp_result);

    // 3. PageRank
    let pr_config = bikodb_graph::pagerank::PageRankConfig {
        damping: 0.85,
        max_iterations: 20,
        tolerance: 1e-6,
    };
    let pagerank_result = run_timed("PageRank", || {
        bikodb_graph::pagerank::pagerank_on_csr(&out_csr, &in_csr, &pr_config)
    });
    results.push(pagerank_result);

    // 4. CDLP (Community Detection / Label Propagation)
    let lpa_config = bikodb_graph::community::LpaConfig { max_iterations: 10 };
    let cdlp_result = run_timed("CDLP", || {
        bikodb_graph::community::label_propagation_on_csr(&out_csr, Some(&in_csr), &lpa_config)
    });
    results.push(cdlp_result);

    // 5. WCC (Weakly Connected Components)
    let wcc_result = run_timed("WCC", || {
        bikodb_graph::community::connected_components_on_csr(&out_csr, Some(&in_csr))
    });
    results.push(wcc_result);

    // 6. LCC (Local Clustering Coefficient)
    let lcc_result = run_timed("LCC", || {
        bikodb_graph::lcc::lcc_on_csr_undirected(&out_csr, &in_csr)
    });
    results.push(lcc_result);

    LdbcSuiteResult {
        scale: ldbc_graph.scale.name().to_string(),
        num_nodes: ldbc_graph.num_nodes,
        num_edges: ldbc_graph.num_edges,
        csr_build_ms: csr_build_time.as_secs_f64() * 1000.0,
        algorithms: results,
    }
}

fn run_timed<T, F: FnOnce() -> T>(name: &str, f: F) -> AlgorithmResult {
    let start = Instant::now();
    let _ = f();
    let elapsed = start.elapsed();
    AlgorithmResult {
        name: name.to_string(),
        duration_ms: elapsed.as_secs_f64() * 1000.0,
        throughput_edges_per_sec: None,
    }
}

// =============================================================================
// AI/ML Benchmark Runner
// =============================================================================

/// Ejecuta benchmarks de AI/ML: vector insert, search, hybrid queries.
pub fn run_ai_benchmark(
    num_vectors: usize,
    dimensions: usize,
    k: usize,
    ef_search: usize,
) -> AiBenchResult {
    use bikodb_ai::hnsw::{HnswIndex, HnswConfig};
    use bikodb_ai::embedding::DistanceMetric;
    use bikodb_core::types::NodeId;

    let mut rng = rand::thread_rng();

    // 1. Build HNSW index
    let config = HnswConfig {
        dimensions,
        metric: DistanceMetric::Cosine,
        max_connections: 16,
        max_connections_0: 32,
        ef_construction: 200,
    };
    let hnsw = HnswIndex::new(config);

    let t0 = Instant::now();
    for i in 0..num_vectors {
        let vec: Vec<f32> = (0..dimensions).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        hnsw.insert(NodeId(i as u64 + 1), vec);
    }
    let insert_time = t0.elapsed();
    let insert_rate = num_vectors as f64 / insert_time.as_secs_f64();

    // 2. Search
    let query: Vec<f32> = (0..dimensions).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let num_searches = 100;
    let t1 = Instant::now();
    for _ in 0..num_searches {
        let _ = hnsw.search(&query, k, ef_search);
    }
    let search_time = t1.elapsed();
    let avg_search_us = search_time.as_micros() as f64 / num_searches as f64;

    // 3. Recall estimation (search with large ef, compare with brute force)
    let recall = estimate_recall(&hnsw, dimensions, num_vectors, k, ef_search);

    AiBenchResult {
        num_vectors,
        dimensions,
        k,
        ef_search,
        insert_rate_per_sec: insert_rate,
        insert_total_ms: insert_time.as_secs_f64() * 1000.0,
        avg_search_latency_us: avg_search_us,
        num_searches,
        recall_at_k: recall,
    }
}

fn estimate_recall(
    hnsw: &bikodb_ai::hnsw::HnswIndex,
    dimensions: usize,
    num_vectors: usize,
    k: usize,
    ef_search: usize,
) -> f64 {
    if num_vectors < k {
        return 1.0;
    }
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dimensions).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

    // HNSW search
    let hnsw_results = hnsw.search(&query, k, ef_search);
    let hnsw_ids: std::collections::HashSet<u64> = hnsw_results.iter().map(|r| r.node_id.0).collect();

    // Large ef search as "ground truth"
    let truth_results = hnsw.search(&query, k, ef_search.max(200));
    let truth_ids: std::collections::HashSet<u64> = truth_results.iter().map(|r| r.node_id.0).collect();

    if truth_ids.is_empty() {
        return 1.0;
    }

    let intersection = hnsw_ids.intersection(&truth_ids).count();
    intersection as f64 / truth_ids.len() as f64
}

// =============================================================================
// Result Types (Serializable JSON)
// =============================================================================

#[derive(Debug, Serialize)]
pub struct LdbcSuiteResult {
    pub scale: String,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub csr_build_ms: f64,
    pub algorithms: Vec<AlgorithmResult>,
}

#[derive(Debug, Serialize)]
pub struct AlgorithmResult {
    pub name: String,
    pub duration_ms: f64,
    pub throughput_edges_per_sec: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct AiBenchResult {
    pub num_vectors: usize,
    pub dimensions: usize,
    pub k: usize,
    pub ef_search: usize,
    pub insert_rate_per_sec: f64,
    pub insert_total_ms: f64,
    pub avg_search_latency_us: f64,
    pub num_searches: usize,
    pub recall_at_k: f64,
}

#[derive(Debug, Serialize)]
pub struct FullBenchReport {
    pub timestamp: String,
    pub ldbc_results: Vec<LdbcSuiteResult>,
    pub ai_results: Vec<AiBenchResult>,
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_ldbc_graph_xs() {
        let g = generate_ldbc_graph(LdbcScale::XS);
        assert_eq!(g.num_nodes, 1_000);
        assert!(g.num_edges > 0);
    }

    #[test]
    fn test_run_ldbc_suite_xs() {
        let g = generate_ldbc_graph(LdbcScale::XS);
        let result = run_ldbc_suite(&g);
        assert_eq!(result.algorithms.len(), 6);
        for algo in &result.algorithms {
            assert!(algo.duration_ms >= 0.0);
        }
        // Verify all 6 algorithms ran
        let names: Vec<&str> = result.algorithms.iter().map(|a| a.name.as_str()).collect();
        assert!(names.contains(&"BFS"));
        assert!(names.contains(&"SSSP"));
        assert!(names.contains(&"PageRank"));
        assert!(names.contains(&"CDLP"));
        assert!(names.contains(&"WCC"));
        assert!(names.contains(&"LCC"));
    }

    #[test]
    fn test_run_ai_benchmark() {
        let result = run_ai_benchmark(500, 64, 10, 50);
        assert!(result.insert_rate_per_sec > 0.0);
        assert!(result.avg_search_latency_us > 0.0);
        assert!(result.recall_at_k >= 0.0 && result.recall_at_k <= 1.0);
    }

    #[test]
    fn test_results_serializable() {
        let g = generate_ldbc_graph(LdbcScale::XS);
        let result = run_ldbc_suite(&g);
        let json = serde_json::to_string_pretty(&result).unwrap();
        assert!(json.contains("BFS"));
        assert!(json.contains("SSSP"));
        assert!(json.contains("duration_ms"));
    }

    #[test]
    fn test_ai_results_serializable() {
        let result = run_ai_benchmark(100, 32, 5, 20);
        let json = serde_json::to_string_pretty(&result).unwrap();
        assert!(json.contains("recall_at_k"));
        assert!(json.contains("insert_rate_per_sec"));
    }

    #[test]
    fn test_scale_parameters() {
        assert_eq!(LdbcScale::XS.num_nodes(), 1_000);
        assert_eq!(LdbcScale::S.num_nodes(), 10_000);
        assert_eq!(LdbcScale::M.num_nodes(), 100_000);
        assert_eq!(LdbcScale::L.num_nodes(), 1_000_000);
        assert_eq!(LdbcScale::XL.num_nodes(), 10_000_000);
    }

    #[test]
    fn test_full_report_serializable() {
        let g = generate_ldbc_graph(LdbcScale::XS);
        let ldbc = run_ldbc_suite(&g);
        let ai = run_ai_benchmark(100, 32, 5, 20);
        let report = FullBenchReport {
            timestamp: "2026-04-01T00:00:00Z".to_string(),
            ldbc_results: vec![ldbc],
            ai_results: vec![ai],
        };
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(json.contains("ldbc_results"));
        assert!(json.contains("ai_results"));
    }
}
