// =============================================================================
// Comparison Benchmark Suite — BikoDB vs ArcadeDB vs Kuzu
// =============================================================================
//
// Benchmarks Criterion que miden BikoDB en las mismas dimensiones que ArcadeDB
// y Kuzu benchmark en sus propios repos. Los resultados de competidores son
// valores de referencia; aquí se mide BikoDB directamente.
//
// ## Categorías
// 1. Graph Algorithm Performance (9 algos × 2 scales)
// 2. Insertion Throughput (single, bulk, concurrent)
// 3. HNSW Vector Search (insert + query)
// 4. Full algorithm suite (all 9 at once)
//
// ## Ejecutar
// ```sh
// cargo bench -p bikodb-bench --bench comparison
// cargo bench -p bikodb-bench --bench comparison -- --test  # quick verify
// ```
// =============================================================================

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use bikodb_bench::comparison::{feature_scores, run_perf_suite};
use bikodb_core::record::Direction;
use bikodb_core::types::{NodeId, TypeId};
use bikodb_core::value::Value;
use bikodb_graph::community::{LpaConfig, connected_components_on_csr, label_propagation_on_csr};
use bikodb_graph::csr::CsrGraph;
use bikodb_graph::kcore::kcore_on_csr_undirected;
use bikodb_graph::lcc::lcc_on_csr_undirected;
use bikodb_graph::louvain::{LouvainConfig, louvain_on_csr};
use bikodb_graph::pagerank::{PageRankConfig, pagerank_on_csr};
use bikodb_graph::parallel_bfs::{parallel_bfs, direction_optimizing_bfs, DoBfsConfig};
use bikodb_graph::scc::tarjan_scc_on_csr;
use bikodb_graph::sssp::sssp;
use bikodb_graph::weighted_csr::WeightedCsrGraph;
use bikodb_graph::ConcurrentGraph;
use rand::Rng;

// =============================================================================
// Graph generators
// =============================================================================

fn build_graph(num_nodes: usize, avg_degree: usize) -> ConcurrentGraph {
    let graph = ConcurrentGraph::with_capacity(num_nodes, num_nodes * avg_degree * 2);
    let mut rng = rand::thread_rng();
    let node_type = TypeId(1);
    let edge_type = TypeId(10);

    let nodes: Vec<NodeId> = (0..num_nodes)
        .map(|_| graph.insert_node(node_type))
        .collect();

    for &src in &nodes {
        let degree = if rng.gen_range(0..100) < 5 {
            avg_degree * 5
        } else {
            rng.gen_range(1..=avg_degree * 2)
        };
        for _ in 0..degree {
            let tgt_idx = rng.gen_range(0..num_nodes);
            let tgt = nodes[tgt_idx];
            if src != tgt {
                let w = rng.gen_range(0.1_f64..10.0);
                let _ = graph.insert_edge_with_props(
                    src, tgt, edge_type, vec![(0, Value::Float(w))],
                );
                let _ = graph.insert_edge_with_props(
                    tgt, src, edge_type, vec![(0, Value::Float(w))],
                );
            }
        }
    }
    graph
}

struct PreparedGraph {
    graph: ConcurrentGraph,
    out_csr: CsrGraph,
    in_csr: CsrGraph,
    wcsr: WeightedCsrGraph,
    start: NodeId,
    num_nodes: usize,
}

fn prepare(num_nodes: usize, avg_degree: usize) -> PreparedGraph {
    let graph = build_graph(num_nodes, avg_degree);
    let out_csr = CsrGraph::from_concurrent(&graph, Direction::Out);
    let in_csr = CsrGraph::from_concurrent(&graph, Direction::In);
    let wcsr = WeightedCsrGraph::from_concurrent(&graph, Direction::Out, 0, 1.0);
    PreparedGraph {
        graph,
        out_csr,
        in_csr,
        wcsr,
        start: NodeId(1),
        num_nodes,
    }
}

// =============================================================================
// 1. Graph Algorithm Benchmarks — equivalent to ArcadeDB GraphOLAPBenchmark
//    and Kuzu Graph500/soc-livejournal benchmarks
// =============================================================================

fn bench_algo_bfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmp/algo/BFS");
    for &n in &[10_000usize, 100_000] {
        let pg = prepare(n, 10);
        group.throughput(Throughput::Elements(pg.out_csr.num_edges() as u64));
        group.bench_with_input(
            BenchmarkId::new("bikodb_push", format!("{}K", n / 1000)),
            &pg,
            |b, pg| b.iter(|| parallel_bfs(&pg.out_csr, pg.start, None)),
        );
    }
    group.finish();
}

fn bench_algo_do_bfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmp/algo/DO_BFS");
    let config = DoBfsConfig::default();
    for &n in &[10_000usize, 100_000] {
        let pg = prepare(n, 10);
        group.throughput(Throughput::Elements(pg.out_csr.num_edges() as u64));
        group.bench_with_input(
            BenchmarkId::new("bikodb_direction_optimizing", format!("{}K", n / 1000)),
            &pg,
            |b, pg| {
                b.iter(|| {
                    direction_optimizing_bfs(&pg.out_csr, &pg.in_csr, pg.start, &config)
                })
            },
        );
    }
    group.finish();
}

fn bench_algo_pagerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmp/algo/PageRank");
    let config = PageRankConfig {
        damping: 0.85,
        max_iterations: 20,
        tolerance: 1e-6,
    };
    for &n in &[10_000usize, 100_000] {
        let pg = prepare(n, 10);
        group.throughput(Throughput::Elements(pg.out_csr.num_edges() as u64 * 20));
        group.bench_with_input(
            BenchmarkId::new("bikodb", format!("{}K", n / 1000)),
            &pg,
            |b, pg| b.iter(|| pagerank_on_csr(&pg.out_csr, &pg.in_csr, &config)),
        );
    }
    group.finish();
}

fn bench_algo_sssp(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmp/algo/SSSP");
    for &n in &[10_000usize, 100_000] {
        let pg = prepare(n, 10);
        group.bench_with_input(
            BenchmarkId::new("bikodb", format!("{}K", n / 1000)),
            &pg,
            |b, pg| b.iter(|| sssp(&pg.wcsr, pg.start)),
        );
    }
    group.finish();
}

fn bench_algo_wcc(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmp/algo/WCC");
    for &n in &[10_000usize, 100_000] {
        let pg = prepare(n, 10);
        group.throughput(Throughput::Elements(pg.out_csr.num_edges() as u64));
        group.bench_with_input(
            BenchmarkId::new("bikodb", format!("{}K", n / 1000)),
            &pg,
            |b, pg| b.iter(|| connected_components_on_csr(&pg.out_csr, Some(&pg.in_csr))),
        );
    }
    group.finish();
}

fn bench_algo_cdlp(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmp/algo/CDLP");
    let config = LpaConfig { max_iterations: 10 };
    for &n in &[10_000usize, 100_000] {
        let pg = prepare(n, 10);
        group.bench_with_input(
            BenchmarkId::new("bikodb", format!("{}K", n / 1000)),
            &pg,
            |b, pg| {
                b.iter(|| label_propagation_on_csr(&pg.out_csr, Some(&pg.in_csr), &config))
            },
        );
    }
    group.finish();
}

fn bench_algo_lcc(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmp/algo/LCC");
    for &n in &[10_000usize, 100_000] {
        let pg = prepare(n, 10);
        group.bench_with_input(
            BenchmarkId::new("bikodb", format!("{}K", n / 1000)),
            &pg,
            |b, pg| b.iter(|| lcc_on_csr_undirected(&pg.out_csr, &pg.in_csr)),
        );
    }
    group.finish();
}

fn bench_algo_scc(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmp/algo/SCC");
    for &n in &[10_000usize, 100_000] {
        let pg = prepare(n, 10);
        group.throughput(Throughput::Elements(pg.out_csr.num_edges() as u64));
        group.bench_with_input(
            BenchmarkId::new("bikodb_tarjan", format!("{}K", n / 1000)),
            &pg,
            |b, pg| b.iter(|| tarjan_scc_on_csr(&pg.out_csr)),
        );
    }
    group.finish();
}

fn bench_algo_louvain(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmp/algo/Louvain");
    let config = LouvainConfig::default();
    for &n in &[10_000usize] {
        let pg = prepare(n, 10);
        group.bench_with_input(
            BenchmarkId::new("bikodb", format!("{}K", n / 1000)),
            &pg,
            |b, pg| b.iter(|| louvain_on_csr(&pg.out_csr, &pg.in_csr, &config)),
        );
    }
    group.finish();
}

fn bench_algo_kcore(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmp/algo/Kcore");
    for &n in &[10_000usize, 100_000] {
        let pg = prepare(n, 10);
        group.bench_with_input(
            BenchmarkId::new("bikodb", format!("{}K", n / 1000)),
            &pg,
            |b, pg| b.iter(|| kcore_on_csr_undirected(&pg.out_csr, &pg.in_csr)),
        );
    }
    group.finish();
}

// =============================================================================
// 2. Insertion Throughput — equivalent to ArcadeDB LocalDatabaseBenchmark
// =============================================================================

fn bench_insert_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmp/insert/single_thread");
    let count = 100_000usize;
    group.throughput(Throughput::Elements(count as u64));
    group.bench_function("bikodb_100K", |b| {
        b.iter(|| {
            let graph = ConcurrentGraph::with_capacity(count, 0);
            let t = TypeId(1);
            for _ in 0..count {
                graph.insert_node(t);
            }
        });
    });
    group.finish();
}

fn bench_insert_concurrent(c: &mut Criterion) {
    use std::sync::Arc;
    let mut group = c.benchmark_group("cmp/insert/concurrent");
    let count = 100_000usize;
    let threads = 4usize;
    group.throughput(Throughput::Elements(count as u64));
    group.bench_function("bikodb_100K_4threads", |b| {
        b.iter(|| {
            let graph = Arc::new(ConcurrentGraph::with_capacity(count, 0));
            let per_thread = count / threads;
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    let g = Arc::clone(&graph);
                    std::thread::spawn(move || {
                        let t = TypeId(1);
                        for _ in 0..per_thread {
                            g.insert_node(t);
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    group.finish();
}

// =============================================================================
// 3. HNSW Vector Search — equivalent to ArcadeDB LSMVectorIndexJMHBenchmark
// =============================================================================

fn bench_hnsw_comparison(c: &mut Criterion) {
    use bikodb_ai::embedding::DistanceMetric;
    use bikodb_ai::hnsw::{HnswConfig, HnswIndex};

    let mut group = c.benchmark_group("cmp/vector/hnsw");

    // Build index
    let dims = 128;
    let count = 10_000usize;
    let config = HnswConfig {
        dimensions: dims,
        metric: DistanceMetric::Cosine,
        max_connections: 16,
        max_connections_0: 32,
        ef_construction: 200,
    };
    let hnsw = HnswIndex::new(config);
    let mut rng = rand::thread_rng();
    for i in 0..count {
        let v: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        hnsw.insert(NodeId(i as u64 + 1), v);
    }
    let query: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

    for &k in &[10usize, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("bikodb_search", format!("top{}_10K_128d", k)),
            &k,
            |b, &k| b.iter(|| hnsw.search(&query, k, 100)),
        );
    }
    group.finish();
}

// =============================================================================
// 4. Full suite benchmark (all algorithms, single run)
// =============================================================================

fn bench_full_suite(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmp/full_suite");
    group.bench_function("bikodb_10K_all_12_operations", |b| {
        b.iter(|| run_perf_suite(10_000, 10));
    });
    group.finish();
}

// =============================================================================
// Criterion Groups & Main
// =============================================================================

criterion_group!(
    algo_benches,
    bench_algo_bfs,
    bench_algo_do_bfs,
    bench_algo_pagerank,
    bench_algo_sssp,
    bench_algo_wcc,
    bench_algo_cdlp,
    bench_algo_lcc,
    bench_algo_scc,
    bench_algo_louvain,
    bench_algo_kcore,
);

criterion_group!(
    insert_benches,
    bench_insert_single,
    bench_insert_concurrent,
);

criterion_group!(
    vector_benches,
    bench_hnsw_comparison,
);

criterion_group!(
    suite_benches,
    bench_full_suite,
);

criterion_main!(algo_benches, insert_benches, vector_benches, suite_benches);
