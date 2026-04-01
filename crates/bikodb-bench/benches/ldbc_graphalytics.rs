// =============================================================================
// LDBC Graphalytics Benchmark Suite
// =============================================================================
//
// Implementa benchmarks Criterion para los 6 algoritmos core de LDBC Graphalytics:
// BFS, SSSP, PageRank, CDLP (Label Propagation), WCC, LCC
//
// Más benchmarks de AI/ML: HNSW insert, HNSW k-NN search, recall estimation.
//
// Escalas: XS (1K), S (10K), M (100K) para CI; L/XL para benchmarks manuales.
// =============================================================================

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use bikodb_bench::ldbc::{generate_ldbc_graph, LdbcScale};
use bikodb_core::record::Direction;
use bikodb_core::types::NodeId;
use bikodb_graph::community::{LpaConfig, connected_components_on_csr, label_propagation_on_csr};
use bikodb_graph::csr::CsrGraph;
use bikodb_graph::lcc::lcc_on_csr_undirected;
use bikodb_graph::pagerank::{PageRankConfig, pagerank_on_csr};
use bikodb_graph::parallel_bfs::parallel_bfs;
use bikodb_graph::sssp::sssp;
use bikodb_graph::weighted_csr::WeightedCsrGraph;

// =============================================================================
// LDBC Graphalytics — 6 Core Algorithms
// =============================================================================

fn bench_bfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("ldbc/BFS");
    for scale in &[LdbcScale::XS, LdbcScale::S] {
        let lg = generate_ldbc_graph(*scale);
        let csr = CsrGraph::from_concurrent(&lg.graph, Direction::Out);
        let start = NodeId(1);
        group.bench_with_input(
            BenchmarkId::new("parallel_bfs", scale.name()),
            &(csr, start),
            |b, (csr, start)| {
                b.iter(|| parallel_bfs(csr, *start, None));
            },
        );
    }
    group.finish();
}

fn bench_sssp(c: &mut Criterion) {
    let mut group = c.benchmark_group("ldbc/SSSP");
    for scale in &[LdbcScale::XS, LdbcScale::S] {
        let lg = generate_ldbc_graph(*scale);
        let wcsr = WeightedCsrGraph::from_concurrent(
            &lg.graph, Direction::Out, lg.weight_prop_id, 1.0,
        );
        let start = NodeId(1);
        group.bench_with_input(
            BenchmarkId::new("adaptive", scale.name()),
            &(wcsr, start),
            |b, (wcsr, start)| {
                b.iter(|| sssp(wcsr, *start));
            },
        );
    }
    group.finish();
}

fn bench_pagerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("ldbc/PageRank");
    let config = PageRankConfig {
        damping: 0.85,
        max_iterations: 20,
        tolerance: 1e-6,
    };
    for scale in &[LdbcScale::XS, LdbcScale::S] {
        let lg = generate_ldbc_graph(*scale);
        let out_csr = CsrGraph::from_concurrent(&lg.graph, Direction::Out);
        let in_csr = CsrGraph::from_concurrent(&lg.graph, Direction::In);
        group.bench_with_input(
            BenchmarkId::new("pull_csr", scale.name()),
            &(out_csr, in_csr, config.clone()),
            |b, (out_csr, in_csr, cfg)| {
                b.iter(|| pagerank_on_csr(out_csr, in_csr, cfg));
            },
        );
    }
    group.finish();
}

fn bench_cdlp(c: &mut Criterion) {
    let mut group = c.benchmark_group("ldbc/CDLP");
    let config = LpaConfig { max_iterations: 10 };
    for scale in &[LdbcScale::XS, LdbcScale::S] {
        let lg = generate_ldbc_graph(*scale);
        let out_csr = CsrGraph::from_concurrent(&lg.graph, Direction::Out);
        let in_csr = CsrGraph::from_concurrent(&lg.graph, Direction::In);
        group.bench_with_input(
            BenchmarkId::new("label_propagation", scale.name()),
            &(out_csr, in_csr, config.clone()),
            |b, (out_csr, in_csr, cfg)| {
                b.iter(|| label_propagation_on_csr(out_csr, Some(in_csr), cfg));
            },
        );
    }
    group.finish();
}

fn bench_wcc(c: &mut Criterion) {
    let mut group = c.benchmark_group("ldbc/WCC");
    for scale in &[LdbcScale::XS, LdbcScale::S] {
        let lg = generate_ldbc_graph(*scale);
        let out_csr = CsrGraph::from_concurrent(&lg.graph, Direction::Out);
        let in_csr = CsrGraph::from_concurrent(&lg.graph, Direction::In);
        group.bench_with_input(
            BenchmarkId::new("connected_components", scale.name()),
            &(out_csr, in_csr),
            |b, (out_csr, in_csr)| {
                b.iter(|| connected_components_on_csr(out_csr, Some(in_csr)));
            },
        );
    }
    group.finish();
}

fn bench_lcc(c: &mut Criterion) {
    let mut group = c.benchmark_group("ldbc/LCC");
    for scale in &[LdbcScale::XS, LdbcScale::S] {
        let lg = generate_ldbc_graph(*scale);
        let out_csr = CsrGraph::from_concurrent(&lg.graph, Direction::Out);
        let in_csr = CsrGraph::from_concurrent(&lg.graph, Direction::In);
        group.bench_with_input(
            BenchmarkId::new("local_clustering_coeff", scale.name()),
            &(out_csr, in_csr),
            |b, (out_csr, in_csr)| {
                b.iter(|| lcc_on_csr_undirected(out_csr, in_csr));
            },
        );
    }
    group.finish();
}

// =============================================================================
// AI/ML Benchmarks — HNSW Vector Search
// =============================================================================

fn bench_hnsw_insert(c: &mut Criterion) {
    use bikodb_ai::embedding::DistanceMetric;
    use bikodb_ai::hnsw::{HnswConfig, HnswIndex};
    use rand::Rng;

    let mut group = c.benchmark_group("ai/hnsw_insert");
    for &(count, dims) in &[(1_000usize, 64usize), (5_000, 128)] {
        group.bench_with_input(
            BenchmarkId::new(format!("{}d", dims), count),
            &(count, dims),
            |b, &(count, dims)| {
                b.iter(|| {
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
                });
            },
        );
    }
    group.finish();
}

fn bench_hnsw_search(c: &mut Criterion) {
    use bikodb_ai::embedding::DistanceMetric;
    use bikodb_ai::hnsw::{HnswConfig, HnswIndex};
    use rand::Rng;

    let mut group = c.benchmark_group("ai/hnsw_search");
    for &(count, dims) in &[(5_000usize, 64usize), (10_000, 128)] {
        // Build index outside measurement
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

        for &k in &[10usize, 50] {
            group.bench_with_input(
                BenchmarkId::new(format!("{}d_top{}", dims, k), count),
                &(k),
                |b, &k| {
                    b.iter(|| hnsw.search(&query, k, 100));
                },
            );
        }
    }
    group.finish();
}

// =============================================================================
// Full Suite Report — JSON output  (run via `cargo bench -- --test` for quick verify)
// =============================================================================

fn bench_full_ldbc_suite(c: &mut Criterion) {
    use bikodb_bench::ldbc::run_ldbc_suite;

    let mut group = c.benchmark_group("ldbc/full_suite");
    let lg = generate_ldbc_graph(LdbcScale::XS);
    group.bench_function("XS_all_6_algos", |b| {
        b.iter(|| run_ldbc_suite(&lg));
    });
    group.finish();
}

// =============================================================================
// Criterion Groups & Main
// =============================================================================

criterion_group!(
    ldbc_benches,
    bench_bfs,
    bench_sssp,
    bench_pagerank,
    bench_cdlp,
    bench_wcc,
    bench_lcc,
    bench_full_ldbc_suite,
);

criterion_group!(
    ai_benches,
    bench_hnsw_insert,
    bench_hnsw_search,
);

criterion_main!(ldbc_benches, ai_benches);
