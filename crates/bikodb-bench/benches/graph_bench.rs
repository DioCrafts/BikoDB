// =============================================================================
// BikoDB Benchmarks — Criterion suite
// =============================================================================
// Run with: cargo bench -p bikodb-bench
// =============================================================================

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use bikodb_bench::generators;
use bikodb_core::record::Direction;
use bikodb_core::types::{NodeId, TypeId};
use bikodb_core::value::Value;
use bikodb_graph::{ConcurrentGraph, traversal};
use bikodb_graph::bulk::{BulkLoader, BulkConfig};
use bikodb_graph::csr::CsrGraph;
use bikodb_graph::parallel_bfs;
use bikodb_graph::transaction::{TxManager, EdgeEndpoint};
use bikodb_ai::embedding::DistanceMetric;
use bikodb_ai::vector_idx::VectorIndex;
use std::sync::Arc;

/// Benchmark: Insert N nodos.
fn bench_node_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_insertion");
    for size in [1_000, 10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let g = ConcurrentGraph::with_capacity(size, 0);
                    for _ in 0..size {
                        g.insert_node(black_box(TypeId(1)));
                    }
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: BFS traversal en grafo social.
fn bench_bfs_traversal(c: &mut Criterion) {
    let graph = generators::generate_social_graph(10_000, 10);
    let start = NodeId(1);

    c.bench_function("bfs_10k_nodes", |b| {
        b.iter(|| {
            traversal::bfs(black_box(&graph), black_box(start), Direction::Out, Some(5))
                .unwrap()
        });
    });
}

/// Benchmark: DFS traversal en grafo social.
fn bench_dfs_traversal(c: &mut Criterion) {
    let graph = generators::generate_social_graph(10_000, 10);
    let start = NodeId(1);

    c.bench_function("dfs_10k_nodes", |b| {
        b.iter(|| {
            traversal::dfs(black_box(&graph), black_box(start), Direction::Out, Some(5))
                .unwrap()
        });
    });
}

/// Benchmark: Vector k-NN search.
fn bench_vector_search(c: &mut Criterion) {
    let dim = 128;
    let vectors = generators::generate_random_vectors(1_000, dim);
    let idx = VectorIndex::new(dim, DistanceMetric::Cosine);

    for (i, v) in vectors.iter().enumerate() {
        idx.insert(NodeId(i as u64 + 1), v.clone());
    }

    let query: Vec<f32> = (0..dim).map(|i| (i as f32) / dim as f32).collect();

    c.bench_function("vector_knn_1k_dim128", |b| {
        b.iter(|| {
            idx.search(black_box(&query), black_box(10))
        });
    });
}

/// Benchmark: Shortest path.
fn bench_shortest_path(c: &mut Criterion) {
    let graph = generators::generate_social_graph(10_000, 10);

    c.bench_function("shortest_path_10k", |b| {
        b.iter(|| {
            traversal::shortest_path(
                black_box(&graph),
                black_box(NodeId(1)),
                black_box(NodeId(500)),
                Direction::Out,
            )
            .unwrap()
        });
    });
}

/// Benchmark: Parallel BFS sobre CSR (100K nodos).
fn bench_parallel_bfs(c: &mut Criterion) {
    let graph = generators::generate_social_graph(100_000, 10);
    let csr = CsrGraph::from_concurrent(&graph, Direction::Out);
    let start = NodeId(1);

    let mut group = c.benchmark_group("bfs_100k");
    group.bench_function("parallel_bfs", |b| {
        b.iter(|| {
            parallel_bfs::parallel_bfs(black_box(&csr), black_box(start), None)
                .unwrap()
        });
    });
    group.bench_function("sequential_bfs_csr", |b| {
        b.iter(|| {
            parallel_bfs::sequential_bfs(black_box(&csr), black_box(start), None)
                .unwrap()
        });
    });
    group.bench_function("bfs_hashmap_original", |b| {
        b.iter(|| {
            traversal::bfs(black_box(&graph), black_box(start), Direction::Out, None)
                .unwrap()
        });
    });
    group.finish();
}

// =============================================================================
// Bulk Insertion Benchmarks
// =============================================================================

/// Benchmark: Batch node insertion (single-threaded BulkLoader).
fn bench_bulk_node_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_node_insertion");
    for size in [10_000, 100_000, 1_000_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let g = ConcurrentGraph::with_capacity(size, 0);
                    let config = BulkConfig {
                        flush_threshold: 50_000,
                        wal_enabled: false,
                    };
                    let mut loader = BulkLoader::new(&g, config);
                    for _ in 0..size {
                        loader.add_node_typed(black_box(TypeId(1)));
                    }
                    loader.finish().unwrap()
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: Direct batch API (insert_nodes_batch_typed).
fn bench_batch_api_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_api_insertion");
    for size in [10_000, 100_000, 1_000_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let g = ConcurrentGraph::with_capacity(size, 0);
                    g.insert_nodes_batch_typed(black_box(TypeId(1)), size);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: Concurrent multi-thread insertion (4 threads).
fn bench_concurrent_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_insertion_4t");
    for size in [100_000, 1_000_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                let per_thread = size / 4;
                b.iter(|| {
                    let g = Arc::new(ConcurrentGraph::with_capacity(size, 0));
                    std::thread::scope(|s| {
                        for _ in 0..4 {
                            let gr = Arc::clone(&g);
                            s.spawn(move || {
                                gr.insert_nodes_batch_typed(black_box(TypeId(1)), per_thread);
                            });
                        }
                    });
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: Batch edge insertion.
fn bench_batch_edge_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_edge_insertion");
    for size in [10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                b.iter_with_setup(
                    || {
                        let g = ConcurrentGraph::with_capacity(size, size);
                        let ids = g.insert_nodes_batch_typed(TypeId(1), size);
                        let edges: Vec<(NodeId, NodeId, TypeId)> = (0..size - 1)
                            .map(|i| (ids[i], ids[i + 1], TypeId(10)))
                            .collect();
                        (g, edges)
                    },
                    |(g, edges)| {
                        g.insert_edges_batch(black_box(&edges));
                    },
                );
            },
        );
    }
    group.finish();
}

// =============================================================================
// Transactional Insertion Benchmarks
// =============================================================================

/// Benchmark: Transactional node insertion (1 tx per node).
fn bench_tx_single_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("tx_single_insert");
    for size in [1_000, 10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let g = ConcurrentGraph::with_capacity(size, 0);
                    let mgr = TxManager::in_memory();
                    for _ in 0..size {
                        let mut tx = mgr.begin(&g);
                        tx.insert_node(black_box(TypeId(1)), vec![]);
                        mgr.commit(&mut tx).unwrap();
                    }
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: Transactional batch insertion (1 tx for N nodes).
fn bench_tx_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("tx_batch_insert");
    for size in [10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let g = ConcurrentGraph::with_capacity(size, 0);
                    let mgr = TxManager::in_memory();
                    let mut tx = mgr.begin_with_capacity(&g, size);
                    for _ in 0..size {
                        tx.insert_node(black_box(TypeId(1)), vec![]);
                    }
                    mgr.commit(&mut tx).unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: Concurrent transactional insertion (4 threads).
fn bench_tx_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("tx_concurrent_4t");
    for size in [10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                let per_thread = size / 4;
                let batch_size = 100;
                b.iter(|| {
                    let g = ConcurrentGraph::with_capacity(size, 0);
                    let mgr = TxManager::in_memory();
                    std::thread::scope(|s| {
                        for _ in 0..4 {
                            let mgr_ref = &mgr;
                            let g_ref = &g;
                            s.spawn(move || {
                                let batches = per_thread / batch_size;
                                for _ in 0..batches {
                                    let mut tx = mgr_ref.begin_with_capacity(g_ref, batch_size);
                                    for _ in 0..batch_size {
                                        tx.insert_node(black_box(TypeId(1)), vec![]);
                                    }
                                    mgr_ref.commit(&mut tx).unwrap();
                                }
                            });
                        }
                    });
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: Transactional insert nodes + edges.
fn bench_tx_nodes_and_edges(c: &mut Criterion) {
    let mut group = c.benchmark_group("tx_nodes_edges");
    for size in [1_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let g = ConcurrentGraph::with_capacity(size, size);
                    let mgr = TxManager::in_memory();
                    let mut tx = mgr.begin_with_capacity(&g, size * 2);
                    let mut prev: Option<usize> = None;
                    for _ in 0..size {
                        let idx = tx.insert_node(black_box(TypeId(1)), vec![]);
                        if let Some(p) = prev {
                            tx.insert_edge(
                                EdgeEndpoint::Pending(p),
                                EdgeEndpoint::Pending(idx),
                                black_box(TypeId(10)),
                            );
                        }
                        prev = Some(idx);
                    }
                    mgr.commit(&mut tx).unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_node_insertion,
    bench_bfs_traversal,
    bench_dfs_traversal,
    bench_vector_search,
    bench_shortest_path,
    bench_parallel_bfs,
    bench_bulk_node_insertion,
    bench_batch_api_insertion,
    bench_concurrent_insertion,
    bench_batch_edge_insertion,
    bench_tx_single_insert,
    bench_tx_batch_insert,
    bench_tx_concurrent,
    bench_tx_nodes_and_edges,
);
criterion_main!(benches);
