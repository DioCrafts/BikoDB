# Benchmarks

BikoDB already ships benchmark code and a generated comparison report, but the right way to read those results is with explicit caveats.

## What exists today

- a benchmark crate in `crates/bikodb-bench`,
- a generated `BENCHMARK_COMPARISON.md` report in the repo root,
- measured BikoDB timings for graph workloads,
- reference competitor values included in the comparison report.

## Why methodology matters

Benchmark claims are often the first thing external readers scrutinise.

For BikoDB, the serious posture is:

- explain which numbers are directly measured,
- explain which competitor numbers are reference values,
- explain the graph shape and scaling assumptions,
- give exact reproduction commands for BikoDB runs.

## Read next

- [Benchmark methodology →](/benchmarks/methodology)
