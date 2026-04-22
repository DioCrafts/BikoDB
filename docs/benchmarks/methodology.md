# Benchmark Methodology

This page documents what the current repository can support honestly.

## What the current report says

The repository includes `BENCHMARK_COMPARISON.md`, which compares BikoDB against ArcadeDB, Kuzu, and Neo4j.

The current report explicitly states:

- BikoDB values are benchmarked directly,
- competitor values are reference values from published material or internal reference points,
- competitor values are treated as approximate,
- some values are scaled linearly across graph sizes.

That means the current comparison should be read as **directional**, not as a final apples-to-apples audited benchmark suite.

## Reproduce BikoDB measurements

From the repository root:

```bash
cargo bench -p bikodb-bench
cargo run -p bikodb-bench --release --bin comparison_report
```

These commands are the current starting point for local reproduction.

## Current assumptions visible in the repo

The published comparison report currently describes:

- power-law graph workloads,
- example scales such as 10K and 100K nodes,
- average degree around 10,
- CSR-based graph execution for BikoDB.

## What is still missing for a fully defensible external benchmark story

A stricter public benchmark methodology should eventually include all of the following in one place:

- exact hardware model,
- CPU and RAM configuration,
- OS and kernel version,
- compiler/runtime versions,
- warmup rules,
- number of repetitions,
- exact dataset generator settings,
- competitor configuration flags,
- explicit notes on which competitor features/plugins were or were not used,
- raw result files or scripts for reruns.

## How to present the current results responsibly

Today the safest public framing is:

- BikoDB has promising measured graph performance,
- the repo includes reproducible BikoDB benchmark commands,
- the cross-database comparison is informative but not yet a fully audited neutral benchmark harness.

## Practical guidance for external readers

If you are evaluating BikoDB seriously:

1. reproduce the BikoDB numbers locally,
2. treat competitor numbers in the current markdown report as reference context,
3. avoid making procurement or production decisions from the comparison report alone.
