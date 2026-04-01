// =============================================================================
// comparison_report — Genera BENCHMARK_COMPARISON.md
// =============================================================================
//
// Ejecuta la suite completa de benchmarks comparativos BikoDB vs ArcadeDB vs Kuzu
// y genera un archivo Markdown con tablas de features, rendimiento y análisis.
//
// ## Uso
// ```sh
// cargo run -p bikodb-bench --release --bin comparison_report
// cargo run -p bikodb-bench --release --bin comparison_report -- --scales 10000,100000
// cargo run -p bikodb-bench --release --bin comparison_report -- --output my_report.md
// ```
// =============================================================================

use bikodb_bench::comparison::generate_markdown_report;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut output = PathBuf::from("BENCHMARK_COMPARISON.md");
    let mut scales: Vec<(usize, usize)> = vec![(10_000, 10), (100_000, 10)];

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output" | "-o" => {
                i += 1;
                if i < args.len() {
                    output = PathBuf::from(&args[i]);
                }
            }
            "--scales" | "-s" => {
                i += 1;
                if i < args.len() {
                    scales = args[i]
                        .split(',')
                        .filter_map(|s| s.trim().parse::<usize>().ok())
                        .map(|n| (n, 10))
                        .collect();
                }
            }
            "--help" | "-h" => {
                eprintln!("Usage: comparison_report [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  -o, --output <FILE>    Output file [default: BENCHMARK_COMPARISON.md]");
                eprintln!("  -s, --scales <N,N,...>  Comma-separated node counts [default: 10000,100000]");
                eprintln!("  -h, --help             Show this help");
                return;
            }
            _ => {
                eprintln!("Unknown argument: {}. Use --help for usage.", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  BikoDB Benchmark Comparison Report Generator               ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();

    for (n, d) in &scales {
        let label = if *n >= 1_000_000 {
            format!("{}M nodes", n / 1_000_000)
        } else {
            format!("{}K nodes", n / 1000)
        };
        eprintln!("  → Scale: {}, avg degree {}", label, d);
    }
    eprintln!();

    eprintln!("Running benchmarks...");
    let md = generate_markdown_report(&scales);

    eprintln!("Writing report to: {}", output.display());
    if let Err(e) = std::fs::write(&output, &md) {
        eprintln!("Error writing file: {}", e);
        std::process::exit(1);
    }

    // Count some stats for summary
    let lines = md.lines().count();
    let tables = md.matches("|---").count();
    eprintln!();
    eprintln!("✓ Report generated: {} lines, {} tables", lines, tables);
    eprintln!("✓ Saved to: {}", output.display());
}
