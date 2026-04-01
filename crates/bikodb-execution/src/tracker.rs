// =============================================================================
// bikodb-execution::tracker — Access pattern tracker
// =============================================================================
// Collects runtime statistics about query execution:
//   - Query frequency histogram
//   - Per-operator runtime metrics (rows produced, execution time)
//   - Hot-path detection for automatic index suggestions
// =============================================================================

use std::collections::HashMap;
use std::time::{Duration, Instant};

use bikodb_core::types::TypeId;

/// Records per-operator execution metrics.
#[derive(Debug, Clone, Default)]
pub struct OperatorMetrics {
    /// Number of times this operator has been invoked.
    pub invocations: u64,
    /// Total rows produced across all invocations.
    pub rows_produced: u64,
    /// Cumulative execution time.
    pub total_time: Duration,
}

impl OperatorMetrics {
    /// Average rows produced per invocation.
    pub fn avg_rows(&self) -> f64 {
        if self.invocations == 0 {
            0.0
        } else {
            self.rows_produced as f64 / self.invocations as f64
        }
    }

    /// Average execution time per invocation.
    pub fn avg_time(&self) -> Duration {
        if self.invocations == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.invocations as u32
        }
    }
}

/// Tracks query execution patterns and suggests optimizations.
#[derive(Debug, Default)]
pub struct AccessTracker {
    /// Query frequency: query_hash → invocation count.
    query_freq: HashMap<u64, u64>,
    /// Per-operator metrics keyed by operator label (e.g., "NodeScan(1)").
    operator_metrics: HashMap<String, OperatorMetrics>,
    /// Tracks (type_id, property_id) pairs used in Filter predicates.
    filter_patterns: HashMap<(TypeId, u16), u64>,
}

impl AccessTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a query execution. Call once per query with its hash.
    pub fn record_query(&mut self, query_hash: u64) {
        *self.query_freq.entry(query_hash).or_insert(0) += 1;
    }

    /// Records a filter access pattern (type + property).
    pub fn record_filter(&mut self, type_id: TypeId, property_id: u16) {
        *self
            .filter_patterns
            .entry((type_id, property_id))
            .or_insert(0) += 1;
    }

    /// Records operator execution metrics.
    pub fn record_operator(
        &mut self,
        label: &str,
        rows_produced: u64,
        elapsed: Duration,
    ) {
        let m = self
            .operator_metrics
            .entry(label.to_string())
            .or_default();
        m.invocations += 1;
        m.rows_produced += rows_produced;
        m.total_time += elapsed;
    }

    /// Returns the top-N most frequently queried hashes.
    pub fn top_queries(&self, n: usize) -> Vec<(u64, u64)> {
        let mut freq: Vec<_> = self.query_freq.iter().map(|(&k, &v)| (k, v)).collect();
        freq.sort_by(|a, b| b.1.cmp(&a.1));
        freq.truncate(n);
        freq
    }

    /// Suggests indexes based on frequently-filtered (type, property) pairs.
    ///
    /// Returns pairs that have been filtered more than `threshold` times.
    pub fn suggest_indexes(&self, threshold: u64) -> Vec<(TypeId, u16)> {
        self.filter_patterns
            .iter()
            .filter(|(_, &count)| count >= threshold)
            .map(|(&key, _)| key)
            .collect()
    }

    /// Returns metrics for a specific operator label.
    pub fn get_operator_metrics(&self, label: &str) -> Option<&OperatorMetrics> {
        self.operator_metrics.get(label)
    }

    /// Returns the number of distinct queries seen.
    pub fn distinct_queries(&self) -> usize {
        self.query_freq.len()
    }

    /// Returns the total number of query executions.
    pub fn total_executions(&self) -> u64 {
        self.query_freq.values().sum()
    }
}

/// RAII-style timer for measuring operator execution time.
pub struct OpTimer {
    label: String,
    start: Instant,
}

impl OpTimer {
    pub fn start(label: &str) -> Self {
        Self {
            label: label.to_string(),
            start: Instant::now(),
        }
    }

    /// Stops the timer and records metrics into the tracker.
    pub fn stop(self, tracker: &mut AccessTracker, rows_produced: u64) {
        let elapsed = self.start.elapsed();
        tracker.record_operator(&self.label, rows_produced, elapsed);
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_top_queries() {
        let mut tracker = AccessTracker::new();
        tracker.record_query(100);
        tracker.record_query(100);
        tracker.record_query(100);
        tracker.record_query(200);
        tracker.record_query(300);
        tracker.record_query(300);

        let top = tracker.top_queries(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0], (100, 3)); // most frequent
        assert_eq!(top[1], (300, 2)); // second
    }

    #[test]
    fn test_suggest_indexes() {
        let mut tracker = AccessTracker::new();
        // Filter on Person.name 10 times
        for _ in 0..10 {
            tracker.record_filter(TypeId(1), 0);
        }
        // Filter on Person.age 3 times
        for _ in 0..3 {
            tracker.record_filter(TypeId(1), 1);
        }

        let suggestions = tracker.suggest_indexes(5);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0], (TypeId(1), 0));
    }

    #[test]
    fn test_operator_metrics() {
        let mut tracker = AccessTracker::new();
        tracker.record_operator("NodeScan(1)", 1000, Duration::from_millis(50));
        tracker.record_operator("NodeScan(1)", 1000, Duration::from_millis(60));

        let m = tracker.get_operator_metrics("NodeScan(1)").unwrap();
        assert_eq!(m.invocations, 2);
        assert_eq!(m.rows_produced, 2000);
        assert!((m.avg_rows() - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_total_executions() {
        let mut tracker = AccessTracker::new();
        tracker.record_query(1);
        tracker.record_query(2);
        tracker.record_query(1);

        assert_eq!(tracker.total_executions(), 3);
        assert_eq!(tracker.distinct_queries(), 2);
    }

    #[test]
    fn test_op_timer() {
        let mut tracker = AccessTracker::new();
        let timer = OpTimer::start("FilterOp");
        // Simulate some work
        std::thread::sleep(Duration::from_millis(1));
        timer.stop(&mut tracker, 42);

        let m = tracker.get_operator_metrics("FilterOp").unwrap();
        assert_eq!(m.invocations, 1);
        assert_eq!(m.rows_produced, 42);
        assert!(m.total_time >= Duration::from_millis(1));
    }
}
