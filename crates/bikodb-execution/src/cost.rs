// =============================================================================
// bikodb-execution::cost — Cost model & statistics for optimization
// =============================================================================
// Provides cardinality estimates, cost calculations, and index metadata
// so the optimizer can make cost-based decisions (e.g., scan vs. index).
// =============================================================================

use std::collections::{HashMap, HashSet};

use bikodb_core::types::TypeId;

use crate::plan::LogicalOp;

/// Estimated cost of a physical execution plan.
#[derive(Debug, Clone, PartialEq)]
pub struct CostEstimate {
    /// Estimated output row count after this operator.
    pub rows: f64,
    /// CPU cost units (predicate evaluations, comparisons, etc.).
    pub cpu: f64,
    /// I/O cost units (pages read).
    pub io: f64,
}

impl CostEstimate {
    pub fn total(&self) -> f64 {
        self.cpu + self.io * 4.0 // IO is ~4x more expensive than CPU
    }
}

/// Per-type cardinality statistics.
#[derive(Debug, Clone)]
pub struct TypeStats {
    /// Number of rows (nodes/edges) of this type.
    pub row_count: u64,
    /// Average number of properties per row.
    pub avg_properties: f64,
}

/// Cost model holding index metadata and cardinality statistics.
///
/// The optimizer consults this to decide whether to use an index scan
/// and to estimate result set sizes.
#[derive(Debug, Clone)]
pub struct CostModel {
    /// (type_id, property_id) pairs that have an index.
    indexes: HashSet<(TypeId, u16)>,
    /// Per-type statistics.
    type_stats: HashMap<TypeId, TypeStats>,
    /// Default selectivity for equality predicates (1 / NDV).
    default_eq_selectivity: f64,
    /// Default selectivity for range predicates.
    default_range_selectivity: f64,
}

impl CostModel {
    pub fn new() -> Self {
        Self {
            indexes: HashSet::new(),
            type_stats: HashMap::new(),
            default_eq_selectivity: 0.01,   // 1%
            default_range_selectivity: 0.33, // 33%
        }
    }

    /// Registers an index on (type_id, property_id).
    pub fn register_index(&mut self, type_id: TypeId, property_id: u16) {
        self.indexes.insert((type_id, property_id));
    }

    /// Returns true if there's an index on (type_id, property_id).
    pub fn has_index(&self, type_id: TypeId, property_id: u16) -> bool {
        self.indexes.contains(&(type_id, property_id))
    }

    /// Records cardinality stats for a type.
    pub fn set_type_stats(&mut self, type_id: TypeId, stats: TypeStats) {
        self.type_stats.insert(type_id, stats);
    }

    /// Returns row count for a type, defaulting to 1000 if unknown.
    pub fn row_count(&self, type_id: TypeId) -> f64 {
        self.type_stats
            .get(&type_id)
            .map(|s| s.row_count as f64)
            .unwrap_or(1000.0)
    }

    /// Estimates cost for a logical plan.
    pub fn estimate(&self, plan: &LogicalOp) -> CostEstimate {
        match plan {
            LogicalOp::NodeScan { type_id } => {
                let rows = self.row_count(*type_id);
                CostEstimate {
                    rows,
                    cpu: rows,
                    io: (rows / 100.0).ceil(), // ~100 rows per page
                }
            }
            LogicalOp::NodeById { .. } => CostEstimate {
                rows: 1.0,
                cpu: 1.0,
                io: 1.0,
            },
            LogicalOp::IndexLookup { type_id, .. } => {
                let base = self.row_count(*type_id);
                let rows = (base * self.default_eq_selectivity).max(1.0);
                CostEstimate {
                    rows,
                    cpu: (rows.log2()).max(1.0),
                    io: (rows.log2()).max(1.0),
                }
            }
            LogicalOp::Expand { input, .. } => {
                let child = self.estimate(input);
                // Each row may produce ~3 expansions on average
                let rows = child.rows * 3.0;
                CostEstimate {
                    rows,
                    cpu: child.cpu + rows,
                    io: child.io + (rows / 100.0).ceil(),
                }
            }
            LogicalOp::Filter { input, .. } => {
                let child = self.estimate(input);
                let rows = child.rows * self.default_range_selectivity;
                CostEstimate {
                    rows,
                    cpu: child.cpu + child.rows, // evaluate predicate for each input row
                    io: child.io,
                }
            }
            LogicalOp::Project { input, .. } => {
                let child = self.estimate(input);
                CostEstimate {
                    rows: child.rows,
                    cpu: child.cpu + child.rows * 0.1, // minimal per-row cost
                    io: child.io,
                }
            }
            LogicalOp::Limit { input, count } => {
                let child = self.estimate(input);
                let rows = child.rows.min(*count as f64);
                CostEstimate {
                    rows,
                    cpu: child.cpu,
                    io: child.io,
                }
            }
            LogicalOp::OrderBy { input, .. } => {
                let child = self.estimate(input);
                let sort_cost = child.rows * (child.rows.log2().max(1.0));
                CostEstimate {
                    rows: child.rows,
                    cpu: child.cpu + sort_cost,
                    io: child.io,
                }
            }
            LogicalOp::Count { input } => {
                let child = self.estimate(input);
                CostEstimate {
                    rows: 1.0,
                    cpu: child.cpu + child.rows,
                    io: child.io,
                }
            }
        }
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Predicate;
    use bikodb_core::value::Value;

    #[test]
    fn test_scan_cost() {
        let mut cm = CostModel::new();
        cm.set_type_stats(
            TypeId(1),
            TypeStats {
                row_count: 5000,
                avg_properties: 4.0,
            },
        );

        let plan = LogicalOp::NodeScan { type_id: TypeId(1) };
        let est = cm.estimate(&plan);
        assert!((est.rows - 5000.0).abs() < f64::EPSILON);
        assert!(est.io > 0.0);
    }

    #[test]
    fn test_index_lookup_cheaper_than_scan() {
        let mut cm = CostModel::new();
        cm.register_index(TypeId(1), 0);
        cm.set_type_stats(
            TypeId(1),
            TypeStats {
                row_count: 100_000,
                avg_properties: 5.0,
            },
        );

        let scan = LogicalOp::NodeScan { type_id: TypeId(1) };
        let index = LogicalOp::IndexLookup {
            type_id: TypeId(1),
            property_id: 0,
            value: Value::from("Alice"),
        };

        let scan_cost = cm.estimate(&scan);
        let index_cost = cm.estimate(&index);
        assert!(index_cost.total() < scan_cost.total());
    }

    #[test]
    fn test_filter_reduces_rows() {
        let cm = CostModel::new();
        let scan = LogicalOp::NodeScan { type_id: TypeId(1) };
        let filtered = LogicalOp::Filter {
            input: Box::new(scan),
            predicate: Predicate::Eq {
                property_id: 0,
                value: Value::Int(42),
            },
        };

        let scan_est = cm.estimate(&LogicalOp::NodeScan { type_id: TypeId(1) });
        let filter_est = cm.estimate(&filtered);
        assert!(filter_est.rows < scan_est.rows);
    }

    #[test]
    fn test_limit_caps_rows() {
        let cm = CostModel::new();
        let plan = LogicalOp::Limit {
            count: 5,
            input: Box::new(LogicalOp::NodeScan { type_id: TypeId(1) }),
        };
        let est = cm.estimate(&plan);
        assert!((est.rows - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_row_count() {
        let cm = CostModel::new();
        // No stats registered → default 1000
        assert!((cm.row_count(TypeId(99)) - 1000.0).abs() < f64::EPSILON);
    }
}
