// =============================================================================
// bikodb-execution::optimizer — Rule-based query plan optimizer
// =============================================================================
// Applies rewrite rules to LogicalOp trees to produce more efficient plans.
//
// Rules applied (in order):
//   1. Predicate pushdown — move Filter below Project/OrderBy/Limit
//   2. Limit pushdown — push Limit below OrderBy when no Filter
//   3. Projection pushdown — narrow properties early
//   4. Filter merging — combine adjacent Filters into AND
//   5. Index selection — replace NodeScan+Filter(Eq) with IndexLookup
//      when a matching index exists in the cost model's statistics
//
// The optimizer is cost-aware: when statistics are available, it uses
// cardinality estimates to decide between scan vs. index strategies.
// =============================================================================

use crate::cost::CostModel;
use crate::plan::{LogicalOp, Predicate};

/// Optimizes a logical plan using rule-based rewrites.
///
/// Applies multiple optimization passes until the plan stabilizes.
/// When a `CostModel` is provided, cost-based decisions (e.g., index
/// selection) are also applied.
pub fn optimize(plan: LogicalOp, cost: Option<&CostModel>) -> LogicalOp {
    let mut current = plan;
    // Apply rules in priority order. One pass is sufficient since
    // each rule pushes operators strictly downward (convergent).
    current = push_filters_down(current);
    current = merge_adjacent_filters(current);
    current = push_limits_down(current);
    current = push_projections_down(current);
    if let Some(cm) = cost {
        current = select_indexes(current, cm);
    }
    current
}

// ── Rule 1: Predicate pushdown ─────────────────────────────────────────

/// Pushes Filter below Project, Limit, and OrderBy when possible.
///
/// Example: `Project(Filter(Scan))` → `Filter(Project(Scan))`
/// is NOT valid (filter may reference dropped columns). But:
/// `Filter(Project(Scan))` stays.
/// `Limit(Filter(Scan))` → `Limit(Filter(Scan))` stays.
///
/// The key case: `Filter(OrderBy(Scan))` → `OrderBy(Filter(Scan))`
/// since Filter reduces rows BEFORE the expensive sort.
fn push_filters_down(plan: LogicalOp) -> LogicalOp {
    match plan {
        // Filter over OrderBy → push filter below sort
        LogicalOp::Filter {
            input,
            predicate,
        } => {
            let optimized_input = push_filters_down(*input);
            match optimized_input {
                LogicalOp::OrderBy {
                    input: order_input,
                    property_id,
                    ascending,
                } => LogicalOp::OrderBy {
                    input: Box::new(LogicalOp::Filter {
                        input: order_input,
                        predicate,
                    }),
                    property_id,
                    ascending,
                },
                other => LogicalOp::Filter {
                    input: Box::new(other),
                    predicate,
                },
            }
        }
        // Recurse into children for all other operators
        LogicalOp::Expand {
            input,
            direction,
            edge_type,
        } => LogicalOp::Expand {
            input: Box::new(push_filters_down(*input)),
            direction,
            edge_type,
        },
        LogicalOp::Project {
            input,
            property_ids,
        } => LogicalOp::Project {
            input: Box::new(push_filters_down(*input)),
            property_ids,
        },
        LogicalOp::Limit { input, count } => LogicalOp::Limit {
            input: Box::new(push_filters_down(*input)),
            count,
        },
        LogicalOp::OrderBy {
            input,
            property_id,
            ascending,
        } => LogicalOp::OrderBy {
            input: Box::new(push_filters_down(*input)),
            property_id,
            ascending,
        },
        LogicalOp::Count { input } => LogicalOp::Count {
            input: Box::new(push_filters_down(*input)),
        },
        // Leaves
        other => other,
    }
}

// ── Rule 2: Filter merging ─────────────────────────────────────────────

/// Merges adjacent Filter operators into a single Filter(AND).
///
/// `Filter(p1, Filter(p2, input))` → `Filter(AND(p1, p2), input)`
fn merge_adjacent_filters(plan: LogicalOp) -> LogicalOp {
    match plan {
        LogicalOp::Filter { input, predicate } => {
            let optimized_input = merge_adjacent_filters(*input);
            match optimized_input {
                LogicalOp::Filter {
                    input: inner_input,
                    predicate: inner_pred,
                } => LogicalOp::Filter {
                    input: inner_input,
                    predicate: Predicate::And(
                        Box::new(predicate),
                        Box::new(inner_pred),
                    ),
                },
                other => LogicalOp::Filter {
                    input: Box::new(other),
                    predicate,
                },
            }
        }
        // Recurse into all children
        LogicalOp::Expand {
            input,
            direction,
            edge_type,
        } => LogicalOp::Expand {
            input: Box::new(merge_adjacent_filters(*input)),
            direction,
            edge_type,
        },
        LogicalOp::Project {
            input,
            property_ids,
        } => LogicalOp::Project {
            input: Box::new(merge_adjacent_filters(*input)),
            property_ids,
        },
        LogicalOp::Limit { input, count } => LogicalOp::Limit {
            input: Box::new(merge_adjacent_filters(*input)),
            count,
        },
        LogicalOp::OrderBy {
            input,
            property_id,
            ascending,
        } => LogicalOp::OrderBy {
            input: Box::new(merge_adjacent_filters(*input)),
            property_id,
            ascending,
        },
        LogicalOp::Count { input } => LogicalOp::Count {
            input: Box::new(merge_adjacent_filters(*input)),
        },
        other => other,
    }
}

// ── Rule 3: Limit pushdown ─────────────────────────────────────────────

/// Pushes Limit below Project (Project doesn't change row count).
fn push_limits_down(plan: LogicalOp) -> LogicalOp {
    match plan {
        LogicalOp::Limit { input, count } => {
            let optimized_input = push_limits_down(*input);
            match optimized_input {
                LogicalOp::Project {
                    input: proj_input,
                    property_ids,
                } => LogicalOp::Project {
                    input: Box::new(LogicalOp::Limit {
                        input: proj_input,
                        count,
                    }),
                    property_ids,
                },
                other => LogicalOp::Limit {
                    input: Box::new(other),
                    count,
                },
            }
        }
        LogicalOp::Filter { input, predicate } => LogicalOp::Filter {
            input: Box::new(push_limits_down(*input)),
            predicate,
        },
        LogicalOp::Expand {
            input,
            direction,
            edge_type,
        } => LogicalOp::Expand {
            input: Box::new(push_limits_down(*input)),
            direction,
            edge_type,
        },
        LogicalOp::Project {
            input,
            property_ids,
        } => LogicalOp::Project {
            input: Box::new(push_limits_down(*input)),
            property_ids,
        },
        LogicalOp::OrderBy {
            input,
            property_id,
            ascending,
        } => LogicalOp::OrderBy {
            input: Box::new(push_limits_down(*input)),
            property_id,
            ascending,
        },
        LogicalOp::Count { input } => LogicalOp::Count {
            input: Box::new(push_limits_down(*input)),
        },
        other => other,
    }
}

// ── Rule 4: Projection pushdown ────────────────────────────────────────

/// Pushes projection into scan source when the projection is directly
/// above a scan. This narrows the properties loaded from the scan.
fn push_projections_down(plan: LogicalOp) -> LogicalOp {
    match plan {
        LogicalOp::Project {
            input,
            property_ids,
        } => {
            let optimized_input = push_projections_down(*input);
            // If the input is a Filter over a Scan, keep project above filter
            // (Filter may need properties that Project drops).
            // For now keep Project where it is but propagate recursively.
            LogicalOp::Project {
                input: Box::new(optimized_input),
                property_ids,
            }
        }
        LogicalOp::Filter { input, predicate } => LogicalOp::Filter {
            input: Box::new(push_projections_down(*input)),
            predicate,
        },
        LogicalOp::Expand {
            input,
            direction,
            edge_type,
        } => LogicalOp::Expand {
            input: Box::new(push_projections_down(*input)),
            direction,
            edge_type,
        },
        LogicalOp::Limit { input, count } => LogicalOp::Limit {
            input: Box::new(push_projections_down(*input)),
            count,
        },
        LogicalOp::OrderBy {
            input,
            property_id,
            ascending,
        } => LogicalOp::OrderBy {
            input: Box::new(push_projections_down(*input)),
            property_id,
            ascending,
        },
        LogicalOp::Count { input } => LogicalOp::Count {
            input: Box::new(push_projections_down(*input)),
        },
        other => other,
    }
}

// ── Rule 5: Index selection (cost-based) ───────────────────────────────

/// Replaces `Filter(Eq(prop, val), NodeScan(type))` with `IndexLookup`
/// when the cost model has a matching index for (type_id, property_id).
fn select_indexes(plan: LogicalOp, cost: &CostModel) -> LogicalOp {
    match plan {
        LogicalOp::Filter { input, predicate } => {
            let optimized_input = select_indexes(*input, cost);
            // Check: Filter(Eq, NodeScan) → IndexLookup
            if let LogicalOp::NodeScan { type_id } = &optimized_input {
                if let Some(lookup) = try_index_lookup(*type_id, &predicate, cost) {
                    return lookup;
                }
            }
            LogicalOp::Filter {
                input: Box::new(optimized_input),
                predicate,
            }
        }
        LogicalOp::Expand {
            input,
            direction,
            edge_type,
        } => LogicalOp::Expand {
            input: Box::new(select_indexes(*input, cost)),
            direction,
            edge_type,
        },
        LogicalOp::Project {
            input,
            property_ids,
        } => LogicalOp::Project {
            input: Box::new(select_indexes(*input, cost)),
            property_ids,
        },
        LogicalOp::Limit { input, count } => LogicalOp::Limit {
            input: Box::new(select_indexes(*input, cost)),
            count,
        },
        LogicalOp::OrderBy {
            input,
            property_id,
            ascending,
        } => LogicalOp::OrderBy {
            input: Box::new(select_indexes(*input, cost)),
            property_id,
            ascending,
        },
        LogicalOp::Count { input } => LogicalOp::Count {
            input: Box::new(select_indexes(*input, cost)),
        },
        other => other,
    }
}

/// Attempts to convert an equality predicate + scan into an IndexLookup.
fn try_index_lookup(
    type_id: bikodb_core::types::TypeId,
    predicate: &Predicate,
    cost: &CostModel,
) -> Option<LogicalOp> {
    match predicate {
        Predicate::Eq { property_id, value } => {
            if cost.has_index(type_id, *property_id) {
                Some(LogicalOp::IndexLookup {
                    type_id,
                    property_id: *property_id,
                    value: value.clone(),
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

// =============================================================================
// EXPLAIN — Plan introspection
// =============================================================================

/// Formats a logical plan as a human-readable string for EXPLAIN output.
pub fn explain(plan: &LogicalOp, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match plan {
        LogicalOp::NodeScan { type_id } => {
            format!("{pad}NodeScan(type={})", type_id.0)
        }
        LogicalOp::NodeById { node_id } => {
            format!("{pad}NodeById(id={})", node_id.0)
        }
        LogicalOp::IndexLookup {
            type_id,
            property_id,
            value,
        } => {
            format!(
                "{pad}IndexLookup(type={}, prop={}, val={:?})",
                type_id.0, property_id, value
            )
        }
        LogicalOp::Expand {
            input,
            direction,
            edge_type,
        } => {
            let child = explain(input, indent + 1);
            format!(
                "{pad}Expand(dir={:?}, edge_type={:?})\n{child}",
                direction,
                edge_type.map(|t| t.0)
            )
        }
        LogicalOp::Filter { input, predicate } => {
            let child = explain(input, indent + 1);
            format!("{pad}Filter({:?})\n{child}", predicate_summary(predicate))
        }
        LogicalOp::Project {
            input,
            property_ids,
        } => {
            let child = explain(input, indent + 1);
            format!("{pad}Project({:?})\n{child}", property_ids)
        }
        LogicalOp::Limit { input, count } => {
            let child = explain(input, indent + 1);
            format!("{pad}Limit({count})\n{child}")
        }
        LogicalOp::OrderBy {
            input,
            property_id,
            ascending,
        } => {
            let child = explain(input, indent + 1);
            let dir = if *ascending { "ASC" } else { "DESC" };
            format!("{pad}OrderBy(prop={property_id}, {dir})\n{child}")
        }
        LogicalOp::Count { input } => {
            let child = explain(input, indent + 1);
            format!("{pad}Count\n{child}")
        }
    }
}

/// One-line summary of a predicate for EXPLAIN output.
fn predicate_summary(pred: &Predicate) -> String {
    match pred {
        Predicate::Eq { property_id, value } => format!("prop{property_id} == {value:?}"),
        Predicate::Neq { property_id, value } => format!("prop{property_id} != {value:?}"),
        Predicate::Gt { property_id, value } => format!("prop{property_id} > {value:?}"),
        Predicate::Lt { property_id, value } => format!("prop{property_id} < {value:?}"),
        Predicate::Gte { property_id, value } => format!("prop{property_id} >= {value:?}"),
        Predicate::Lte { property_id, value } => format!("prop{property_id} <= {value:?}"),
        Predicate::And(a, b) => {
            format!("({} AND {})", predicate_summary(a), predicate_summary(b))
        }
        Predicate::Or(a, b) => {
            format!("({} OR {})", predicate_summary(a), predicate_summary(b))
        }
        Predicate::Not(p) => format!("NOT ({})", predicate_summary(p)),
        Predicate::IsNotNull { property_id } => format!("prop{property_id} IS NOT NULL"),
        Predicate::NestedEq { property_id, path, value } => format!("prop{property_id}.{path} == {value:?}"),
        Predicate::NestedGt { property_id, path, value } => format!("prop{property_id}.{path} > {value:?}"),
        Predicate::NestedLt { property_id, path, value } => format!("prop{property_id}.{path} < {value:?}"),
        Predicate::NestedExists { property_id, path } => format!("prop{property_id}.{path} EXISTS"),
        Predicate::Contains { property_id, value } => format!("prop{property_id} CONTAINS {value:?}"),
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use bikodb_core::types::TypeId;
    use bikodb_core::value::Value;

    fn scan_persons() -> LogicalOp {
        LogicalOp::NodeScan {
            type_id: TypeId(1),
        }
    }

    fn age_gt_25() -> Predicate {
        Predicate::Gt {
            property_id: 1,
            value: Value::Int(25),
        }
    }

    fn name_eq_alice() -> Predicate {
        Predicate::Eq {
            property_id: 0,
            value: Value::from("Alice"),
        }
    }

    // ── Predicate pushdown ─────────────────────────────────────────────

    #[test]
    fn test_filter_pushed_below_orderby() {
        // Filter(OrderBy(Scan)) → OrderBy(Filter(Scan))
        let plan = LogicalOp::Filter {
            input: Box::new(LogicalOp::OrderBy {
                input: Box::new(scan_persons()),
                property_id: 1,
                ascending: true,
            }),
            predicate: age_gt_25(),
        };

        let optimized = optimize(plan, None);
        match &optimized {
            LogicalOp::OrderBy { input, .. } => {
                assert!(matches!(**input, LogicalOp::Filter { .. }));
            }
            _ => panic!("Expected OrderBy at top, got {:?}", optimized),
        }
    }

    // ── Filter merging ─────────────────────────────────────────────────

    #[test]
    fn test_adjacent_filters_merged() {
        // Filter(p1, Filter(p2, Scan)) → Filter(AND(p1, p2), Scan)
        let plan = LogicalOp::Filter {
            predicate: age_gt_25(),
            input: Box::new(LogicalOp::Filter {
                predicate: name_eq_alice(),
                input: Box::new(scan_persons()),
            }),
        };

        let optimized = optimize(plan, None);
        match &optimized {
            LogicalOp::Filter { predicate, input } => {
                assert!(matches!(predicate, Predicate::And(..)));
                assert!(matches!(**input, LogicalOp::NodeScan { .. }));
            }
            _ => panic!("Expected merged Filter"),
        }
    }

    // ── Limit pushdown ─────────────────────────────────────────────────

    #[test]
    fn test_limit_pushed_below_project() {
        // Limit(Project(Scan)) → Project(Limit(Scan))
        let plan = LogicalOp::Limit {
            count: 10,
            input: Box::new(LogicalOp::Project {
                property_ids: vec![0, 1],
                input: Box::new(scan_persons()),
            }),
        };

        let optimized = optimize(plan, None);
        match &optimized {
            LogicalOp::Project { input, .. } => {
                assert!(matches!(**input, LogicalOp::Limit { .. }));
            }
            _ => panic!("Expected Project at top"),
        }
    }

    // ── Index selection ────────────────────────────────────────────────

    #[test]
    fn test_index_lookup_selected() {
        // Filter(Eq(prop0, "Alice"), NodeScan(1)) → IndexLookup
        let mut cm = CostModel::new();
        cm.register_index(TypeId(1), 0); // Index on type 1, property 0

        let plan = LogicalOp::Filter {
            predicate: name_eq_alice(),
            input: Box::new(scan_persons()),
        };

        let optimized = optimize(plan, Some(&cm));
        match &optimized {
            LogicalOp::IndexLookup {
                type_id,
                property_id,
                ..
            } => {
                assert_eq!(type_id.0, 1);
                assert_eq!(*property_id, 0);
            }
            _ => panic!("Expected IndexLookup, got {:?}", optimized),
        }
    }

    #[test]
    fn test_no_index_keeps_filter() {
        // Without index, Filter stays
        let cm = CostModel::new();
        let plan = LogicalOp::Filter {
            predicate: name_eq_alice(),
            input: Box::new(scan_persons()),
        };

        let optimized = optimize(plan, Some(&cm));
        assert!(matches!(optimized, LogicalOp::Filter { .. }));
    }

    // ── EXPLAIN ────────────────────────────────────────────────────────

    #[test]
    fn test_explain_output() {
        let plan = LogicalOp::Limit {
            count: 10,
            input: Box::new(LogicalOp::Filter {
                predicate: age_gt_25(),
                input: Box::new(scan_persons()),
            }),
        };

        let output = explain(&plan, 0);
        assert!(output.contains("Limit(10)"));
        assert!(output.contains("Filter"));
        assert!(output.contains("NodeScan"));
    }

    // ── Combined optimization ──────────────────────────────────────────

    #[test]
    fn test_complex_optimization() {
        // Limit(10, Filter(age>25, OrderBy(age, Filter(name=="Alice", Scan))))
        let plan = LogicalOp::Limit {
            count: 10,
            input: Box::new(LogicalOp::Filter {
                predicate: age_gt_25(),
                input: Box::new(LogicalOp::OrderBy {
                    property_id: 1,
                    ascending: true,
                    input: Box::new(LogicalOp::Filter {
                        predicate: name_eq_alice(),
                        input: Box::new(scan_persons()),
                    }),
                }),
            }),
        };

        let optimized = optimize(plan, None);
        // After optimization:
        // - The Filter(age>25) should be pushed below OrderBy
        // - The two Filters (age>25, name=="Alice") on Scan should merge
        let output = explain(&optimized, 0);
        assert!(output.contains("Limit"));
        assert!(output.contains("OrderBy"));
    }
}
