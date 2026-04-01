// =============================================================================
// bikodb-execution::pipeline — Pipeline de ejecución sobre el grafo
// =============================================================================
// Conecta el grafo con los operadores para ejecutar queries completas.
//
// Pipeline: Graph + LogicalOp → Vec<Row>
// Convierte operaciones lógicas en operadores físicos y los ejecuta.
// =============================================================================

use crate::cost::CostModel;
use crate::operator::{CountOp, FilterOp, LimitOp, Operator, ProjectOp, Row, VecSource};
use crate::optimizer;
use crate::plan::LogicalOp;
use bikodb_core::error::BikoResult;
use bikodb_core::record::Direction;
use bikodb_core::types::TypeId;
use bikodb_graph::ConcurrentGraph;

/// Ejecuta un plan lógico completo sobre el grafo.
///
/// Convierte el árbol de LogicalOp en un pipeline de operadores físicos,
/// y materializa todos los resultados.
///
/// # Ejemplo
/// ```
/// use bikodb_execution::pipeline::execute;
/// use bikodb_execution::plan::LogicalOp;
/// use bikodb_graph::ConcurrentGraph;
/// use bikodb_core::types::{NodeId, TypeId};
///
/// let graph = ConcurrentGraph::new();
/// let n = graph.insert_node(TypeId(1));
/// graph.set_node_property(n, 0, bikodb_core::value::Value::from("Alice")).unwrap();
///
/// let plan = LogicalOp::NodeScan { type_id: TypeId(1) };
/// let rows = execute(&graph, plan).unwrap();
/// assert_eq!(rows.len(), 1);
/// ```
pub fn execute(graph: &ConcurrentGraph, plan: LogicalOp) -> BikoResult<Vec<Row>> {
    let mut op = build_operator(graph, plan)?;
    let mut results = Vec::new();

    while let Some(row) = op.next() {
        results.push(row);
    }

    Ok(results)
}

/// Ejecuta un plan lógico con optimización previa.
///
/// Aplica el optimizador rule-based (y cost-based si se provee CostModel)
/// antes de ejecutar.
pub fn execute_optimized(
    graph: &ConcurrentGraph,
    plan: LogicalOp,
    cost: Option<&CostModel>,
) -> BikoResult<Vec<Row>> {
    let optimized = optimizer::optimize(plan, cost);
    execute(graph, optimized)
}

/// Construye recursivamente el árbol de operadores físicos.
fn build_operator(
    graph: &ConcurrentGraph,
    plan: LogicalOp,
) -> BikoResult<Box<dyn Operator>> {
    match plan {
        LogicalOp::NodeScan { type_id } => {
            // Scan all nodes of the given type
            let rows = scan_nodes_by_type(graph, type_id);
            Ok(Box::new(VecSource::new(rows)))
        }

        LogicalOp::NodeById { node_id } => {
            let rows = match graph.get_node(node_id) {
                Some(data) => vec![Row {
                    node_id: data.id,
                    properties: data.properties,
                }],
                None => vec![],
            };
            Ok(Box::new(VecSource::new(rows)))
        }

        LogicalOp::Expand {
            input,
            direction,
            edge_type,
        } => {
            // Materializar input, luego expandir (1-hop traversal)
            let mut input_op = build_operator(graph, *input)?;
            let mut expanded_rows = Vec::new();

            while let Some(row) = input_op.next() {
                let neighbors = match edge_type {
                    Some(et) => graph.neighbors_by_type(row.node_id, direction, et)?,
                    None => graph.neighbors(row.node_id, direction)?,
                };

                for neighbor_id in neighbors {
                    if let Some(ndata) = graph.get_node(neighbor_id) {
                        expanded_rows.push(Row {
                            node_id: ndata.id,
                            properties: ndata.properties,
                        });
                    }
                }
            }

            Ok(Box::new(VecSource::new(expanded_rows)))
        }

        LogicalOp::Filter { input, predicate } => {
            let input_op = build_operator(graph, *input)?;
            Ok(Box::new(FilterOp::new(input_op, predicate)))
        }

        LogicalOp::Project {
            input,
            property_ids,
        } => {
            let input_op = build_operator(graph, *input)?;
            Ok(Box::new(ProjectOp::new(input_op, property_ids)))
        }

        LogicalOp::Limit { input, count } => {
            let input_op = build_operator(graph, *input)?;
            Ok(Box::new(LimitOp::new(input_op, count)))
        }

        LogicalOp::Count { input } => {
            let input_op = build_operator(graph, *input)?;
            Ok(Box::new(CountOp::new(input_op)))
        }

        LogicalOp::OrderBy {
            input,
            property_id,
            ascending,
        } => {
            // Materializar, ordenar, retornar
            let mut input_op = build_operator(graph, *input)?;
            let mut rows = Vec::new();
            while let Some(row) = input_op.next() {
                rows.push(row);
            }

            rows.sort_by(|a, b| {
                let va = a.properties.iter().find(|(k, _)| *k == property_id).map(|(_, v)| v);
                let vb = b.properties.iter().find(|(k, _)| *k == property_id).map(|(_, v)| v);

                let ord = match (va, vb) {
                    (Some(bikodb_core::value::Value::Int(a)), Some(bikodb_core::value::Value::Int(b))) => a.cmp(b),
                    (Some(bikodb_core::value::Value::String(a)), Some(bikodb_core::value::Value::String(b))) => a.cmp(b),
                    _ => std::cmp::Ordering::Equal,
                };

                if ascending { ord } else { ord.reverse() }
            });

            Ok(Box::new(VecSource::new(rows)))
        }

        LogicalOp::IndexLookup {
            type_id,
            property_id,
            value,
        } => {
            // Index lookup falls back to a filtered scan (the storage layer
            // will provide real index access in the future).
            let all = scan_nodes_by_type(graph, type_id);
            let matched: Vec<Row> = all
                .into_iter()
                .filter(|row| {
                    row.properties
                        .iter()
                        .any(|(pid, val)| *pid == property_id && *val == value)
                })
                .collect();
            Ok(Box::new(VecSource::new(matched)))
        }
    }
}

/// Scan de nodos por tipo (parallel DashMap scan via Rayon).
///
/// Uses `ConcurrentGraph::par_scan_nodes_by_type()` to parallelize the
/// type-filter across DashMap shards. For small graphs (<1024 nodes),
/// falls back to sequential scan to avoid Rayon scheduling overhead.
fn scan_nodes_by_type(graph: &ConcurrentGraph, type_id: TypeId) -> Vec<Row> {
    let node_count = graph.node_count();

    if node_count < 1024 {
        // Sequential path for small graphs (avoid rayon overhead)
        let mut rows = Vec::new();
        graph.iter_nodes(|_id, data| {
            if data.type_id == type_id {
                rows.push(Row {
                    node_id: data.id,
                    properties: data.properties.clone(),
                });
            }
        });
        rows
    } else {
        // Parallel scan for large graphs
        graph
            .par_scan_nodes_by_type(type_id)
            .into_iter()
            .map(|(id, props)| Row {
                node_id: id,
                properties: props,
            })
            .collect()
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Predicate;
    use bikodb_core::types::NodeId;
    use bikodb_core::value::Value;

    fn setup_graph() -> ConcurrentGraph {
        let g = ConcurrentGraph::new();

        // 3 personas
        let a = g.insert_node_with_props(
            TypeId(1),
            vec![(0, Value::from("Alice")), (1, Value::Int(30))],
        );
        let b = g.insert_node_with_props(
            TypeId(1),
            vec![(0, Value::from("Bob")), (1, Value::Int(25))],
        );
        let c = g.insert_node_with_props(
            TypeId(1),
            vec![(0, Value::from("Charlie")), (1, Value::Int(35))],
        );

        // Edge: Alice → Bob, Alice → Charlie
        g.insert_edge(a, b, TypeId(10)).unwrap();
        g.insert_edge(a, c, TypeId(10)).unwrap();

        g
    }

    #[test]
    fn test_execute_node_scan() {
        let g = setup_graph();
        let plan = LogicalOp::NodeScan { type_id: TypeId(1) };
        let rows = execute(&g, plan).unwrap();
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_execute_node_by_id() {
        let g = setup_graph();
        let plan = LogicalOp::NodeById {
            node_id: NodeId(1),
        };
        let rows = execute(&g, plan).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].properties[0].1.as_str(), Some("Alice"));
    }

    #[test]
    fn test_execute_filter() {
        let g = setup_graph();
        let plan = LogicalOp::Filter {
            input: Box::new(LogicalOp::NodeScan { type_id: TypeId(1) }),
            predicate: Predicate::Gt {
                property_id: 1,
                value: Value::Int(28),
            },
        };
        let rows = execute(&g, plan).unwrap();
        assert_eq!(rows.len(), 2); // Alice(30) and Charlie(35)
    }

    #[test]
    fn test_execute_expand() {
        let g = setup_graph();
        // Start from Alice (NodeId(1)), expand OUT
        let plan = LogicalOp::Expand {
            input: Box::new(LogicalOp::NodeById {
                node_id: NodeId(1),
            }),
            direction: Direction::Out,
            edge_type: None,
        };
        let rows = execute(&g, plan).unwrap();
        assert_eq!(rows.len(), 2); // Bob and Charlie
    }

    #[test]
    fn test_execute_limit() {
        let g = setup_graph();
        let plan = LogicalOp::Limit {
            input: Box::new(LogicalOp::NodeScan { type_id: TypeId(1) }),
            count: 2,
        };
        let rows = execute(&g, plan).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_execute_count() {
        let g = setup_graph();
        let plan = LogicalOp::Count {
            input: Box::new(LogicalOp::NodeScan { type_id: TypeId(1) }),
        };
        let rows = execute(&g, plan).unwrap();
        assert_eq!(rows[0].properties[0].1, Value::Int(3));
    }

    #[test]
    fn test_execute_order_by() {
        let g = setup_graph();
        let plan = LogicalOp::OrderBy {
            input: Box::new(LogicalOp::NodeScan { type_id: TypeId(1) }),
            property_id: 1, // age
            ascending: true,
        };
        let rows = execute(&g, plan).unwrap();
        assert_eq!(rows[0].properties[1].1, Value::Int(25)); // Bob
        assert_eq!(rows[2].properties[1].1, Value::Int(35)); // Charlie
    }
}
