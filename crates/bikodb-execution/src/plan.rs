// =============================================================================
// bikodb-execution::plan — Plan lógico de ejecución
// =============================================================================
// Representación intermedia entre el parser de queries y los operadores físicos.
//
// Un plan lógico es un árbol de LogicalOp que describe QUÉ hacer,
// sin especificar CÓMO (ej: "filtrar nodos tipo Person" sin elegir índice).
// =============================================================================

use bikodb_core::record::Direction;
use bikodb_core::types::{NodeId, TypeId};
use bikodb_core::value::Value;

/// Operador lógico del plan de ejecución.
///
/// Cada variante describe una operación abstracta sobre el grafo.
/// El planner convierte estos en operadores físicos.
#[derive(Debug, Clone)]
pub enum LogicalOp {
    /// Escanea todos los nodos de un tipo.
    NodeScan {
        type_id: TypeId,
    },

    /// Busca un nodo por ID.
    NodeById {
        node_id: NodeId,
    },

    /// Expande edges desde nodos (traversal de 1 hop).
    Expand {
        input: Box<LogicalOp>,
        direction: Direction,
        edge_type: Option<TypeId>,
    },

    /// Filtra filas por una condición.
    Filter {
        input: Box<LogicalOp>,
        predicate: Predicate,
    },

    /// Selecciona columnas/propiedades específicas.
    Project {
        input: Box<LogicalOp>,
        /// IDs de propiedades a retornar
        property_ids: Vec<u16>,
    },

    /// Limita el número de resultados.
    Limit {
        input: Box<LogicalOp>,
        count: usize,
    },

    /// Ordena resultados por una propiedad.
    OrderBy {
        input: Box<LogicalOp>,
        property_id: u16,
        ascending: bool,
    },

    /// Cuenta resultados.
    Count {
        input: Box<LogicalOp>,
    },

    /// Búsqueda por índice (producida por el optimizador).
    IndexLookup {
        type_id: TypeId,
        property_id: u16,
        value: Value,
    },
}

/// Predicado simple para filtrado.
#[derive(Debug, Clone)]
pub enum Predicate {
    /// Propiedad == valor
    Eq { property_id: u16, value: Value },
    /// Propiedad != valor
    Neq { property_id: u16, value: Value },
    /// Propiedad > valor
    Gt { property_id: u16, value: Value },
    /// Propiedad < valor
    Lt { property_id: u16, value: Value },
    /// Propiedad >= valor
    Gte { property_id: u16, value: Value },
    /// Propiedad <= valor
    Lte { property_id: u16, value: Value },
    /// AND de dos predicados
    And(Box<Predicate>, Box<Predicate>),
    /// OR de dos predicados
    Or(Box<Predicate>, Box<Predicate>),
    /// NOT de un predicado
    Not(Box<Predicate>),
    /// Propiedad IS NOT NULL
    IsNotNull { property_id: u16 },
    /// Nested field: property_id contiene un Map, dot-path señala un campo dentro.
    /// Ejemplo: property_id=5 (campo "metadata"), path="category", value="AI"
    /// Resuelve: node.properties[5].get_path("category") == "AI"
    NestedEq { property_id: u16, path: String, value: Value },
    /// Nested field >
    NestedGt { property_id: u16, path: String, value: Value },
    /// Nested field <
    NestedLt { property_id: u16, path: String, value: Value },
    /// Nested field exists
    NestedExists { property_id: u16, path: String },
    /// Lista contiene valor
    Contains { property_id: u16, value: Value },
}

impl Predicate {
    /// Evalúa el predicado contra un conjunto de propiedades.
    pub fn evaluate(&self, properties: &[(u16, Value)]) -> bool {
        match self {
            Predicate::Eq { property_id, value } => {
                find_prop(properties, *property_id)
                    .map(|v| v == value)
                    .unwrap_or(false)
            }
            Predicate::Neq { property_id, value } => {
                find_prop(properties, *property_id)
                    .map(|v| v != value)
                    .unwrap_or(true)
            }
            Predicate::Gt { property_id, value } => {
                compare_prop(properties, *property_id, value, |ord| {
                    matches!(ord, std::cmp::Ordering::Greater)
                })
            }
            Predicate::Lt { property_id, value } => {
                compare_prop(properties, *property_id, value, |ord| {
                    matches!(ord, std::cmp::Ordering::Less)
                })
            }
            Predicate::Gte { property_id, value } => {
                compare_prop(properties, *property_id, value, |ord| {
                    matches!(ord, std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                })
            }
            Predicate::Lte { property_id, value } => {
                compare_prop(properties, *property_id, value, |ord| {
                    matches!(ord, std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                })
            }
            Predicate::And(a, b) => a.evaluate(properties) && b.evaluate(properties),
            Predicate::Or(a, b) => a.evaluate(properties) || b.evaluate(properties),
            Predicate::Not(p) => !p.evaluate(properties),
            Predicate::IsNotNull { property_id } => {
                find_prop(properties, *property_id)
                    .map(|v| !matches!(v, Value::Null))
                    .unwrap_or(false)
            }
            Predicate::NestedEq { property_id, path, value } => {
                find_prop(properties, *property_id)
                    .and_then(|v| v.get_path(path))
                    .map(|v| v == value)
                    .unwrap_or(false)
            }
            Predicate::NestedGt { property_id, path, value } => {
                nested_compare(properties, *property_id, path, value, |o| {
                    matches!(o, std::cmp::Ordering::Greater)
                })
            }
            Predicate::NestedLt { property_id, path, value } => {
                nested_compare(properties, *property_id, path, value, |o| {
                    matches!(o, std::cmp::Ordering::Less)
                })
            }
            Predicate::NestedExists { property_id, path } => {
                find_prop(properties, *property_id)
                    .and_then(|v| v.get_path(path))
                    .is_some()
            }
            Predicate::Contains { property_id, value } => {
                find_prop(properties, *property_id)
                    .and_then(|v| v.as_list())
                    .map(|list| list.contains(value))
                    .unwrap_or(false)
            }
        }
    }
}

fn find_prop<'a>(props: &'a [(u16, Value)], id: u16) -> Option<&'a Value> {
    props.iter().find(|(k, _)| *k == id).map(|(_, v)| v)
}

fn compare_prop(
    props: &[(u16, Value)],
    id: u16,
    target: &Value,
    cmp_fn: impl Fn(std::cmp::Ordering) -> bool,
) -> bool {
    match find_prop(props, id) {
        Some(Value::Int(a)) => {
            if let Value::Int(b) = target {
                cmp_fn(a.cmp(b))
            } else {
                false
            }
        }
        Some(Value::Float(a)) => {
            if let Value::Float(b) = target {
                a.partial_cmp(b).map(&cmp_fn).unwrap_or(false)
            } else {
                false
            }
        }
        Some(Value::String(a)) => {
            if let Value::String(b) = target {
                cmp_fn(a.cmp(b))
            } else {
                false
            }
        }
        _ => false,
    }
}

fn nested_compare(
    props: &[(u16, Value)],
    property_id: u16,
    path: &str,
    target: &Value,
    cmp_fn: impl Fn(std::cmp::Ordering) -> bool,
) -> bool {
    let resolved = find_prop(props, property_id).and_then(|v| v.get_path(path));
    match resolved {
        Some(Value::Int(a)) => {
            if let Value::Int(b) = target { cmp_fn(a.cmp(b)) } else { false }
        }
        Some(Value::Float(a)) => {
            if let Value::Float(b) = target {
                a.partial_cmp(b).map(&cmp_fn).unwrap_or(false)
            } else { false }
        }
        Some(Value::String(a)) => {
            if let Value::String(b) = target { cmp_fn(a.cmp(b)) } else { false }
        }
        _ => false,
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_props() -> Vec<(u16, Value)> {
        vec![
            (0, Value::from("Alice")),
            (1, Value::Int(30)),
            (2, Value::Float(5.7)),
        ]
    }

    #[test]
    fn test_predicate_eq() {
        let pred = Predicate::Eq {
            property_id: 0,
            value: Value::from("Alice"),
        };
        assert!(pred.evaluate(&sample_props()));
    }

    #[test]
    fn test_predicate_gt() {
        let pred = Predicate::Gt {
            property_id: 1,
            value: Value::Int(25),
        };
        assert!(pred.evaluate(&sample_props()));
    }

    #[test]
    fn test_predicate_and() {
        let pred = Predicate::And(
            Box::new(Predicate::Eq {
                property_id: 0,
                value: Value::from("Alice"),
            }),
            Box::new(Predicate::Gt {
                property_id: 1,
                value: Value::Int(20),
            }),
        );
        assert!(pred.evaluate(&sample_props()));
    }

    #[test]
    fn test_predicate_not() {
        let pred = Predicate::Not(Box::new(Predicate::Eq {
            property_id: 0,
            value: Value::from("Bob"),
        }));
        assert!(pred.evaluate(&sample_props()));
    }

    #[test]
    fn test_predicate_is_not_null() {
        let pred = Predicate::IsNotNull { property_id: 0 };
        assert!(pred.evaluate(&sample_props()));

        let pred2 = Predicate::IsNotNull { property_id: 99 };
        assert!(!pred2.evaluate(&sample_props()));
    }

    // ── Nested / multi-model predicate tests ───────────────────────────

    fn props_with_map() -> Vec<(u16, Value)> {
        let mut address = std::collections::HashMap::new();
        address.insert("city".into(), Value::from("NYC"));
        address.insert("zip".into(), Value::Int(10001));

        vec![
            (0, Value::from("Alice")),
            (1, Value::Int(30)),
            (5, Value::Map(Box::new(address))),
            (6, Value::List(Box::new(vec![
                Value::from("rust"), Value::from("graph"),
            ]))),
        ]
    }

    #[test]
    fn test_nested_eq() {
        let pred = Predicate::NestedEq {
            property_id: 5,
            path: "city".into(),
            value: Value::from("NYC"),
        };
        assert!(pred.evaluate(&props_with_map()));
    }

    #[test]
    fn test_nested_eq_miss() {
        let pred = Predicate::NestedEq {
            property_id: 5,
            path: "city".into(),
            value: Value::from("LA"),
        };
        assert!(!pred.evaluate(&props_with_map()));
    }

    #[test]
    fn test_nested_gt() {
        let pred = Predicate::NestedGt {
            property_id: 5,
            path: "zip".into(),
            value: Value::Int(10000),
        };
        assert!(pred.evaluate(&props_with_map()));
    }

    #[test]
    fn test_nested_lt() {
        let pred = Predicate::NestedLt {
            property_id: 5,
            path: "zip".into(),
            value: Value::Int(20000),
        };
        assert!(pred.evaluate(&props_with_map()));
    }

    #[test]
    fn test_nested_exists() {
        let pred = Predicate::NestedExists { property_id: 5, path: "city".into() };
        assert!(pred.evaluate(&props_with_map()));

        let pred2 = Predicate::NestedExists { property_id: 5, path: "state".into() };
        assert!(!pred2.evaluate(&props_with_map()));
    }

    #[test]
    fn test_contains_list() {
        let pred = Predicate::Contains {
            property_id: 6,
            value: Value::from("rust"),
        };
        assert!(pred.evaluate(&props_with_map()));

        let pred2 = Predicate::Contains {
            property_id: 6,
            value: Value::from("python"),
        };
        assert!(!pred2.evaluate(&props_with_map()));
    }
}
