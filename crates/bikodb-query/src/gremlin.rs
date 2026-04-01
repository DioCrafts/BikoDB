// =============================================================================
// bikodb-query::gremlin — Gremlin traversal language interpreter
// =============================================================================
// Implements a subset of Apache TinkerPop Gremlin traversal language:
//
//   g.V()                          — all vertices
//   g.V().hasLabel('Person')       — vertices by label
//   g.V(id)                        — vertex by ID
//   g.V().has('age', gt(25))       — filter by property
//   g.V().has('name', 'Alice')     — equality shorthand
//   g.V().out('KNOWS')             — outgoing traversal
//   g.V().in('KNOWS')              — incoming traversal
//   g.V().both('KNOWS')            — bidirectional traversal
//   g.V().out('KNOWS').has(...)    — chained traversal + filter
//   g.V().values('name')           — project single property
//   g.V().count()                  — count results
//   g.V().limit(10)                — limit results
//   g.V().order().by('age', asc)   — order results
//
// Predicates:
//   eq(val), neq(val), gt(val), lt(val), gte(val), lte(val)
//
// ## Design
// The parser converts the Gremlin chain into a LogicalOp tree in a single
// pass, building bottom-up: each step wraps the previous as its input.
// =============================================================================

use bikodb_core::record::Direction;
use bikodb_core::types::{NodeId, TypeId};
use bikodb_core::value::Value;
use bikodb_execution::plan::{LogicalOp, Predicate};
use std::collections::HashMap;

/// Error de parsing Gremlin.
#[derive(Debug, thiserror::Error)]
pub enum GremlinParseError {
    #[error("Unexpected token at position {pos}: '{got}'")]
    UnexpectedToken { pos: usize, got: String },
    #[error("Unexpected end of input")]
    UnexpectedEof,
    #[error("Unknown label: '{0}'")]
    UnknownLabel(String),
    #[error("Unknown relationship: '{0}'")]
    UnknownRelType(String),
    #[error("Expected '.': step chaining requires dot notation")]
    ExpectedDot,
    #[error("Invalid Gremlin: must start with 'g.V()' or 'g.E()'")]
    InvalidStart,
}

/// Mapeo de labels a TypeId.
pub type LabelMap = HashMap<String, TypeId>;
/// Mapeo de nombres de propiedad a ID.
pub type PropMap = HashMap<String, u16>;
/// Mapeo de nombres de relación a TypeId.
pub type RelMap = HashMap<String, TypeId>;

/// Parsea una query Gremlin y produce un LogicalOp.
///
/// # Ejemplo
/// ```
/// use bikodb_query::gremlin::{parse_gremlin, LabelMap, PropMap, RelMap};
/// use bikodb_core::types::TypeId;
/// use std::collections::HashMap;
///
/// let mut labels = LabelMap::new();
/// labels.insert("Person".into(), TypeId(1));
///
/// let mut props = PropMap::new();
/// props.insert("name".into(), 0);
/// props.insert("age".into(), 1);
///
/// let rels = RelMap::new();
///
/// let plan = parse_gremlin(
///     "g.V().hasLabel('Person').has('age', gt(25)).limit(10)",
///     &labels, &props, &rels,
/// ).unwrap();
/// ```
pub fn parse_gremlin(
    query: &str,
    label_map: &LabelMap,
    prop_map: &PropMap,
    rel_map: &RelMap,
) -> Result<LogicalOp, GremlinParseError> {
    let mut parser = GremlinParser::new(query, label_map, prop_map, rel_map);
    parser.parse()
}

// ─────────────────────────────────────────────────────────────────────────────

struct GremlinParser<'a> {
    input: &'a str,
    pos: usize,
    label_map: &'a LabelMap,
    prop_map: &'a PropMap,
    rel_map: &'a RelMap,
}

impl<'a> GremlinParser<'a> {
    fn new(
        input: &'a str,
        label_map: &'a LabelMap,
        prop_map: &'a PropMap,
        rel_map: &'a RelMap,
    ) -> Self {
        Self {
            input,
            pos: 0,
            label_map,
            prop_map,
            rel_map,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.input.len()
            && self.input.as_bytes()[self.pos].is_ascii_whitespace()
        {
            self.pos += 1;
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.input.as_bytes().get(self.pos).copied()
    }

    fn expect_char(&mut self, ch: u8) -> Result<(), GremlinParseError> {
        self.skip_ws();
        if self.peek_byte() == Some(ch) {
            self.pos += 1;
            Ok(())
        } else {
            Err(GremlinParseError::UnexpectedToken {
                pos: self.pos,
                got: self.peek_byte().map(|b| (b as char).to_string()).unwrap_or("EOF".into()),
            })
        }
    }

    fn maybe_char(&mut self, ch: u8) -> bool {
        self.skip_ws();
        if self.peek_byte() == Some(ch) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn read_ident(&mut self) -> Result<String, GremlinParseError> {
        self.skip_ws();
        let start = self.pos;
        let bytes = self.input.as_bytes();
        while self.pos < bytes.len()
            && (bytes[self.pos].is_ascii_alphanumeric() || bytes[self.pos] == b'_')
        {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(GremlinParseError::UnexpectedEof);
        }
        Ok(self.input[start..self.pos].to_string())
    }

    fn read_string_literal(&mut self) -> Result<String, GremlinParseError> {
        self.skip_ws();
        let quote = self.peek_byte();
        if quote != Some(b'\'') && quote != Some(b'"') {
            return Err(GremlinParseError::UnexpectedToken {
                pos: self.pos,
                got: self.peek_byte().map(|b| (b as char).to_string()).unwrap_or("EOF".into()),
            });
        }
        let q = quote.unwrap();
        self.pos += 1; // opening quote
        let start = self.pos;
        while self.pos < self.input.len() && self.input.as_bytes()[self.pos] != q {
            self.pos += 1;
        }
        let s = self.input[start..self.pos].to_string();
        if self.pos < self.input.len() {
            self.pos += 1; // closing quote
        }
        Ok(s)
    }

    fn read_number(&mut self) -> Result<Value, GremlinParseError> {
        self.skip_ws();
        let start = self.pos;
        let bytes = self.input.as_bytes();
        if self.pos < bytes.len() && bytes[self.pos] == b'-' {
            self.pos += 1;
        }
        while self.pos < bytes.len() && bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos < bytes.len() && bytes[self.pos] == b'.' {
            self.pos += 1;
            while self.pos < bytes.len() && bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
            let f: f64 = self.input[start..self.pos]
                .parse()
                .map_err(|_| GremlinParseError::UnexpectedToken {
                    pos: start,
                    got: self.input[start..self.pos].to_string(),
                })?;
            Ok(Value::Float(f))
        } else {
            let n: i64 = self.input[start..self.pos]
                .parse()
                .map_err(|_| GremlinParseError::UnexpectedToken {
                    pos: start,
                    got: self.input[start..self.pos].to_string(),
                })?;
            Ok(Value::Int(n))
        }
    }

    /// Read a value: string literal, number, or predicate function.
    fn read_value(&mut self) -> Result<Value, GremlinParseError> {
        self.skip_ws();
        match self.peek_byte() {
            Some(b'\'') | Some(b'"') => {
                let s = self.read_string_literal()?;
                Ok(Value::String(s))
            }
            Some(b) if b.is_ascii_digit() || b == b'-' => self.read_number(),
            _ => {
                // Try boolean/null
                let word = self.read_ident()?;
                match word.as_str() {
                    "true" => Ok(Value::Bool(true)),
                    "false" => Ok(Value::Bool(false)),
                    "null" => Ok(Value::Null),
                    _ => Err(GremlinParseError::UnexpectedToken {
                        pos: self.pos,
                        got: word,
                    }),
                }
            }
        }
    }

    /// Parse predicate function: gt(val), lt(val), gte(val), lte(val), eq(val), neq(val)
    fn read_predicate_fn(&mut self, prop_id: u16) -> Result<Predicate, GremlinParseError> {
        self.skip_ws();
        // Check if it's a predicate function or a plain value (eq shorthand)
        match self.peek_byte() {
            Some(b'\'') | Some(b'"') => {
                // Plain string → eq shorthand
                let val = self.read_value()?;
                Ok(Predicate::Eq {
                    property_id: prop_id,
                    value: val,
                })
            }
            Some(b) if b.is_ascii_digit() || b == b'-' => {
                let val = self.read_value()?;
                Ok(Predicate::Eq {
                    property_id: prop_id,
                    value: val,
                })
            }
            _ => {
                // Predicate function: gt(val), lt(val), etc.
                let fn_name = self.read_ident()?;
                self.expect_char(b'(')?;
                let val = self.read_value()?;
                self.expect_char(b')')?;

                match fn_name.as_str() {
                    "eq" => Ok(Predicate::Eq {
                        property_id: prop_id,
                        value: val,
                    }),
                    "neq" => Ok(Predicate::Neq {
                        property_id: prop_id,
                        value: val,
                    }),
                    "gt" => Ok(Predicate::Gt {
                        property_id: prop_id,
                        value: val,
                    }),
                    "lt" => Ok(Predicate::Lt {
                        property_id: prop_id,
                        value: val,
                    }),
                    "gte" => Ok(Predicate::Gte {
                        property_id: prop_id,
                        value: val,
                    }),
                    "lte" => Ok(Predicate::Lte {
                        property_id: prop_id,
                        value: val,
                    }),
                    _ => Err(GremlinParseError::UnexpectedToken {
                        pos: self.pos,
                        got: fn_name,
                    }),
                }
            }
        }
    }

    /// Main parse entry: g.V()... or g.V(id)...
    fn parse(&mut self) -> Result<LogicalOp, GremlinParseError> {
        self.skip_ws();

        // Expect "g"
        let start_ident = self.read_ident()?;
        if start_ident != "g" {
            return Err(GremlinParseError::InvalidStart);
        }

        self.expect_char(b'.')?;

        let step = self.read_ident()?;
        if step != "V" && step != "E" {
            return Err(GremlinParseError::InvalidStart);
        }

        self.expect_char(b'(')?;

        // Check for V(id)
        let mut plan = if step == "V" {
            self.skip_ws();
            if self.peek_byte() != Some(b')') {
                // V(id) — vertex by ID
                let val = self.read_value()?;
                self.expect_char(b')')?;
                match val {
                    Value::Int(id) => LogicalOp::NodeById {
                        node_id: NodeId(id as u64),
                    },
                    _ => {
                        return Err(GremlinParseError::UnexpectedToken {
                            pos: self.pos,
                            got: format!("{val:?}"),
                        });
                    }
                }
            } else {
                // V() — all vertices
                self.expect_char(b')')?;
                LogicalOp::NodeScan {
                    type_id: TypeId(0), // Wildcard; hasLabel narrows this
                }
            }
        } else {
            // E() — edges (scan all edge types, represented as TypeId(0))
            self.expect_char(b')')?;
            LogicalOp::NodeScan {
                type_id: TypeId(0),
            }
        };

        // Parse chained steps
        while self.maybe_char(b'.') {
            let step_name = self.read_ident()?;

            match step_name.as_str() {
                "hasLabel" => {
                    self.expect_char(b'(')?;
                    let label = self.read_string_literal()?;
                    self.expect_char(b')')?;

                    let type_id = self
                        .label_map
                        .get(&label)
                        .copied()
                        .ok_or_else(|| GremlinParseError::UnknownLabel(label))?;

                    plan = LogicalOp::NodeScan { type_id };
                }
                "has" => {
                    // has('prop', value_or_predicate)
                    self.expect_char(b'(')?;
                    let prop_name = self.read_string_literal()?;
                    let prop_id = self.prop_map.get(&prop_name).copied().unwrap_or(0);

                    self.expect_char(b',')?;
                    let predicate = self.read_predicate_fn(prop_id)?;
                    self.expect_char(b')')?;

                    plan = LogicalOp::Filter {
                        input: Box::new(plan),
                        predicate,
                    };
                }
                "out" => {
                    self.expect_char(b'(')?;
                    let edge_type = if self.peek_byte() != Some(b')') {
                        let rel_name = self.read_string_literal()?;
                        Some(
                            self.rel_map
                                .get(&rel_name)
                                .copied()
                                .ok_or_else(|| GremlinParseError::UnknownRelType(rel_name))?,
                        )
                    } else {
                        None
                    };
                    self.expect_char(b')')?;

                    plan = LogicalOp::Expand {
                        input: Box::new(plan),
                        direction: Direction::Out,
                        edge_type,
                    };
                }
                "in" => {
                    self.expect_char(b'(')?;
                    let edge_type = if self.peek_byte() != Some(b')') {
                        let rel_name = self.read_string_literal()?;
                        Some(
                            self.rel_map
                                .get(&rel_name)
                                .copied()
                                .ok_or_else(|| GremlinParseError::UnknownRelType(rel_name))?,
                        )
                    } else {
                        None
                    };
                    self.expect_char(b')')?;

                    plan = LogicalOp::Expand {
                        input: Box::new(plan),
                        direction: Direction::In,
                        edge_type,
                    };
                }
                "both" => {
                    self.expect_char(b'(')?;
                    let edge_type = if self.peek_byte() != Some(b')') {
                        let rel_name = self.read_string_literal()?;
                        Some(
                            self.rel_map
                                .get(&rel_name)
                                .copied()
                                .ok_or_else(|| GremlinParseError::UnknownRelType(rel_name))?,
                        )
                    } else {
                        None
                    };
                    self.expect_char(b')')?;

                    plan = LogicalOp::Expand {
                        input: Box::new(plan),
                        direction: Direction::Both,
                        edge_type,
                    };
                }
                "values" => {
                    // values('prop') → Project to single property
                    self.expect_char(b'(')?;
                    let prop_name = self.read_string_literal()?;
                    let prop_id = self.prop_map.get(&prop_name).copied().unwrap_or(0);
                    self.expect_char(b')')?;

                    plan = LogicalOp::Project {
                        input: Box::new(plan),
                        property_ids: vec![prop_id],
                    };
                }
                "count" => {
                    self.expect_char(b'(')?;
                    self.expect_char(b')')?;
                    plan = LogicalOp::Count {
                        input: Box::new(plan),
                    };
                }
                "limit" => {
                    self.expect_char(b'(')?;
                    let val = self.read_value()?;
                    self.expect_char(b')')?;
                    let count = match val {
                        Value::Int(n) => n as usize,
                        _ => {
                            return Err(GremlinParseError::UnexpectedToken {
                                pos: self.pos,
                                got: format!("{val:?}"),
                            });
                        }
                    };
                    plan = LogicalOp::Limit {
                        input: Box::new(plan),
                        count,
                    };
                }
                "order" => {
                    // order().by('prop', asc/desc)
                    self.expect_char(b'(')?;
                    self.expect_char(b')')?;

                    // Expect .by(...)
                    self.expect_char(b'.')?;
                    let by = self.read_ident()?;
                    if by != "by" {
                        return Err(GremlinParseError::UnexpectedToken {
                            pos: self.pos,
                            got: by,
                        });
                    }
                    self.expect_char(b'(')?;
                    let prop_name = self.read_string_literal()?;
                    let prop_id = self.prop_map.get(&prop_name).copied().unwrap_or(0);

                    let ascending = if self.maybe_char(b',') {
                        self.skip_ws();
                        let dir = self.read_ident()?;
                        match dir.as_str() {
                            "asc" | "incr" => true,
                            "desc" | "decr" => false,
                            _ => true,
                        }
                    } else {
                        true
                    };
                    self.expect_char(b')')?;

                    plan = LogicalOp::OrderBy {
                        input: Box::new(plan),
                        property_id: prop_id,
                        ascending,
                    };
                }
                other => {
                    return Err(GremlinParseError::UnexpectedToken {
                        pos: self.pos,
                        got: other.to_string(),
                    });
                }
            }
        }

        Ok(plan)
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn test_maps() -> (LabelMap, PropMap, RelMap) {
        let mut labels = LabelMap::new();
        labels.insert("Person".into(), TypeId(1));
        labels.insert("Company".into(), TypeId(2));

        let mut props = PropMap::new();
        props.insert("name".into(), 0);
        props.insert("age".into(), 1);

        let mut rels = RelMap::new();
        rels.insert("KNOWS".into(), TypeId(10));
        rels.insert("WORKS_AT".into(), TypeId(11));

        (labels, props, rels)
    }

    #[test]
    fn test_g_v_all() {
        let (labels, props, rels) = test_maps();
        let plan = parse_gremlin("g.V()", &labels, &props, &rels).unwrap();
        assert!(matches!(plan, LogicalOp::NodeScan { type_id } if type_id == TypeId(0)));
    }

    #[test]
    fn test_g_v_by_id() {
        let (labels, props, rels) = test_maps();
        let plan = parse_gremlin("g.V(42)", &labels, &props, &rels).unwrap();
        assert!(matches!(plan, LogicalOp::NodeById { node_id } if node_id == NodeId(42)));
    }

    #[test]
    fn test_has_label() {
        let (labels, props, rels) = test_maps();
        let plan = parse_gremlin("g.V().hasLabel('Person')", &labels, &props, &rels).unwrap();
        assert!(matches!(plan, LogicalOp::NodeScan { type_id } if type_id == TypeId(1)));
    }

    #[test]
    fn test_has_with_predicate() {
        let (labels, props, rels) = test_maps();
        let plan = parse_gremlin(
            "g.V().hasLabel('Person').has('age', gt(25))",
            &labels,
            &props,
            &rels,
        )
        .unwrap();

        match plan {
            LogicalOp::Filter { predicate, .. } => {
                assert!(matches!(predicate, Predicate::Gt { .. }));
            }
            _ => panic!("Expected Filter, got {plan:?}"),
        }
    }

    #[test]
    fn test_has_eq_shorthand() {
        let (labels, props, rels) = test_maps();
        let plan = parse_gremlin(
            "g.V().hasLabel('Person').has('name', 'Alice')",
            &labels,
            &props,
            &rels,
        )
        .unwrap();

        match plan {
            LogicalOp::Filter { predicate, .. } => match predicate {
                Predicate::Eq { value, .. } => {
                    assert_eq!(value, Value::String("Alice".into()));
                }
                _ => panic!("Expected Eq"),
            },
            _ => panic!("Expected Filter"),
        }
    }

    #[test]
    fn test_out_traversal() {
        let (labels, props, rels) = test_maps();
        let plan = parse_gremlin(
            "g.V().hasLabel('Person').out('KNOWS')",
            &labels,
            &props,
            &rels,
        )
        .unwrap();

        match plan {
            LogicalOp::Expand {
                direction,
                edge_type,
                ..
            } => {
                assert_eq!(direction, Direction::Out);
                assert_eq!(edge_type, Some(TypeId(10)));
            }
            _ => panic!("Expected Expand"),
        }
    }

    #[test]
    fn test_in_traversal() {
        let (labels, props, rels) = test_maps();
        let plan = parse_gremlin(
            "g.V().hasLabel('Person').in('WORKS_AT')",
            &labels,
            &props,
            &rels,
        )
        .unwrap();

        match plan {
            LogicalOp::Expand {
                direction,
                edge_type,
                ..
            } => {
                assert_eq!(direction, Direction::In);
                assert_eq!(edge_type, Some(TypeId(11)));
            }
            _ => panic!("Expected Expand"),
        }
    }

    #[test]
    fn test_both_traversal() {
        let (labels, props, rels) = test_maps();
        let plan = parse_gremlin(
            "g.V().hasLabel('Person').both()",
            &labels,
            &props,
            &rels,
        )
        .unwrap();

        assert!(matches!(plan, LogicalOp::Expand { direction: Direction::Both, .. }));
    }

    #[test]
    fn test_values_projection() {
        let (labels, props, rels) = test_maps();
        let plan = parse_gremlin(
            "g.V().hasLabel('Person').values('name')",
            &labels,
            &props,
            &rels,
        )
        .unwrap();

        match plan {
            LogicalOp::Project { property_ids, .. } => {
                assert_eq!(property_ids, vec![0]);
            }
            _ => panic!("Expected Project"),
        }
    }

    #[test]
    fn test_count() {
        let (labels, props, rels) = test_maps();
        let plan = parse_gremlin(
            "g.V().hasLabel('Person').count()",
            &labels,
            &props,
            &rels,
        )
        .unwrap();

        assert!(matches!(plan, LogicalOp::Count { .. }));
    }

    #[test]
    fn test_limit() {
        let (labels, props, rels) = test_maps();
        let plan = parse_gremlin(
            "g.V().hasLabel('Person').limit(5)",
            &labels,
            &props,
            &rels,
        )
        .unwrap();

        match plan {
            LogicalOp::Limit { count, .. } => assert_eq!(count, 5),
            _ => panic!("Expected Limit"),
        }
    }

    #[test]
    fn test_order_by() {
        let (labels, props, rels) = test_maps();
        let plan = parse_gremlin(
            "g.V().hasLabel('Person').order().by('age', desc)",
            &labels,
            &props,
            &rels,
        )
        .unwrap();

        match plan {
            LogicalOp::OrderBy {
                property_id,
                ascending,
                ..
            } => {
                assert_eq!(property_id, 1);
                assert!(!ascending);
            }
            _ => panic!("Expected OrderBy"),
        }
    }

    #[test]
    fn test_complex_chain() {
        let (labels, props, rels) = test_maps();
        // g.V().hasLabel('Person').has('age', gt(20)).out('KNOWS').values('name').limit(10)
        let plan = parse_gremlin(
            "g.V().hasLabel('Person').has('age', gt(20)).out('KNOWS').values('name').limit(10)",
            &labels,
            &props,
            &rels,
        )
        .unwrap();

        // Should be: Limit(Project(Expand(Filter(NodeScan))))
        match plan {
            LogicalOp::Limit { count, input } => {
                assert_eq!(count, 10);
                assert!(matches!(*input, LogicalOp::Project { .. }));
            }
            _ => panic!("Expected Limit at top"),
        }
    }

    #[test]
    fn test_unknown_label_error() {
        let (labels, props, rels) = test_maps();
        let result = parse_gremlin("g.V().hasLabel('Unknown')", &labels, &props, &rels);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_start() {
        let (labels, props, rels) = test_maps();
        let result = parse_gremlin("x.V()", &labels, &props, &rels);
        assert!(result.is_err());
    }

    #[test]
    fn test_has_with_all_predicates() {
        let (labels, props, rels) = test_maps();

        for (pred, expected) in [
            ("eq(10)", "Eq"),
            ("neq(10)", "Neq"),
            ("gt(10)", "Gt"),
            ("lt(10)", "Lt"),
            ("gte(10)", "Gte"),
            ("lte(10)", "Lte"),
        ] {
            let q = format!("g.V().has('age', {pred})");
            let plan = parse_gremlin(&q, &labels, &props, &rels).unwrap();
            match plan {
                LogicalOp::Filter { predicate, .. } => {
                    let name = format!("{predicate:?}");
                    assert!(
                        name.contains(expected),
                        "Expected {expected} in {name} for {pred}"
                    );
                }
                _ => panic!("Expected Filter for {pred}"),
            }
        }
    }
}
