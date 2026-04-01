// =============================================================================
// bikodb-query::cypher — Parser Cypher
// =============================================================================
// Subconjunto de openCypher (Neo4j query language):
//
//   MATCH (n:Type) RETURN n
//   MATCH (n:Type) WHERE n.prop > value RETURN n ORDER BY n.prop LIMIT 10
//   MATCH (a:Type)-[:EDGE]->(b) RETURN a, b
//   MATCH (a)-[:EDGE*1..3]->(b) RETURN b
//   MATCH (n:Type) RETURN count(n)
//   CREATE (n:Type { prop: value })
//   MATCH (n:Type) WHERE n.prop = val DELETE n
//   MATCH (n:Type) WHERE n.prop = val SET n.prop2 = val2 RETURN n
//
// ## Supported:
// MATCH, OPTIONAL MATCH, WHERE, RETURN, ORDER BY, LIMIT, CREATE, DELETE, SET,
// count() aggregation, variable-length paths *N..M
// =============================================================================

use bikodb_core::record::Direction;
use bikodb_core::types::{NodeId, TypeId};
use bikodb_core::value::Value;
use bikodb_execution::plan::{LogicalOp, Predicate};
use std::collections::HashMap;

/// Error de parsing Cypher.
#[derive(Debug, thiserror::Error)]
pub enum CypherParseError {
    #[error("Unexpected token at position {pos}: '{got}'")]
    UnexpectedToken { pos: usize, got: String },
    #[error("Unexpected end of input")]
    UnexpectedEof,
    #[error("Unknown label: '{0}'")]
    UnknownLabel(String),
}

/// Mapeo de labels a TypeId.
pub type LabelMap = HashMap<String, TypeId>;
/// Mapeo de nombres de propiedad a ID.
pub type PropMap = HashMap<String, u16>;
/// Mapeo de nombres de relación a TypeId.
pub type RelMap = HashMap<String, TypeId>;

/// CREATE node plan.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateNodePlan {
    pub label: Option<TypeId>,
    pub properties: Vec<(u16, Value)>,
}

/// DELETE plan.
#[derive(Debug, Clone)]
pub struct DeletePlan {
    pub scan: LogicalOp,
}

/// SET plan: mutation applied after MATCH.
#[derive(Debug, Clone)]
pub struct SetPlan {
    pub scan: LogicalOp,
    pub assignments: Vec<(u16, Value)>,
}

/// Resultado unificado del parser Cypher.
#[derive(Debug)]
pub enum CypherPlan {
    /// Read query (MATCH...RETURN).
    Query(LogicalOp),
    /// CREATE node.
    Create(CreateNodePlan),
    /// DELETE matched nodes.
    Delete(DeletePlan),
    /// SET properties on matched nodes.
    Set(SetPlan),
}

/// Parsea una query Cypher y produce un LogicalOp (solo MATCH...RETURN).
///
/// # Ejemplo
/// ```
/// use bikodb_query::cypher::{parse_cypher, LabelMap, PropMap, RelMap};
/// use bikodb_core::types::TypeId;
/// use std::collections::HashMap;
///
/// let mut labels = LabelMap::new();
/// labels.insert("Person".into(), TypeId(1));
///
/// let props = PropMap::new();
/// let rels = RelMap::new();
///
/// let plan = parse_cypher("MATCH (n:Person) RETURN n", &labels, &props, &rels).unwrap();
/// ```
pub fn parse_cypher(
    query: &str,
    label_map: &LabelMap,
    prop_map: &PropMap,
    rel_map: &RelMap,
) -> Result<LogicalOp, CypherParseError> {
    let mut parser = CypherParser::new(query, label_map, prop_map, rel_map);
    parser.parse()
}

/// Parsea una query Cypher completa (MATCH, CREATE, DELETE, SET).
pub fn parse_cypher_full(
    query: &str,
    label_map: &LabelMap,
    prop_map: &PropMap,
    rel_map: &RelMap,
) -> Result<CypherPlan, CypherParseError> {
    let mut parser = CypherParser::new(query, label_map, prop_map, rel_map);
    parser.parse_full()
}

/// Patrón de nodo en MATCH.
#[derive(Debug)]
struct NodePattern {
    /// Variable alias (e.g. "n" in (n:Person))
    variable: String,
    /// Optional label / type
    label: Option<String>,
}

/// Patrón de relación en MATCH.
#[derive(Debug)]
struct RelPattern {
    /// Tipo de relación (e.g. "KNOWS")
    rel_type: Option<String>,
    /// Dirección
    direction: Direction,
    /// Variable-length path: min..max hops. None = exactly 1 hop.
    var_length: Option<(usize, usize)>,
}

struct CypherParser<'a> {
    input: &'a str,
    pos: usize,
    label_map: &'a LabelMap,
    prop_map: &'a PropMap,
    rel_map: &'a RelMap,
}

impl<'a> CypherParser<'a> {
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

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len()
            && self.input.as_bytes()[self.pos].is_ascii_whitespace()
        {
            self.pos += 1;
        }
    }

    fn peek_word(&self) -> Option<&str> {
        let bytes = self.input.as_bytes();
        let mut i = self.pos;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let start = i;
        while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
            i += 1;
        }
        if start == i {
            None
        } else {
            Some(&self.input[start..i])
        }
    }

    fn consume_word(&mut self) -> Result<String, CypherParseError> {
        self.skip_whitespace();
        let start = self.pos;
        let bytes = self.input.as_bytes();
        while self.pos < bytes.len()
            && (bytes[self.pos].is_ascii_alphanumeric() || bytes[self.pos] == b'_')
        {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(CypherParseError::UnexpectedEof);
        }
        Ok(self.input[start..self.pos].to_string())
    }

    fn expect_char(&mut self, ch: char) -> Result<(), CypherParseError> {
        self.skip_whitespace();
        if self.pos < self.input.len() && self.input.as_bytes()[self.pos] == ch as u8 {
            self.pos += 1;
            Ok(())
        } else {
            Err(CypherParseError::UnexpectedToken {
                pos: self.pos,
                got: if self.pos < self.input.len() {
                    self.input[self.pos..self.pos + 1].to_string()
                } else {
                    "EOF".into()
                },
            })
        }
    }

    fn maybe_char(&mut self, ch: char) -> bool {
        self.skip_whitespace();
        if self.pos < self.input.len() && self.input.as_bytes()[self.pos] == ch as u8 {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn has_more(&self) -> bool {
        let mut i = self.pos;
        let bytes = self.input.as_bytes();
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        i < bytes.len()
    }

    /// Parse: MATCH pattern [WHERE cond] RETURN vars [ORDER BY] [LIMIT n]
    fn parse(&mut self) -> Result<LogicalOp, CypherParseError> {
        self.skip_whitespace();
        let keyword = self.consume_word()?;
        let kw_upper = keyword.to_uppercase();

        if kw_upper != "MATCH" && kw_upper != "OPTIONAL" {
            return Err(CypherParseError::UnexpectedToken {
                pos: 0,
                got: keyword,
            });
        }

        // OPTIONAL MATCH
        if kw_upper == "OPTIONAL" {
            self.skip_whitespace();
            let next = self.consume_word()?;
            if next.to_uppercase() != "MATCH" {
                return Err(CypherParseError::UnexpectedToken {
                    pos: self.pos,
                    got: next,
                });
            }
            // We parse the same as MATCH (OPTIONAL semantics handled at execution)
        }

        let plan = self.parse_match_body()?;
        Ok(plan)
    }

    /// Parse full Cypher: MATCH, CREATE, DELETE, SET.
    fn parse_full(&mut self) -> Result<CypherPlan, CypherParseError> {
        self.skip_whitespace();
        let keyword = self.consume_word()?;
        let kw_upper = keyword.to_uppercase();

        match kw_upper.as_str() {
            "MATCH" | "OPTIONAL" => {
                if kw_upper == "OPTIONAL" {
                    self.skip_whitespace();
                    let next = self.consume_word()?;
                    if next.to_uppercase() != "MATCH" {
                        return Err(CypherParseError::UnexpectedToken {
                            pos: self.pos,
                            got: next,
                        });
                    }
                }

                let plan = self.parse_match_body_full()?;
                Ok(plan)
            }
            "CREATE" => {
                let create = self.parse_create()?;
                Ok(CypherPlan::Create(create))
            }
            _ => Err(CypherParseError::UnexpectedToken {
                pos: 0,
                got: keyword,
            }),
        }
    }

    /// MATCH body: pattern [WHERE] then RETURN/DELETE/SET
    fn parse_match_body(&mut self) -> Result<LogicalOp, CypherParseError> {
        // Parse match pattern
        let (start_node, rel, end_node) = self.parse_pattern()?;

        let mut plan = self.build_scan_expand(&start_node, &rel, &end_node)?;

        // WHERE
        self.skip_whitespace();
        if let Some(word) = self.peek_word() {
            if word.eq_ignore_ascii_case("WHERE") {
                self.consume_word()?;
                let predicate = self.parse_where()?;
                plan = LogicalOp::Filter {
                    input: Box::new(plan),
                    predicate,
                };
            }
        }

        // RETURN
        self.skip_whitespace();
        let mut is_count = false;
        if let Some(word) = self.peek_word() {
            if word.eq_ignore_ascii_case("RETURN") {
                self.consume_word()?;
                let items = self.parse_return_items()?;
                // Check if RETURN count(...)
                if items.len() == 1 && items[0].to_lowercase().starts_with("count") {
                    is_count = true;
                }
            }
        }

        // ORDER BY
        self.skip_whitespace();
        if let Some(word) = self.peek_word() {
            if word.eq_ignore_ascii_case("ORDER") {
                self.consume_word()?; // ORDER
                self.skip_whitespace();
                let by = self.consume_word()?; // BY
                if by.to_uppercase() != "BY" {
                    return Err(CypherParseError::UnexpectedToken {
                        pos: self.pos,
                        got: by,
                    });
                }
                // Parse var.prop or just prop
                let first = self.consume_word()?;
                self.skip_whitespace();
                let prop_name = if self.pos < self.input.len()
                    && self.input.as_bytes()[self.pos] == b'.'
                {
                    self.pos += 1; // consume dot
                    self.consume_word()?
                } else {
                    first
                };
                let prop_id = self.prop_map.get(&prop_name).copied().unwrap_or(0);

                let ascending = self.skip_whitespace_and_check_direction();

                plan = LogicalOp::OrderBy {
                    input: Box::new(plan),
                    property_id: prop_id,
                    ascending,
                };
            }
        }

        // LIMIT
        self.skip_whitespace();
        if let Some(word) = self.peek_word() {
            if word.eq_ignore_ascii_case("LIMIT") {
                self.consume_word()?;
                let count_str = self.consume_word()?;
                let count = count_str
                    .parse::<usize>()
                    .map_err(|_| CypherParseError::UnexpectedToken {
                        pos: self.pos,
                        got: count_str,
                    })?;
                plan = LogicalOp::Limit {
                    input: Box::new(plan),
                    count,
                };
            }
        }

        if is_count {
            plan = LogicalOp::Count {
                input: Box::new(plan),
            };
        }

        Ok(plan)
    }

    /// MATCH body for full parse: can produce DELETE/SET plans.
    fn parse_match_body_full(&mut self) -> Result<CypherPlan, CypherParseError> {
        let (start_node, rel, end_node) = self.parse_pattern()?;
        let mut plan = self.build_scan_expand(&start_node, &rel, &end_node)?;

        // WHERE
        self.skip_whitespace();
        if let Some(word) = self.peek_word() {
            if word.eq_ignore_ascii_case("WHERE") {
                self.consume_word()?;
                let predicate = self.parse_where()?;
                plan = LogicalOp::Filter {
                    input: Box::new(plan),
                    predicate,
                };
            }
        }

        // Next keyword: RETURN, DELETE, or SET
        self.skip_whitespace();
        if let Some(word) = self.peek_word() {
            let upper = word.to_uppercase();
            match upper.as_str() {
                "DELETE" => {
                    self.consume_word()?;
                    // consume variable name
                    if self.has_more() {
                        self.consume_word()?;
                    }
                    return Ok(CypherPlan::Delete(DeletePlan { scan: plan }));
                }
                "SET" => {
                    self.consume_word()?;
                    let assignments = self.parse_set_assignments()?;

                    // Check if RETURN follows
                    self.skip_whitespace();
                    let mut return_plan = plan.clone();
                    if let Some(w) = self.peek_word() {
                        if w.eq_ignore_ascii_case("RETURN") {
                            self.consume_word()?;
                            self.parse_return_items()?;
                        }
                    }

                    return Ok(CypherPlan::Set(SetPlan {
                        scan: plan,
                        assignments,
                    }));
                }
                "RETURN" => {
                    self.consume_word()?;
                    let items = self.parse_return_items()?;
                    let mut is_count = false;
                    if items.len() == 1 && items[0].to_lowercase().starts_with("count") {
                        is_count = true;
                    }

                    // ORDER BY
                    self.skip_whitespace();
                    if let Some(w) = self.peek_word() {
                        if w.eq_ignore_ascii_case("ORDER") {
                            self.consume_word()?;
                            self.skip_whitespace();
                            let by = self.consume_word()?;
                            if by.to_uppercase() != "BY" {
                                return Err(CypherParseError::UnexpectedToken {
                                    pos: self.pos,
                                    got: by,
                                });
                            }
                            let first = self.consume_word()?;
                            self.skip_whitespace();
                            let prop_name = if self.pos < self.input.len()
                                && self.input.as_bytes()[self.pos] == b'.'
                            {
                                self.pos += 1;
                                self.consume_word()?
                            } else {
                                first
                            };
                            let prop_id = self.prop_map.get(&prop_name).copied().unwrap_or(0);
                            let ascending = self.skip_whitespace_and_check_direction();
                            plan = LogicalOp::OrderBy {
                                input: Box::new(plan),
                                property_id: prop_id,
                                ascending,
                            };
                        }
                    }

                    // LIMIT
                    self.skip_whitespace();
                    if let Some(w) = self.peek_word() {
                        if w.eq_ignore_ascii_case("LIMIT") {
                            self.consume_word()?;
                            let count_str = self.consume_word()?;
                            let count = count_str
                                .parse::<usize>()
                                .map_err(|_| CypherParseError::UnexpectedToken {
                                    pos: self.pos,
                                    got: count_str,
                                })?;
                            plan = LogicalOp::Limit {
                                input: Box::new(plan),
                                count,
                            };
                        }
                    }

                    if is_count {
                        plan = LogicalOp::Count {
                            input: Box::new(plan),
                        };
                    }
                    return Ok(CypherPlan::Query(plan));
                }
                _ => {}
            }
        }

        Ok(CypherPlan::Query(plan))
    }

    /// Build NodeScan + optional Expand from parsed pattern.
    fn build_scan_expand(
        &self,
        start_node: &NodePattern,
        rel: &Option<RelPattern>,
        end_node: &Option<NodePattern>,
    ) -> Result<LogicalOp, CypherParseError> {
        let start_type = match &start_node.label {
            Some(label) => self
                .label_map
                .get(label)
                .copied()
                .ok_or_else(|| CypherParseError::UnknownLabel(label.clone()))?,
            None => TypeId(0),
        };

        let mut plan = LogicalOp::NodeScan {
            type_id: start_type,
        };

        if let (Some(rel_pat), Some(_end)) = (rel, end_node) {
            let edge_type = rel_pat
                .rel_type
                .as_ref()
                .and_then(|rt| self.rel_map.get(rt).copied());

            // Variable-length paths: chain N Expands
            let hops = rel_pat.var_length.map(|(_, max)| max).unwrap_or(1);
            for _ in 0..hops {
                plan = LogicalOp::Expand {
                    input: Box::new(plan),
                    direction: rel_pat.direction,
                    edge_type,
                };
            }
        }

        Ok(plan)
    }

    fn skip_whitespace_and_check_direction(&mut self) -> bool {
        self.skip_whitespace();
        if let Some(word) = self.peek_word() {
            let upper = word.to_uppercase();
            if upper == "DESC" {
                self.consume_word().ok();
                return false;
            }
            if upper == "ASC" {
                self.consume_word().ok();
                return true;
            }
        }
        true // default ascending
    }

    /// Parse SET assignments: var.prop = value [, var.prop2 = value2 ...]
    fn parse_set_assignments(&mut self) -> Result<Vec<(u16, Value)>, CypherParseError> {
        let mut assignments = Vec::new();
        loop {
            self.skip_whitespace();
            // var.prop
            let _var = self.consume_word()?;
            self.expect_char('.')?;
            let prop_name = self.consume_word()?;
            let prop_id = self.prop_map.get(&prop_name).copied().unwrap_or(0);

            // =
            self.skip_whitespace();
            self.expect_char('=')?;

            // value
            let value = self.parse_value()?;
            assignments.push((prop_id, value));

            // Check for comma (more assignments)
            self.skip_whitespace();
            if self.pos < self.input.len() && self.input.as_bytes()[self.pos] == b',' {
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(assignments)
    }

    /// Parse CREATE (n:Label { prop: value, ... })
    fn parse_create(&mut self) -> Result<CreateNodePlan, CypherParseError> {
        self.expect_char('(')?;
        let _var = self.consume_word()?;

        let label = if self.maybe_char(':') {
            let label_name = self.consume_word()?;
            Some(
                self.label_map
                    .get(&label_name)
                    .copied()
                    .ok_or_else(|| CypherParseError::UnknownLabel(label_name))?,
            )
        } else {
            None
        };

        // Optional properties: { prop: value, ... }
        let mut properties = Vec::new();
        self.skip_whitespace();
        if self.pos < self.input.len() && self.input.as_bytes()[self.pos] == b'{' {
            self.pos += 1;
            loop {
                self.skip_whitespace();
                if self.pos < self.input.len() && self.input.as_bytes()[self.pos] == b'}' {
                    self.pos += 1;
                    break;
                }
                let prop_name = self.consume_word()?;
                let prop_id = self.prop_map.get(&prop_name).copied().unwrap_or(0);
                self.expect_char(':')?;
                let value = self.parse_value()?;
                properties.push((prop_id, value));

                self.skip_whitespace();
                if self.pos < self.input.len() && self.input.as_bytes()[self.pos] == b',' {
                    self.pos += 1;
                }
            }
        }

        self.expect_char(')')?;
        Ok(CreateNodePlan { label, properties })
    }

    /// Parse: (var:Label) or (a:Label)-[:REL]->(b:Label)
    fn parse_pattern(
        &mut self,
    ) -> Result<(NodePattern, Option<RelPattern>, Option<NodePattern>), CypherParseError> {
        let start = self.parse_node_pattern()?;

        self.skip_whitespace();
        // Check for relationship pattern
        if self.pos < self.input.len() && self.input.as_bytes()[self.pos] == b'-' {
            let (rel, end) = self.parse_rel_and_end_node()?;
            Ok((start, Some(rel), Some(end)))
        } else {
            Ok((start, None, None))
        }
    }

    /// Parse: (var:Label) or (var)
    fn parse_node_pattern(&mut self) -> Result<NodePattern, CypherParseError> {
        self.expect_char('(')?;

        let variable = self.consume_word()?;

        let label = if self.maybe_char(':') {
            Some(self.consume_word()?)
        } else {
            None
        };

        self.expect_char(')')?;

        Ok(NodePattern { variable, label })
    }

    /// Parse: -[:REL]->(node) or <-[:REL]-(node) or -[:REL]-(node)
    fn parse_rel_and_end_node(
        &mut self,
    ) -> Result<(RelPattern, NodePattern), CypherParseError> {
        self.skip_whitespace();

        // Determine direction from arrow syntax
        // -[:REL]-> or -[:REL]- or <-[:REL]-
        let mut left_arrow = false;

        if self.maybe_char('<') {
            left_arrow = true;
        }

        self.expect_char('-')?;

        // Optional [:REL] or [:REL*1..3]
        let mut rel_type = None;
        let mut var_length = None;
        if self.maybe_char('[') {
            self.maybe_char(':');
            // Try to read relationship type
            if let Some(b) = {
                self.skip_whitespace();
                if self.pos < self.input.len() {
                    Some(self.input.as_bytes()[self.pos])
                } else {
                    None
                }
            } {
                if b != b'*' && b != b']' {
                    rel_type = Some(self.consume_word()?);
                }
            }

            // Check for *min..max
            if self.pos < self.input.len() && self.input.as_bytes()[self.pos] == b'*' {
                self.pos += 1; // consume *
                let min_str = self.consume_word()?;
                let min: usize = min_str.parse().unwrap_or(1);
                // expect ".."
                if self.pos + 1 < self.input.len()
                    && self.input.as_bytes()[self.pos] == b'.'
                    && self.input.as_bytes()[self.pos + 1] == b'.'
                {
                    self.pos += 2;
                    let max_str = self.consume_word()?;
                    let max: usize = max_str.parse().unwrap_or(min);
                    var_length = Some((min, max));
                } else {
                    var_length = Some((min, min));
                }
            }

            self.expect_char(']')?;
        }

        self.expect_char('-')?;
        let right_arrow = self.maybe_char('>');

        let direction = match (left_arrow, right_arrow) {
            (false, true) => Direction::Out,
            (true, false) => Direction::In,
            _ => Direction::Both,
        };

        let end_node = self.parse_node_pattern()?;

        Ok((RelPattern { rel_type, direction, var_length }, end_node))
    }

    /// Parse WHERE condition: var.prop op value
    fn parse_where(&mut self) -> Result<Predicate, CypherParseError> {
        self.skip_whitespace();

        // Parse var.prop
        let _var = self.consume_word()?;
        self.expect_char('.')?;
        let prop_name = self.consume_word()?;
        let prop_id = self.prop_map.get(&prop_name).copied().unwrap_or(0);

        self.skip_whitespace();

        // Parse operator
        let bytes = self.input.as_bytes();
        let op_start = self.pos;
        let op = if self.pos + 1 < bytes.len() && bytes[self.pos] == b'>' && bytes[self.pos + 1] == b'=' {
            self.pos += 2;
            ">="
        } else if self.pos + 1 < bytes.len() && bytes[self.pos] == b'<' && bytes[self.pos + 1] == b'=' {
            self.pos += 2;
            "<="
        } else if self.pos + 1 < bytes.len() && bytes[self.pos] == b'!' && bytes[self.pos + 1] == b'=' {
            self.pos += 2;
            "!="
        } else if bytes[self.pos] == b'=' {
            self.pos += 1;
            "="
        } else if bytes[self.pos] == b'>' {
            self.pos += 1;
            ">"
        } else if bytes[self.pos] == b'<' {
            self.pos += 1;
            "<"
        } else {
            return Err(CypherParseError::UnexpectedToken {
                pos: op_start,
                got: self.input[op_start..op_start + 1].to_string(),
            });
        };

        // Parse value
        self.skip_whitespace();
        let value = self.parse_value()?;

        let predicate = match op {
            "=" => Predicate::Eq {
                property_id: prop_id,
                value,
            },
            "!=" => Predicate::Neq {
                property_id: prop_id,
                value,
            },
            ">" => Predicate::Gt {
                property_id: prop_id,
                value,
            },
            "<" => Predicate::Lt {
                property_id: prop_id,
                value,
            },
            ">=" => Predicate::Gte {
                property_id: prop_id,
                value,
            },
            "<=" => Predicate::Lte {
                property_id: prop_id,
                value,
            },
            _ => unreachable!(),
        };

        // AND / OR
        self.skip_whitespace();
        if let Some(word) = self.peek_word() {
            if word.eq_ignore_ascii_case("AND") {
                self.consume_word()?;
                let right = self.parse_where()?;
                return Ok(Predicate::And(Box::new(predicate), Box::new(right)));
            }
            if word.eq_ignore_ascii_case("OR") {
                self.consume_word()?;
                let right = self.parse_where()?;
                return Ok(Predicate::Or(Box::new(predicate), Box::new(right)));
            }
        }

        Ok(predicate)
    }

    fn parse_value(&mut self) -> Result<Value, CypherParseError> {
        self.skip_whitespace();
        let bytes = self.input.as_bytes();

        if self.pos >= bytes.len() {
            return Err(CypherParseError::UnexpectedEof);
        }

        // String literal
        if bytes[self.pos] == b'\'' {
            self.pos += 1;
            let start = self.pos;
            while self.pos < bytes.len() && bytes[self.pos] != b'\'' {
                self.pos += 1;
            }
            let s = self.input[start..self.pos].to_string();
            if self.pos < bytes.len() {
                self.pos += 1; // closing quote
            }
            return Ok(Value::String(s));
        }

        // Number
        let start = self.pos;
        if bytes[self.pos] == b'-' {
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
            let s = &self.input[start..self.pos];
            let f = s.parse::<f64>().map_err(|_| CypherParseError::UnexpectedToken {
                pos: start,
                got: s.to_string(),
            })?;
            Ok(Value::Float(f))
        } else if self.pos > start {
            let s = &self.input[start..self.pos];
            let n = s.parse::<i64>().map_err(|_| CypherParseError::UnexpectedToken {
                pos: start,
                got: s.to_string(),
            })?;
            Ok(Value::Int(n))
        } else {
            // Try as boolean or null
            let word = self.consume_word()?;
            match word.to_uppercase().as_str() {
                "NULL" => Ok(Value::Null),
                "TRUE" => Ok(Value::Bool(true)),
                "FALSE" => Ok(Value::Bool(false)),
                _ => Err(CypherParseError::UnexpectedToken {
                    pos: start,
                    got: word,
                }),
            }
        }
    }

    fn parse_return_items(&mut self) -> Result<Vec<String>, CypherParseError> {
        let mut items = Vec::new();
        loop {
            self.skip_whitespace();
            if !self.has_more() {
                break;
            }
            match self.peek_word() {
                Some(w) if w.eq_ignore_ascii_case("LIMIT") => break,
                Some(_) => {
                    let item = self.consume_word()?;
                    items.push(item);
                    self.skip_whitespace();
                    self.maybe_char(',');
                }
                None => break,
            }
        }
        Ok(items)
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
    fn test_simple_match() {
        let (labels, props, rels) = test_maps();
        let plan = parse_cypher("MATCH (n:Person) RETURN n", &labels, &props, &rels).unwrap();

        match plan {
            LogicalOp::NodeScan { type_id } => assert_eq!(type_id, TypeId(1)),
            _ => panic!("Expected NodeScan, got {plan:?}"),
        }
    }

    #[test]
    fn test_match_with_where() {
        let (labels, props, rels) = test_maps();
        let plan = parse_cypher(
            "MATCH (n:Person) WHERE n.age > 25 RETURN n",
            &labels,
            &props,
            &rels,
        )
        .unwrap();

        match plan {
            LogicalOp::Filter { predicate, .. } => {
                assert!(matches!(predicate, Predicate::Gt { .. }));
            }
            _ => panic!("Expected Filter"),
        }
    }

    #[test]
    fn test_match_with_relationship() {
        let (labels, props, rels) = test_maps();
        let plan = parse_cypher(
            "MATCH (a:Person)-[:KNOWS]->(b) RETURN b",
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
    fn test_match_with_limit() {
        let (labels, props, rels) = test_maps();
        let plan = parse_cypher(
            "MATCH (n:Person) RETURN n LIMIT 5",
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
    fn test_match_where_string() {
        let (labels, props, rels) = test_maps();
        let plan = parse_cypher(
            "MATCH (n:Person) WHERE n.name = 'Alice' RETURN n",
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
    fn test_unknown_label() {
        let (labels, props, rels) = test_maps();
        let result = parse_cypher("MATCH (n:Unknown) RETURN n", &labels, &props, &rels);
        assert!(result.is_err());
    }
}
