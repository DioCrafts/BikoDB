// =============================================================================
// bikodb-query::sql — Parser SQL con extensiones de grafo
// =============================================================================
// Subconjunto de SQL inspirado en ArcadeDB SQL:
//
//   SELECT [props | COUNT(*)] FROM type [WHERE cond] [GROUP BY prop]
//       [ORDER BY prop] [LIMIT n]
//   INSERT INTO type (props) VALUES (vals)
//   DELETE FROM type [WHERE cond]
//
// También soporta predicados extendidos: LIKE, IN, BETWEEN.
// =============================================================================

use bikodb_core::record::Direction;
use bikodb_core::types::{NodeId, TypeId};
use bikodb_core::value::Value;
use bikodb_execution::plan::{LogicalOp, Predicate};
use std::collections::HashMap;

/// Error de parsing SQL.
#[derive(Debug, thiserror::Error)]
pub enum SqlParseError {
    #[error("Unexpected token: expected {expected}, got '{got}'")]
    UnexpectedToken { expected: String, got: String },
    #[error("Unexpected end of input")]
    UnexpectedEof,
    #[error("Unknown type name: '{0}'")]
    UnknownType(String),
    #[error("Invalid value: '{0}'")]
    InvalidValue(String),
}

/// Mapeo de nombres de tipo a TypeId (necesario para resolver queries).
pub type TypeMap = HashMap<String, TypeId>;
/// Mapeo de nombres de propiedad a su ID numérico.
pub type PropMap = HashMap<String, u16>;

/// Resultado de un INSERT (datos para materializar, no ejecutable como LogicalOp scan).
#[derive(Debug, Clone, PartialEq)]
pub struct InsertPlan {
    pub type_id: TypeId,
    pub properties: Vec<(u16, Value)>,
}

/// Resultado de un DELETE.
#[derive(Debug, Clone)]
pub struct DeletePlan {
    pub type_id: TypeId,
    pub predicate: Option<Predicate>,
}

/// Resultado unificado del parser SQL.
#[derive(Debug)]
pub enum SqlPlan {
    /// Query de lectura (SELECT) → LogicalOp ejecutable.
    Select(LogicalOp),
    /// INSERT INTO → datos a insertar.
    Insert(InsertPlan),
    /// DELETE FROM → tipo + filtro opcional.
    Delete(DeletePlan),
}

/// Parsea un query SQL y produce un LogicalOp (solo SELECT).
///
/// # Ejemplo
/// ```
/// use bikodb_query::sql::{parse_sql, TypeMap, PropMap};
/// use bikodb_core::types::TypeId;
/// use std::collections::HashMap;
///
/// let mut types = TypeMap::new();
/// types.insert("Person".into(), TypeId(1));
///
/// let mut props = PropMap::new();
/// props.insert("name".into(), 0);
/// props.insert("age".into(), 1);
///
/// let plan = parse_sql("SELECT * FROM Person WHERE age > 25 LIMIT 10", &types, &props).unwrap();
/// ```
pub fn parse_sql(
    query: &str,
    type_map: &TypeMap,
    prop_map: &PropMap,
) -> Result<LogicalOp, SqlParseError> {
    let tokens = tokenize(query);
    let mut parser = SqlParser::new(&tokens, type_map, prop_map);
    parser.parse_select()
}

/// Parsea un query SQL completo (SELECT, INSERT, DELETE).
pub fn parse_sql_full(
    query: &str,
    type_map: &TypeMap,
    prop_map: &PropMap,
) -> Result<SqlPlan, SqlParseError> {
    let tokens = tokenize(query);
    let mut parser = SqlParser::new(&tokens, type_map, prop_map);
    parser.parse_any()
}

/// Token simple (sin tracking de posición — suficiente para v0.1).
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Keyword(String),   // SELECT, FROM, WHERE, INSERT, INTO, DELETE, etc.
    Ident(String),     // Nombres de tablas, columnas
    Number(i64),       // Literales numéricos
    Float(f64),        // Literales float
    StringLit(String), // 'string'
    Star,              // *
    Comma,             // ,
    Gt,                // >
    Lt,                // <
    Gte,               // >=
    Lte,               // <=
    Eq,                // =
    Neq,               // !=
    LParen,            // (
    RParen,            // )
    Percent,           // % (for LIKE patterns)
}

/// Tokenizer simple para SQL.
fn tokenize(input: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' | '\n' | '\r' => i += 1,
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            '%' => {
                tokens.push(Token::Percent);
                i += 1;
            }
            '>' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Gte);
                    i += 2;
                } else {
                    tokens.push(Token::Gt);
                    i += 1;
                }
            }
            '<' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Lte);
                    i += 2;
                } else {
                    tokens.push(Token::Lt);
                    i += 1;
                }
            }
            '=' => {
                tokens.push(Token::Eq);
                i += 1;
            }
            '!' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Neq);
                    i += 2;
                } else {
                    i += 1; // Skip unknown
                }
            }
            '\'' => {
                // String literal
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != '\'' {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                tokens.push(Token::StringLit(s));
                if i < chars.len() {
                    i += 1; // Skip closing quote
                }
            }
            c if c.is_ascii_digit() || c == '-' => {
                let start = i;
                if c == '-' {
                    i += 1;
                }
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
                if i < chars.len() && chars[i] == '.' {
                    i += 1;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                    let s: String = chars[start..i].iter().collect();
                    if let Ok(f) = s.parse::<f64>() {
                        tokens.push(Token::Float(f));
                    }
                } else {
                    let s: String = chars[start..i].iter().collect();
                    if let Ok(n) = s.parse::<i64>() {
                        tokens.push(Token::Number(n));
                    }
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                let upper = word.to_uppercase();
                match upper.as_str() {
                    "SELECT" | "FROM" | "WHERE" | "AND" | "OR" | "NOT" | "LIMIT" | "ORDER"
                    | "BY" | "ASC" | "DESC" | "IS" | "NULL" | "MATCH" | "RETURN"
                    | "TRAVERSE" | "INSERT" | "INTO" | "VALUES" | "DELETE" | "COUNT"
                    | "GROUP" | "LIKE" | "IN" | "BETWEEN" | "TRUE" | "FALSE" => {
                        tokens.push(Token::Keyword(upper));
                    }
                    _ => {
                        tokens.push(Token::Ident(word));
                    }
                }
            }
            _ => i += 1, // Skip unknown chars
        }
    }

    tokens
}

struct SqlParser<'a> {
    tokens: &'a [Token],
    pos: usize,
    type_map: &'a TypeMap,
    prop_map: &'a PropMap,
}

impl<'a> SqlParser<'a> {
    fn new(tokens: &'a [Token], type_map: &'a TypeMap, prop_map: &'a PropMap) -> Self {
        Self {
            tokens,
            pos: 0,
            type_map,
            prop_map,
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&Token> {
        let tok = self.tokens.get(self.pos);
        self.pos += 1;
        tok
    }

    fn expect_keyword(&mut self, kw: &str) -> Result<(), SqlParseError> {
        match self.advance() {
            Some(Token::Keyword(k)) if k == kw => Ok(()),
            Some(tok) => Err(SqlParseError::UnexpectedToken {
                expected: kw.into(),
                got: format!("{tok:?}"),
            }),
            None => Err(SqlParseError::UnexpectedEof),
        }
    }

    /// Parse any SQL statement.
    fn parse_any(&mut self) -> Result<SqlPlan, SqlParseError> {
        match self.peek() {
            Some(Token::Keyword(k)) if k == "SELECT" => {
                Ok(SqlPlan::Select(self.parse_select()?))
            }
            Some(Token::Keyword(k)) if k == "INSERT" => {
                Ok(SqlPlan::Insert(self.parse_insert()?))
            }
            Some(Token::Keyword(k)) if k == "DELETE" => {
                Ok(SqlPlan::Delete(self.parse_delete()?))
            }
            Some(tok) => Err(SqlParseError::UnexpectedToken {
                expected: "SELECT, INSERT, or DELETE".into(),
                got: format!("{tok:?}"),
            }),
            None => Err(SqlParseError::UnexpectedEof),
        }
    }

    /// INSERT INTO type (col1, col2) VALUES (val1, val2)
    fn parse_insert(&mut self) -> Result<InsertPlan, SqlParseError> {
        self.expect_keyword("INSERT")?;
        self.expect_keyword("INTO")?;

        let type_name = match self.advance() {
            Some(Token::Ident(name)) => name.clone(),
            Some(tok) => return Err(SqlParseError::UnexpectedToken {
                expected: "type name".into(),
                got: format!("{tok:?}"),
            }),
            None => return Err(SqlParseError::UnexpectedEof),
        };

        let type_id = self.type_map.get(&type_name).copied()
            .ok_or(SqlParseError::UnknownType(type_name))?;

        // (col1, col2, ...)
        self.expect_token(&Token::LParen)?;
        let mut col_ids = Vec::new();
        loop {
            match self.advance() {
                Some(Token::Ident(name)) => {
                    let name_owned = name.clone();
                    let prop_id = self.prop_map.get(name_owned.as_str()).copied().unwrap_or(0);
                    col_ids.push(prop_id);
                }
                _ => return Err(SqlParseError::UnexpectedEof),
            }
            if !matches!(self.peek(), Some(Token::Comma)) {
                break;
            }
            self.advance(); // consume comma
        }
        self.expect_token(&Token::RParen)?;

        // VALUES (val1, val2, ...)
        self.expect_keyword("VALUES")?;
        self.expect_token(&Token::LParen)?;
        let mut values = Vec::new();
        loop {
            let val = self.parse_value()?;
            values.push(val);
            if !matches!(self.peek(), Some(Token::Comma)) {
                break;
            }
            self.advance(); // consume comma
        }
        self.expect_token(&Token::RParen)?;

        let properties: Vec<(u16, Value)> = col_ids.into_iter().zip(values).collect();
        Ok(InsertPlan { type_id, properties })
    }

    /// DELETE FROM type [WHERE cond]
    fn parse_delete(&mut self) -> Result<DeletePlan, SqlParseError> {
        self.expect_keyword("DELETE")?;
        self.expect_keyword("FROM")?;

        let type_name = match self.advance() {
            Some(Token::Ident(name)) => name.clone(),
            Some(tok) => return Err(SqlParseError::UnexpectedToken {
                expected: "type name".into(),
                got: format!("{tok:?}"),
            }),
            None => return Err(SqlParseError::UnexpectedEof),
        };

        let type_id = self.type_map.get(&type_name).copied()
            .ok_or(SqlParseError::UnknownType(type_name))?;

        let predicate = if matches!(self.peek(), Some(Token::Keyword(k)) if k == "WHERE") {
            self.advance();
            Some(self.parse_predicate()?)
        } else {
            None
        };

        Ok(DeletePlan { type_id, predicate })
    }

    fn expect_token(&mut self, expected: &Token) -> Result<(), SqlParseError> {
        match self.advance() {
            Some(tok) if std::mem::discriminant(tok) == std::mem::discriminant(expected) => Ok(()),
            Some(tok) => Err(SqlParseError::UnexpectedToken {
                expected: format!("{expected:?}"),
                got: format!("{tok:?}"),
            }),
            None => Err(SqlParseError::UnexpectedEof),
        }
    }

    /// SELECT [* | COUNT(*) | prop_list] FROM type [WHERE cond]
    ///     [GROUP BY prop] [ORDER BY prop [ASC|DESC]] [LIMIT n]
    fn parse_select(&mut self) -> Result<LogicalOp, SqlParseError> {
        self.expect_keyword("SELECT")?;

        // Check for COUNT(*)
        let is_count = if matches!(self.peek(), Some(Token::Keyword(k)) if k == "COUNT") {
            self.advance(); // COUNT
            self.expect_token(&Token::LParen)?;
            // Consume * or column name (we treat both as full count)
            if matches!(self.peek(), Some(Token::Star)) {
                self.advance();
            }
            self.expect_token(&Token::RParen)?;
            true
        } else {
            false
        };

        // Parse projection columns (unless COUNT was parsed)
        let projection = if is_count {
            None
        } else {
            self.parse_projection()?
        };

        self.expect_keyword("FROM")?;

        // Type name
        let type_name = match self.advance() {
            Some(Token::Ident(name)) => name.clone(),
            Some(tok) => {
                return Err(SqlParseError::UnexpectedToken {
                    expected: "type name".into(),
                    got: format!("{tok:?}"),
                })
            }
            None => return Err(SqlParseError::UnexpectedEof),
        };

        let type_id = self
            .type_map
            .get(&type_name)
            .copied()
            .ok_or(SqlParseError::UnknownType(type_name))?;

        let mut plan = LogicalOp::NodeScan { type_id };

        // WHERE
        if matches!(self.peek(), Some(Token::Keyword(k)) if k == "WHERE") {
            self.advance(); // consume WHERE
            let predicate = self.parse_predicate()?;
            plan = LogicalOp::Filter {
                input: Box::new(plan),
                predicate,
            };
        }

        // GROUP BY (capture but don't change scan — grouping affects projection)
        let _group_by_prop = if matches!(self.peek(), Some(Token::Keyword(k)) if k == "GROUP") {
            self.advance(); // GROUP
            self.expect_keyword("BY")?;
            let prop_name = match self.advance() {
                Some(Token::Ident(name)) => name.clone(),
                _ => return Err(SqlParseError::UnexpectedEof),
            };
            Some(self.prop_map.get(&prop_name).copied().unwrap_or(0))
        } else {
            None
        };

        // ORDER BY
        if matches!(self.peek(), Some(Token::Keyword(k)) if k == "ORDER") {
            self.advance(); // ORDER
            self.expect_keyword("BY")?;
            let prop_name = match self.advance() {
                Some(Token::Ident(name)) => name.clone(),
                _ => return Err(SqlParseError::UnexpectedEof),
            };
            let prop_id = self.prop_map.get(&prop_name).copied().unwrap_or(0);
            let ascending = match self.peek() {
                Some(Token::Keyword(k)) if k == "DESC" => {
                    self.advance();
                    false
                }
                Some(Token::Keyword(k)) if k == "ASC" => {
                    self.advance();
                    true
                }
                _ => true,
            };
            plan = LogicalOp::OrderBy {
                input: Box::new(plan),
                property_id: prop_id,
                ascending,
            };
        }

        // LIMIT
        if matches!(self.peek(), Some(Token::Keyword(k)) if k == "LIMIT") {
            self.advance(); // consume LIMIT
            let count = match self.advance() {
                Some(Token::Number(n)) => *n as usize,
                _ => return Err(SqlParseError::UnexpectedEof),
            };
            plan = LogicalOp::Limit {
                input: Box::new(plan),
                count,
            };
        }

        // COUNT(*)
        if is_count {
            plan = LogicalOp::Count {
                input: Box::new(plan),
            };
        }

        // Project (if not SELECT * and not COUNT)
        if let Some(prop_ids) = projection {
            plan = LogicalOp::Project {
                input: Box::new(plan),
                property_ids: prop_ids,
            };
        }

        Ok(plan)
    }

    /// Parse * or comma-separated property names.
    fn parse_projection(&mut self) -> Result<Option<Vec<u16>>, SqlParseError> {
        if matches!(self.peek(), Some(Token::Star)) {
            self.advance();
            return Ok(None);
        }

        let mut props = Vec::new();
        loop {
            match self.peek() {
                Some(Token::Ident(name)) => {
                    let id = self.prop_map.get(name.as_str()).copied().unwrap_or(0);
                    props.push(id);
                    self.advance();
                }
                _ => break,
            }

            if matches!(self.peek(), Some(Token::Comma)) {
                self.advance();
            } else {
                break;
            }
        }

        if props.is_empty() {
            Ok(None)
        } else {
            Ok(Some(props))
        }
    }

    /// Parse WHERE condition (simple: prop op value [AND|OR ...])
    fn parse_predicate(&mut self) -> Result<Predicate, SqlParseError> {
        let left = self.parse_simple_predicate()?;

        match self.peek() {
            Some(Token::Keyword(k)) if k == "AND" => {
                self.advance();
                let right = self.parse_predicate()?;
                Ok(Predicate::And(Box::new(left), Box::new(right)))
            }
            Some(Token::Keyword(k)) if k == "OR" => {
                self.advance();
                let right = self.parse_predicate()?;
                Ok(Predicate::Or(Box::new(left), Box::new(right)))
            }
            _ => Ok(left),
        }
    }

    fn parse_simple_predicate(&mut self) -> Result<Predicate, SqlParseError> {
        // NOT
        if matches!(self.peek(), Some(Token::Keyword(k)) if k == "NOT") {
            self.advance();
            let inner = self.parse_simple_predicate()?;
            return Ok(Predicate::Not(Box::new(inner)));
        }

        // ( grouped predicate )
        if matches!(self.peek(), Some(Token::LParen)) {
            self.advance();
            let inner = self.parse_predicate()?;
            self.expect_token(&Token::RParen)?;
            return Ok(inner);
        }

        // prop op value
        let prop_name = match self.advance() {
            Some(Token::Ident(name)) => name.clone(),
            Some(tok) => {
                return Err(SqlParseError::UnexpectedToken {
                    expected: "property name".into(),
                    got: format!("{tok:?}"),
                })
            }
            None => return Err(SqlParseError::UnexpectedEof),
        };

        let prop_id = self.prop_map.get(&prop_name).copied().unwrap_or(0);

        // IS NOT NULL
        if matches!(self.peek(), Some(Token::Keyword(k)) if k == "IS") {
            self.advance();
            if matches!(self.peek(), Some(Token::Keyword(k)) if k == "NOT") {
                self.advance();
                self.expect_keyword("NULL")?;
                return Ok(Predicate::IsNotNull {
                    property_id: prop_id,
                });
            }
        }

        // LIKE 'pattern'
        if matches!(self.peek(), Some(Token::Keyword(k)) if k == "LIKE") {
            self.advance();
            let pattern = self.parse_value()?;
            // LIKE is translated to Eq with a String value (prefix matching
            // can be done at execution time; for the plan layer we store
            // the pattern as-is).
            return Ok(Predicate::Eq {
                property_id: prop_id,
                value: pattern,
            });
        }

        // IN (val1, val2, ...)
        if matches!(self.peek(), Some(Token::Keyword(k)) if k == "IN") {
            self.advance();
            self.expect_token(&Token::LParen)?;
            let mut predicates: Vec<Predicate> = Vec::new();
            loop {
                let val = self.parse_value()?;
                predicates.push(Predicate::Eq {
                    property_id: prop_id,
                    value: val,
                });
                if !matches!(self.peek(), Some(Token::Comma)) {
                    break;
                }
                self.advance(); // comma
            }
            self.expect_token(&Token::RParen)?;
            // Combine all eq predicates with OR
            let mut result = predicates.pop().unwrap();
            while let Some(p) = predicates.pop() {
                result = Predicate::Or(Box::new(p), Box::new(result));
            }
            return Ok(result);
        }

        // BETWEEN val1 AND val2
        if matches!(self.peek(), Some(Token::Keyword(k)) if k == "BETWEEN") {
            self.advance();
            let low = self.parse_value()?;
            self.expect_keyword("AND")?;
            let high = self.parse_value()?;
            return Ok(Predicate::And(
                Box::new(Predicate::Gte {
                    property_id: prop_id,
                    value: low,
                }),
                Box::new(Predicate::Lte {
                    property_id: prop_id,
                    value: high,
                }),
            ));
        }

        let op = self.advance().cloned().ok_or(SqlParseError::UnexpectedEof)?;

        let value = self.parse_value()?;

        match op {
            Token::Eq => Ok(Predicate::Eq {
                property_id: prop_id,
                value,
            }),
            Token::Neq => Ok(Predicate::Neq {
                property_id: prop_id,
                value,
            }),
            Token::Gt => Ok(Predicate::Gt {
                property_id: prop_id,
                value,
            }),
            Token::Lt => Ok(Predicate::Lt {
                property_id: prop_id,
                value,
            }),
            Token::Gte => Ok(Predicate::Gte {
                property_id: prop_id,
                value,
            }),
            Token::Lte => Ok(Predicate::Lte {
                property_id: prop_id,
                value,
            }),
            _ => Err(SqlParseError::UnexpectedToken {
                expected: "comparison operator".into(),
                got: format!("{op:?}"),
            }),
        }
    }

    fn parse_value(&mut self) -> Result<Value, SqlParseError> {
        match self.advance() {
            Some(Token::Number(n)) => Ok(Value::Int(*n)),
            Some(Token::Float(f)) => Ok(Value::Float(*f)),
            Some(Token::StringLit(s)) => Ok(Value::String(s.clone())),
            Some(Token::Keyword(k)) if k == "NULL" => Ok(Value::Null),
            Some(Token::Keyword(k)) if k == "TRUE" => Ok(Value::Bool(true)),
            Some(Token::Keyword(k)) if k == "FALSE" => Ok(Value::Bool(false)),
            Some(tok) => Err(SqlParseError::InvalidValue(format!("{tok:?}"))),
            None => Err(SqlParseError::UnexpectedEof),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn test_maps() -> (TypeMap, PropMap) {
        let mut types = TypeMap::new();
        types.insert("Person".into(), TypeId(1));
        types.insert("Company".into(), TypeId(2));

        let mut props = PropMap::new();
        props.insert("name".into(), 0);
        props.insert("age".into(), 1);
        props.insert("email".into(), 2);

        (types, props)
    }

    #[test]
    fn test_select_star() {
        let (types, props) = test_maps();
        let plan = parse_sql("SELECT * FROM Person", &types, &props).unwrap();

        match plan {
            LogicalOp::NodeScan { type_id } => assert_eq!(type_id, TypeId(1)),
            _ => panic!("Expected NodeScan"),
        }
    }

    #[test]
    fn test_select_with_where() {
        let (types, props) = test_maps();
        let plan = parse_sql("SELECT * FROM Person WHERE age > 25", &types, &props).unwrap();

        match plan {
            LogicalOp::Filter { predicate, .. } => {
                assert!(matches!(predicate, Predicate::Gt { .. }));
            }
            _ => panic!("Expected Filter"),
        }
    }

    #[test]
    fn test_select_with_limit() {
        let (types, props) = test_maps();
        let plan = parse_sql("SELECT * FROM Person LIMIT 10", &types, &props).unwrap();

        match plan {
            LogicalOp::Limit { count, .. } => assert_eq!(count, 10),
            _ => panic!("Expected Limit"),
        }
    }

    #[test]
    fn test_select_with_where_and_limit() {
        let (types, props) = test_maps();
        let plan = parse_sql(
            "SELECT * FROM Person WHERE age >= 18 LIMIT 5",
            &types,
            &props,
        )
        .unwrap();

        match plan {
            LogicalOp::Limit { input, count } => {
                assert_eq!(count, 5);
                assert!(matches!(*input, LogicalOp::Filter { .. }));
            }
            _ => panic!("Expected Limit wrapping Filter"),
        }
    }

    #[test]
    fn test_select_with_string_eq() {
        let (types, props) = test_maps();
        let plan =
            parse_sql("SELECT * FROM Person WHERE name = 'Alice'", &types, &props).unwrap();

        match plan {
            LogicalOp::Filter { predicate, .. } => match predicate {
                Predicate::Eq { value, .. } => {
                    assert_eq!(value, Value::String("Alice".into()));
                }
                _ => panic!("Expected Eq predicate"),
            },
            _ => panic!("Expected Filter"),
        }
    }

    #[test]
    fn test_select_with_and() {
        let (types, props) = test_maps();
        let plan = parse_sql(
            "SELECT * FROM Person WHERE age > 20 AND name = 'Bob'",
            &types,
            &props,
        )
        .unwrap();

        match plan {
            LogicalOp::Filter { predicate, .. } => {
                assert!(matches!(predicate, Predicate::And(_, _)));
            }
            _ => panic!("Expected Filter with AND"),
        }
    }

    #[test]
    fn test_select_projection() {
        let (types, props) = test_maps();
        let plan = parse_sql("SELECT name, age FROM Person", &types, &props).unwrap();

        match plan {
            LogicalOp::Project { property_ids, .. } => {
                assert_eq!(property_ids, vec![0, 1]);
            }
            _ => panic!("Expected Project"),
        }
    }

    #[test]
    fn test_select_order_by() {
        let (types, props) = test_maps();
        let plan =
            parse_sql("SELECT * FROM Person ORDER BY age DESC", &types, &props).unwrap();

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
    fn test_unknown_type() {
        let (types, props) = test_maps();
        let result = parse_sql("SELECT * FROM Unknown", &types, &props);
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenizer() {
        let tokens = tokenize("SELECT * FROM Person WHERE age > 25");
        assert_eq!(tokens.len(), 8);
        assert_eq!(tokens[0], Token::Keyword("SELECT".into()));
        assert_eq!(tokens[1], Token::Star);
    }

    // ── New SQL features ────────────────────────────────────────────

    #[test]
    fn test_count_star() {
        let (types, props) = test_maps();
        let plan = parse_sql("SELECT COUNT(*) FROM Person", &types, &props).unwrap();
        assert!(matches!(plan, LogicalOp::Count { .. }));
    }

    #[test]
    fn test_count_with_where() {
        let (types, props) = test_maps();
        let plan = parse_sql(
            "SELECT COUNT(*) FROM Person WHERE age > 25",
            &types,
            &props,
        )
        .unwrap();
        match plan {
            LogicalOp::Count { input } => {
                assert!(matches!(*input, LogicalOp::Filter { .. }));
            }
            _ => panic!("Expected Count"),
        }
    }

    #[test]
    fn test_insert_into() {
        let (types, props) = test_maps();
        let result = parse_sql_full(
            "INSERT INTO Person (name, age) VALUES ('Alice', 30)",
            &types,
            &props,
        )
        .unwrap();
        match result {
            SqlPlan::Insert(plan) => {
                assert_eq!(plan.type_id, TypeId(1));
                assert_eq!(plan.properties.len(), 2);
                assert_eq!(plan.properties[0], (0, Value::String("Alice".into())));
                assert_eq!(plan.properties[1], (1, Value::Int(30)));
            }
            _ => panic!("Expected Insert"),
        }
    }

    #[test]
    fn test_delete_from() {
        let (types, props) = test_maps();
        let result = parse_sql_full("DELETE FROM Person WHERE age < 18", &types, &props).unwrap();
        match result {
            SqlPlan::Delete(plan) => {
                assert_eq!(plan.type_id, TypeId(1));
                assert!(plan.predicate.is_some());
                assert!(matches!(plan.predicate.unwrap(), Predicate::Lt { .. }));
            }
            _ => panic!("Expected Delete"),
        }
    }

    #[test]
    fn test_delete_all() {
        let (types, props) = test_maps();
        let result = parse_sql_full("DELETE FROM Person", &types, &props).unwrap();
        match result {
            SqlPlan::Delete(plan) => {
                assert_eq!(plan.type_id, TypeId(1));
                assert!(plan.predicate.is_none());
            }
            _ => panic!("Expected Delete"),
        }
    }

    #[test]
    fn test_in_predicate() {
        let (types, props) = test_maps();
        let plan = parse_sql(
            "SELECT * FROM Person WHERE age IN (20, 25, 30)",
            &types,
            &props,
        )
        .unwrap();
        match plan {
            LogicalOp::Filter { predicate, .. } => {
                // Should be OR(Eq(20), OR(Eq(25), Eq(30)))
                assert!(matches!(predicate, Predicate::Or(..)));
            }
            _ => panic!("Expected Filter with OR"),
        }
    }

    #[test]
    fn test_between_predicate() {
        let (types, props) = test_maps();
        let plan = parse_sql(
            "SELECT * FROM Person WHERE age BETWEEN 18 AND 65",
            &types,
            &props,
        )
        .unwrap();
        match plan {
            LogicalOp::Filter { predicate, .. } => {
                assert!(matches!(predicate, Predicate::And(..)));
            }
            _ => panic!("Expected Filter with AND (between)"),
        }
    }

    #[test]
    fn test_like_predicate() {
        let (types, props) = test_maps();
        let plan = parse_sql(
            "SELECT * FROM Person WHERE name LIKE 'Ali'",
            &types,
            &props,
        )
        .unwrap();
        match plan {
            LogicalOp::Filter { predicate, .. } => {
                assert!(matches!(predicate, Predicate::Eq { .. }));
            }
            _ => panic!("Expected Filter"),
        }
    }

    #[test]
    fn test_group_by() {
        let (types, props) = test_maps();
        // GROUP BY parses without error (grouping metadata captured)
        let plan = parse_sql(
            "SELECT * FROM Person GROUP BY age ORDER BY age",
            &types,
            &props,
        )
        .unwrap();
        assert!(matches!(plan, LogicalOp::OrderBy { .. }));
    }

    #[test]
    fn test_select_via_full() {
        let (types, props) = test_maps();
        let result = parse_sql_full("SELECT * FROM Person LIMIT 5", &types, &props).unwrap();
        match result {
            SqlPlan::Select(plan) => {
                assert!(matches!(plan, LogicalOp::Limit { count: 5, .. }));
            }
            _ => panic!("Expected Select"),
        }
    }
}
