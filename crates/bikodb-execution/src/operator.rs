// =============================================================================
// bikodb-execution::operator — Operadores físicos de ejecución
// =============================================================================
// Modelo pull-based (Volcano): cada operador implementa `next()`.
//
// Los operadores procesan una fila (Row) a la vez, evitando materializar
// resultados intermedios grandes en memoria.
// =============================================================================

use crate::plan::Predicate;
use bikodb_core::types::NodeId;
use bikodb_core::value::Value;

/// Una fila de resultado: nodo + propiedades seleccionadas.
#[derive(Debug, Clone)]
pub struct Row {
    pub node_id: NodeId,
    pub properties: Vec<(u16, Value)>,
}

/// Trait para operadores pull-based.
///
/// Cada operador produce filas bajo demanda (`next()`).
/// El consumidor llama `next()` repetidamente hasta obtener `None`.
pub trait Operator {
    /// Produce la siguiente fila, o None si no hay más.
    fn next(&mut self) -> Option<Row>;

    /// Resetea el operador al inicio (para re-ejecución).
    fn reset(&mut self);
}

/// Operador que produce filas desde un Vec pre-computado.
///
/// Usado como fuente de datos después de un scan o lookup.
pub struct VecSource {
    rows: Vec<Row>,
    pos: usize,
}

impl VecSource {
    pub fn new(rows: Vec<Row>) -> Self {
        Self { rows, pos: 0 }
    }
}

impl Operator for VecSource {
    fn next(&mut self) -> Option<Row> {
        if self.pos < self.rows.len() {
            let row = self.rows[self.pos].clone();
            self.pos += 1;
            Some(row)
        } else {
            None
        }
    }

    fn reset(&mut self) {
        self.pos = 0;
    }
}

/// Operador de filtrado: pasa solo filas que cumplen el predicado.
pub struct FilterOp {
    input: Box<dyn Operator>,
    predicate: Predicate,
}

impl FilterOp {
    pub fn new(input: Box<dyn Operator>, predicate: Predicate) -> Self {
        Self { input, predicate }
    }
}

impl Operator for FilterOp {
    fn next(&mut self) -> Option<Row> {
        loop {
            let row = self.input.next()?;
            if self.predicate.evaluate(&row.properties) {
                return Some(row);
            }
        }
    }

    fn reset(&mut self) {
        self.input.reset();
    }
}

/// Operador de proyección: selecciona solo ciertas propiedades.
pub struct ProjectOp {
    input: Box<dyn Operator>,
    property_ids: Vec<u16>,
}

impl ProjectOp {
    pub fn new(input: Box<dyn Operator>, property_ids: Vec<u16>) -> Self {
        Self {
            input,
            property_ids,
        }
    }
}

impl Operator for ProjectOp {
    fn next(&mut self) -> Option<Row> {
        let row = self.input.next()?;
        let projected: Vec<(u16, Value)> = row
            .properties
            .into_iter()
            .filter(|(id, _)| self.property_ids.contains(id))
            .collect();

        Some(Row {
            node_id: row.node_id,
            properties: projected,
        })
    }

    fn reset(&mut self) {
        self.input.reset();
    }
}

/// Operador de límite: retorna como máximo N filas.
pub struct LimitOp {
    input: Box<dyn Operator>,
    max_count: usize,
    emitted: usize,
}

impl LimitOp {
    pub fn new(input: Box<dyn Operator>, max_count: usize) -> Self {
        Self {
            input,
            max_count,
            emitted: 0,
        }
    }
}

impl Operator for LimitOp {
    fn next(&mut self) -> Option<Row> {
        if self.emitted >= self.max_count {
            return None;
        }
        let row = self.input.next()?;
        self.emitted += 1;
        Some(row)
    }

    fn reset(&mut self) {
        self.input.reset();
        self.emitted = 0;
    }
}

/// Operador de conteo: consume todo el input y retorna una fila con el conteo.
pub struct CountOp {
    input: Box<dyn Operator>,
    done: bool,
}

impl CountOp {
    pub fn new(input: Box<dyn Operator>) -> Self {
        Self { input, done: false }
    }
}

impl Operator for CountOp {
    fn next(&mut self) -> Option<Row> {
        if self.done {
            return None;
        }
        self.done = true;

        let mut count = 0i64;
        while self.input.next().is_some() {
            count += 1;
        }

        Some(Row {
            node_id: NodeId(0),
            properties: vec![(0, Value::Int(count))],
        })
    }

    fn reset(&mut self) {
        self.input.reset();
        self.done = false;
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rows() -> Vec<Row> {
        vec![
            Row {
                node_id: NodeId(1),
                properties: vec![(0, Value::from("Alice")), (1, Value::Int(30))],
            },
            Row {
                node_id: NodeId(2),
                properties: vec![(0, Value::from("Bob")), (1, Value::Int(25))],
            },
            Row {
                node_id: NodeId(3),
                properties: vec![(0, Value::from("Charlie")), (1, Value::Int(35))],
            },
        ]
    }

    #[test]
    fn test_vec_source() {
        let mut src = VecSource::new(sample_rows());
        assert!(src.next().is_some());
        assert!(src.next().is_some());
        assert!(src.next().is_some());
        assert!(src.next().is_none());
    }

    #[test]
    fn test_filter_op() {
        let src = Box::new(VecSource::new(sample_rows()));
        let pred = Predicate::Gt {
            property_id: 1,
            value: Value::Int(28),
        };
        let mut filter = FilterOp::new(src, pred);

        let r1 = filter.next().unwrap();
        assert_eq!(r1.node_id, NodeId(1)); // Alice, age 30

        let r2 = filter.next().unwrap();
        assert_eq!(r2.node_id, NodeId(3)); // Charlie, age 35

        assert!(filter.next().is_none());
    }

    #[test]
    fn test_project_op() {
        let src = Box::new(VecSource::new(sample_rows()));
        let mut proj = ProjectOp::new(src, vec![0]); // Solo prop 0 (name)

        let row = proj.next().unwrap();
        assert_eq!(row.properties.len(), 1);
        assert_eq!(row.properties[0].0, 0);
    }

    #[test]
    fn test_limit_op() {
        let src = Box::new(VecSource::new(sample_rows()));
        let mut limit = LimitOp::new(src, 2);

        assert!(limit.next().is_some());
        assert!(limit.next().is_some());
        assert!(limit.next().is_none());
    }

    #[test]
    fn test_count_op() {
        let src = Box::new(VecSource::new(sample_rows()));
        let mut count = CountOp::new(src);

        let row = count.next().unwrap();
        assert_eq!(row.properties[0].1, Value::Int(3));
        assert!(count.next().is_none());
    }

    #[test]
    fn test_pipeline_composition() {
        // Filter age > 25 → Limit 1
        let src = Box::new(VecSource::new(sample_rows()));
        let filter = Box::new(FilterOp::new(
            src,
            Predicate::Gt {
                property_id: 1,
                value: Value::Int(25),
            },
        ));
        let mut limit = LimitOp::new(filter, 1);

        let row = limit.next().unwrap();
        assert_eq!(row.node_id, NodeId(1)); // Alice
        assert!(limit.next().is_none());
    }
}
