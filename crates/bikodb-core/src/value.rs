// =============================================================================
// bikodb-core::value — Sistema de tipos dinámico para propiedades
// =============================================================================
// Cada propiedad de un nodo o arista puede contener un Value.
// Este enum cubre todos los tipos soportados, inspirado en:
//   - ArcadeDB Type enum (BOOLEAN, INTEGER, STRING, LIST, MAP, EMBEDDED, etc.)
//   - Neo4j property types
//
// Diseño:
//   - Los variantes pequeños (Null, Bool, Int, Float) caben en 16 bytes
//   - Strings y colecciones usan heap allocation (inevitable)
//   - Binary usa Vec<u8> para datos crudos (embeddings, blobs)
//   - Serde support para serialización JSON/bincode
// =============================================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Valor dinámico que puede almacenar cualquier propiedad de un nodo o arista.
///
/// # Tamaño en memoria
/// El enum ocupa 32 bytes en stack (tag + max variant size).
/// Para propiedades numéricas, no hay heap allocation.
/// Los variantes grandes (List, Map, Binary, Embedding) se almacenan en Box
/// para mantener el tamaño del enum compacto (~32 bytes).
///
/// # Ejemplo
/// ```
/// use bikodb_core::value::Value;
///
/// let name = Value::String("Alice".into());
/// let age = Value::Int(30);
/// let scores = Value::List(Box::new(vec![Value::Float(9.5), Value::Float(8.2)]));
/// let is_active = Value::Bool(true);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Ausencia de valor (equivalente a NULL en SQL)
    Null,

    /// Booleano: true / false
    Bool(bool),

    /// Entero con signo de 64 bits (cubre i8, i16, i32, i64)
    /// Se usa i64 como tipo canónico para evitar conversiones internas.
    Int(i64),

    /// Número de punto flotante de 64 bits (cubre f32, f64)
    Float(f64),

    /// Cadena de texto UTF-8
    String(String),

    /// Datos binarios crudos (embeddings, blobs, imágenes).
    /// Boxed para mantener el enum compacto.
    Binary(Box<Vec<u8>>),

    /// Lista ordenada de valores (heterogéneos).
    /// Boxed para mantener el enum compacto.
    List(Box<Vec<Value>>),

    /// Mapa clave-valor (propiedades anidadas, documentos embebidos).
    /// Boxed para mantener el enum compacto.
    Map(Box<HashMap<String, Value>>),

    /// Fecha y hora con zona horaria (almacenado como millis desde epoch)
    DateTime(i64),

    /// Referencia a otro registro por RID (como un LINK en ArcadeDB)
    Link(crate::types::RID),

    /// Vector de floats para embeddings de IA.
    /// Separado de List para acceso directo sin unwrap por elemento.
    /// Almacena float32 para compatibilidad con modelos ML.
    /// Boxed para mantener el enum compacto.
    Embedding(Box<Vec<f32>>),
}

impl Value {
    // ── Constructores de conveniencia ──────────────────────────────────

    /// Crea un Value::String desde cualquier tipo que implemente Into<String>.
    pub fn string(s: impl Into<String>) -> Self {
        Value::String(s.into())
    }

    /// Crea un Value::Embedding desde un slice de f32.
    pub fn embedding(v: &[f32]) -> Self {
        Value::Embedding(Box::new(v.to_vec()))
    }

    // ── Type checking ─────────────────────────────────────────────────

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self, Value::Int(_) | Value::Float(_))
    }

    // ── Getters tipados (retornan Option) ─────────────────────────────

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64), // Conversión implícita int → float
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_embedding(&self) -> Option<&[f32]> {
        match self {
            Value::Embedding(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(l) => Some(l.as_slice()),
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&HashMap<String, Value>> {
        match self {
            Value::Map(m) => Some(m.as_ref()),
            _ => None,
        }
    }

    pub fn as_map_mut(&mut self) -> Option<&mut HashMap<String, Value>> {
        match self {
            Value::Map(m) => Some(m.as_mut()),
            _ => None,
        }
    }

    // ── Dot-path resolution (for document nested fields) ──────────────

    /// Resolve a dot-path like `"address.city"` or `"tags.0"` within a nested
    /// `Value::Map` / `Value::List` hierarchy.
    ///
    /// Returns `None` if any segment is missing or the type doesn't support nesting.
    pub fn get_path(&self, path: &str) -> Option<&Value> {
        let mut current = self;
        for segment in path.split('.') {
            match current {
                Value::Map(m) => {
                    current = m.get(segment)?;
                }
                Value::List(l) => {
                    let idx: usize = segment.parse().ok()?;
                    current = l.get(idx)?;
                }
                _ => return None,
            }
        }
        Some(current)
    }

    /// Set a value at a dot-path, creating intermediate Maps as needed.
    ///
    /// Returns `Ok(())` if the path was set successfully, `Err(())` if a
    /// non-map intermediate was encountered.
    pub fn set_path(&mut self, path: &str, value: Value) -> Result<(), ()> {
        let segments: Vec<&str> = path.split('.').collect();
        if segments.is_empty() {
            return Err(());
        }
        let mut current = self;
        for &seg in &segments[..segments.len() - 1] {
            // Navigate or create intermediate maps
            match current {
                Value::Map(m) => {
                    if !m.contains_key(seg) {
                        m.insert(seg.to_string(), Value::Map(Box::default()));
                    }
                    current = m.get_mut(seg).unwrap();
                }
                _ => return Err(()),
            }
        }
        let last = segments[segments.len() - 1];
        match current {
            Value::Map(m) => {
                m.insert(last.to_string(), value);
                Ok(())
            }
            _ => Err(()),
        }
    }

    /// Remove a value at a dot-path. Returns the removed value if it existed.
    pub fn remove_path(&mut self, path: &str) -> Option<Value> {
        let segments: Vec<&str> = path.split('.').collect();
        if segments.is_empty() {
            return None;
        }
        let mut current = self;
        for &seg in &segments[..segments.len() - 1] {
            match current {
                Value::Map(m) => {
                    current = m.get_mut(seg)?;
                }
                _ => return None,
            }
        }
        let last = segments[segments.len() - 1];
        match current {
            Value::Map(m) => m.remove(last),
            _ => None,
        }
    }

    /// Estimated heap size in bytes (for memory tracking).
    pub fn estimated_heap_bytes(&self) -> usize {
        match self {
            Value::Null | Value::Bool(_) | Value::Int(_) | Value::Float(_)
            | Value::DateTime(_) | Value::Link(_) => 0,
            Value::String(s) => s.capacity(),
            Value::Binary(b) => std::mem::size_of::<Vec<u8>>() + b.capacity(),
            Value::List(l) => {
                std::mem::size_of::<Vec<Value>>()
                    + l.capacity() * std::mem::size_of::<Value>()
                    + l.iter().map(|v| v.estimated_heap_bytes()).sum::<usize>()
            }
            Value::Map(m) => {
                std::mem::size_of::<HashMap<String, Value>>()
                    + m.iter()
                        .map(|(k, v)| k.capacity() + std::mem::size_of::<Value>() + v.estimated_heap_bytes())
                        .sum::<usize>()
            }
            Value::Embedding(v) => std::mem::size_of::<Vec<f32>>() + v.capacity() * 4,
        }
    }

    /// Nombre del tipo para mensajes de error y debug.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "Null",
            Value::Bool(_) => "Bool",
            Value::Int(_) => "Int",
            Value::Float(_) => "Float",
            Value::String(_) => "String",
            Value::Binary(_) => "Binary",
            Value::List(_) => "List",
            Value::Map(_) => "Map",
            Value::DateTime(_) => "DateTime",
            Value::Link(_) => "Link",
            Value::Embedding(_) => "Embedding",
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "NULL"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::Float(fl) => write!(f, "{fl}"),
            Value::String(s) => write!(f, "\"{s}\""),
            Value::Binary(b) => write!(f, "<binary:{} bytes>", b.len()),
            Value::List(l) => write!(f, "[{} items]", l.len()),
            Value::Map(m) => write!(f, "{{{} entries}}", m.len()),
            Value::DateTime(ms) => write!(f, "DateTime({ms})"),
            Value::Link(rid) => write!(f, "→{rid}"),
            Value::Embedding(v) => write!(f, "<embedding:dim={}>", v.len()),
        }
    }
}

// ── Conversiones From para ergonomía ───────────────────────────────────────

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}
impl From<i32> for Value {
    fn from(i: i32) -> Self {
        Value::Int(i as i64)
    }
}
impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Int(i)
    }
}
impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Float(f)
    }
}
impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.to_string())
    }
}
impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_type_checking() {
        assert!(Value::Null.is_null());
        assert!(Value::Int(42).is_numeric());
        assert!(Value::Float(3.14).is_numeric());
        assert!(!Value::String("hello".into()).is_numeric());
    }

    #[test]
    fn test_value_getters() {
        assert_eq!(Value::Bool(true).as_bool(), Some(true));
        assert_eq!(Value::Int(99).as_int(), Some(99));
        assert_eq!(Value::Float(2.5).as_float(), Some(2.5));
        assert_eq!(Value::Int(10).as_float(), Some(10.0)); // int → float
        assert_eq!(Value::string("hi").as_str(), Some("hi"));
    }

    #[test]
    fn test_value_from_conversions() {
        let v: Value = 42.into();
        assert_eq!(v, Value::Int(42));

        let v: Value = "hello".into();
        assert_eq!(v, Value::String("hello".into()));

        let v: Value = true.into();
        assert_eq!(v, Value::Bool(true));
    }

    #[test]
    fn test_value_embedding() {
        let emb = Value::embedding(&[1.0, 2.0, 3.0]);
        assert_eq!(emb.as_embedding(), Some([1.0f32, 2.0, 3.0].as_slice()));
    }

    #[test]
    fn test_value_display() {
        assert_eq!(format!("{}", Value::Null), "NULL");
        assert_eq!(format!("{}", Value::Int(42)), "42");
        assert_eq!(format!("{}", Value::string("test")), "\"test\"");
    }

    #[test]
    fn test_value_size_is_reasonable() {
        // After boxing large variants, Value should be ≤ 32 bytes
        let size = std::mem::size_of::<Value>();
        assert!(size <= 32, "Value size is {} bytes, expected ≤ 32", size);
    }

    #[test]
    fn test_value_estimated_heap_bytes() {
        assert_eq!(Value::Null.estimated_heap_bytes(), 0);
        assert_eq!(Value::Int(42).estimated_heap_bytes(), 0);
        assert_eq!(Value::Bool(true).estimated_heap_bytes(), 0);

        // String has heap allocation
        let s = Value::String("hello world".to_string());
        assert!(s.estimated_heap_bytes() >= 11);

        // Embedding has box + vec allocation
        let emb = Value::embedding(&[1.0, 2.0, 3.0]);
        assert!(emb.estimated_heap_bytes() > 0);
    }

    // ── Dot-path tests ─────────────────────────────────────────────────

    fn make_nested_doc() -> Value {
        let mut address = HashMap::new();
        address.insert("city".into(), Value::string("NYC"));
        address.insert("zip".into(), Value::Int(10001));

        let mut root = HashMap::new();
        root.insert("name".into(), Value::string("Alice"));
        root.insert("age".into(), Value::Int(30));
        root.insert("address".into(), Value::Map(Box::new(address)));
        root.insert("tags".into(), Value::List(Box::new(vec![
            Value::string("rust"),
            Value::string("graph"),
        ])));
        Value::Map(Box::new(root))
    }

    #[test]
    fn test_get_path_flat() {
        let doc = make_nested_doc();
        assert_eq!(doc.get_path("name"), Some(&Value::string("Alice")));
        assert_eq!(doc.get_path("age"), Some(&Value::Int(30)));
    }

    #[test]
    fn test_get_path_nested() {
        let doc = make_nested_doc();
        assert_eq!(doc.get_path("address.city"), Some(&Value::string("NYC")));
        assert_eq!(doc.get_path("address.zip"), Some(&Value::Int(10001)));
    }

    #[test]
    fn test_get_path_list_index() {
        let doc = make_nested_doc();
        assert_eq!(doc.get_path("tags.0"), Some(&Value::string("rust")));
        assert_eq!(doc.get_path("tags.1"), Some(&Value::string("graph")));
        assert_eq!(doc.get_path("tags.2"), None);
    }

    #[test]
    fn test_get_path_missing() {
        let doc = make_nested_doc();
        assert_eq!(doc.get_path("nonexistent"), None);
        assert_eq!(doc.get_path("address.state"), None);
    }

    #[test]
    fn test_set_path_existing() {
        let mut doc = make_nested_doc();
        doc.set_path("address.city", Value::string("LA")).unwrap();
        assert_eq!(doc.get_path("address.city"), Some(&Value::string("LA")));
    }

    #[test]
    fn test_set_path_creates_intermediate() {
        let mut doc = make_nested_doc();
        doc.set_path("metadata.version", Value::Int(2)).unwrap();
        assert_eq!(doc.get_path("metadata.version"), Some(&Value::Int(2)));
    }

    #[test]
    fn test_remove_path() {
        let mut doc = make_nested_doc();
        let removed = doc.remove_path("address.zip");
        assert_eq!(removed, Some(Value::Int(10001)));
        assert_eq!(doc.get_path("address.zip"), None);
    }

    #[test]
    fn test_as_map_mut() {
        let mut doc = make_nested_doc();
        let map = doc.as_map_mut().unwrap();
        map.insert("extra".into(), Value::Bool(true));
        assert_eq!(doc.get_path("extra"), Some(&Value::Bool(true)));
    }
}
