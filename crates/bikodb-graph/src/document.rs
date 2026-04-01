// =============================================================================
// bikodb-graph::document — Document Store (multi-modelo)
// =============================================================================
// Almacén de documentos genéricos (JSON-like) independiente del grafo.
// Cada documento es un `Value::Map` con un ID único.
//
// Permite:
//   - Colecciones con nombre ("users", "products", etc.)
//   - CRUD completo por DocumentId
//   - Queries con filtros en campos anidados (dot-path)
//   - Proyecciones parciales
//   - Nodos del grafo pueden linkear a documentos y viceversa
//
// Diseño: DashMap<DocumentId, Value::Map> por colección, thread-safe.
// =============================================================================

use dashmap::DashMap;
use bikodb_core::types::NodeId;
use bikodb_core::value::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// ID único de un documento.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DocumentId(pub u64);

impl std::fmt::Display for DocumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "doc:{}", self.0)
    }
}

/// Un documento almacenado: mapa de campos + metadatos.
#[derive(Debug, Clone)]
pub struct Document {
    pub id: DocumentId,
    /// Campos del documento (un Value::Map).
    pub fields: HashMap<String, Value>,
    /// Nodo del grafo vinculado (si existe).
    pub linked_node: Option<NodeId>,
}

impl Document {
    /// Obtiene un campo por dot-path.
    pub fn get_field(&self, path: &str) -> Option<&Value> {
        let segments: Vec<&str> = path.split('.').collect();
        if segments.is_empty() {
            return None;
        }
        // First segment: lookup in top-level fields
        let mut current = self.fields.get(segments[0])?;
        // Remaining segments: dive into nested Values
        for &seg in &segments[1..] {
            match current {
                Value::Map(m) => {
                    current = m.get(seg)?;
                }
                Value::List(l) => {
                    let idx: usize = seg.parse().ok()?;
                    current = l.get(idx)?;
                }
                _ => return None,
            }
        }
        Some(current)
    }
}

/// Operador de comparación para filtros de documentos.
#[derive(Debug, Clone)]
pub enum DocFilter {
    /// Campo == valor
    Eq { path: String, value: Value },
    /// Campo != valor
    Neq { path: String, value: Value },
    /// Campo > valor
    Gt { path: String, value: Value },
    /// Campo < valor
    Lt { path: String, value: Value },
    /// Campo >= valor
    Gte { path: String, value: Value },
    /// Campo <= valor
    Lte { path: String, value: Value },
    /// Campo existe (IS NOT NULL)
    Exists { path: String },
    /// Campo contiene valor (para listas)
    Contains { path: String, value: Value },
    /// AND de filtros
    And(Vec<DocFilter>),
    /// OR de filtros
    Or(Vec<DocFilter>),
    /// NOT de un filtro
    Not(Box<DocFilter>),
}

impl DocFilter {
    /// Evalúa el filtro contra un documento.
    pub fn matches(&self, doc: &Document) -> bool {
        match self {
            DocFilter::Eq { path, value } => {
                doc.get_field(path).map(|v| v == value).unwrap_or(false)
            }
            DocFilter::Neq { path, value } => {
                doc.get_field(path).map(|v| v != value).unwrap_or(true)
            }
            DocFilter::Gt { path, value } => doc_compare(doc, path, value, |o| {
                matches!(o, std::cmp::Ordering::Greater)
            }),
            DocFilter::Lt { path, value } => doc_compare(doc, path, value, |o| {
                matches!(o, std::cmp::Ordering::Less)
            }),
            DocFilter::Gte { path, value } => doc_compare(doc, path, value, |o| {
                matches!(o, std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
            }),
            DocFilter::Lte { path, value } => doc_compare(doc, path, value, |o| {
                matches!(o, std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            }),
            DocFilter::Exists { path } => doc.get_field(path).is_some(),
            DocFilter::Contains { path, value } => {
                if let Some(Value::List(list)) = doc.get_field(path) {
                    list.contains(value)
                } else {
                    false
                }
            }
            DocFilter::And(filters) => filters.iter().all(|f| f.matches(doc)),
            DocFilter::Or(filters) => filters.iter().any(|f| f.matches(doc)),
            DocFilter::Not(f) => !f.matches(doc),
        }
    }
}

fn doc_compare(
    doc: &Document,
    path: &str,
    target: &Value,
    cmp_fn: impl Fn(std::cmp::Ordering) -> bool,
) -> bool {
    match doc.get_field(path) {
        Some(Value::Int(a)) => {
            if let Value::Int(b) = target { cmp_fn(a.cmp(b)) } else { false }
        }
        Some(Value::Float(a)) => {
            if let Value::Float(b) = target {
                a.partial_cmp(b).map(&cmp_fn).unwrap_or(false)
            } else {
                false
            }
        }
        Some(Value::String(a)) => {
            if let Value::String(b) = target { cmp_fn(a.cmp(b)) } else { false }
        }
        _ => false,
    }
}

/// Una colección de documentos (thread-safe).
pub struct DocumentCollection {
    name: String,
    docs: DashMap<DocumentId, Document>,
    next_id: AtomicU64,
}

impl DocumentCollection {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            docs: DashMap::new(),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Inserta un documento con campos dados, retorna su ID.
    pub fn insert(&self, fields: HashMap<String, Value>) -> DocumentId {
        let id = DocumentId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let doc = Document { id, fields, linked_node: None };
        self.docs.insert(id, doc);
        id
    }

    /// Inserta un documento vinculado a un nodo del grafo.
    pub fn insert_linked(&self, fields: HashMap<String, Value>, node_id: NodeId) -> DocumentId {
        let id = DocumentId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let doc = Document { id, fields, linked_node: Some(node_id) };
        self.docs.insert(id, doc);
        id
    }

    /// Obtiene un documento por ID.
    pub fn get(&self, id: DocumentId) -> Option<Document> {
        self.docs.get(&id).map(|r| r.clone())
    }

    /// Actualiza campos de un documento (merge).
    pub fn update(&self, id: DocumentId, updates: HashMap<String, Value>) -> bool {
        if let Some(mut entry) = self.docs.get_mut(&id) {
            for (k, v) in updates {
                entry.fields.insert(k, v);
            }
            true
        } else {
            false
        }
    }

    /// Actualiza un campo anidado por dot-path.
    pub fn set_field(&self, id: DocumentId, path: &str, value: Value) -> bool {
        if let Some(mut entry) = self.docs.get_mut(&id) {
            let mut wrapper = Value::Map(Box::new(entry.fields.clone()));
            if wrapper.set_path(path, value).is_ok() {
                if let Value::Map(m) = wrapper {
                    entry.fields = *m;
                    return true;
                }
            }
            false
        } else {
            false
        }
    }

    /// Elimina un documento por ID.
    pub fn remove(&self, id: DocumentId) -> Option<Document> {
        self.docs.remove(&id).map(|(_, doc)| doc)
    }

    /// Busca documentos que cumplan un filtro.
    pub fn find(&self, filter: &DocFilter) -> Vec<Document> {
        self.docs
            .iter()
            .filter(|entry| filter.matches(entry.value()))
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Cuenta documentos que cumplan un filtro.
    pub fn count(&self, filter: &DocFilter) -> usize {
        self.docs
            .iter()
            .filter(|entry| filter.matches(entry.value()))
            .count()
    }

    /// Proyecta campos específicos de documentos filtrados.
    pub fn find_project(&self, filter: &DocFilter, field_paths: &[&str]) -> Vec<HashMap<String, Value>> {
        self.docs
            .iter()
            .filter(|entry| filter.matches(entry.value()))
            .map(|entry| {
                let doc = entry.value();
                let mut projected = HashMap::new();
                for &path in field_paths {
                    if let Some(val) = doc.get_field(path) {
                        projected.insert(path.to_string(), val.clone());
                    }
                }
                projected
            })
            .collect()
    }

    /// Itera sobre todos los documentos.
    pub fn iter_all(&self) -> Vec<Document> {
        self.docs.iter().map(|e| e.value().clone()).collect()
    }
}

/// Almacén de múltiples colecciones de documentos (thread-safe).
pub struct DocumentStore {
    collections: DashMap<String, Arc<DocumentCollection>>,
}

impl DocumentStore {
    pub fn new() -> Self {
        Self {
            collections: DashMap::new(),
        }
    }

    /// Crea o obtiene una colección.
    pub fn collection(&self, name: &str) -> Arc<DocumentCollection> {
        if let Some(c) = self.collections.get(name) {
            Arc::clone(c.value())
        } else {
            let col = Arc::new(DocumentCollection::new(name));
            self.collections.insert(name.to_string(), Arc::clone(&col));
            col
        }
    }

    /// Lista nombres de todas las colecciones.
    pub fn collection_names(&self) -> Vec<String> {
        self.collections.iter().map(|e| e.key().clone()).collect()
    }

    /// Elimina una colección entera.
    pub fn drop_collection(&self, name: &str) -> bool {
        self.collections.remove(name).is_some()
    }

    /// Número total de documentos en todas las colecciones.
    pub fn total_documents(&self) -> usize {
        self.collections.iter().map(|e| e.value().len()).sum()
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fields() -> HashMap<String, Value> {
        let mut address = HashMap::new();
        address.insert("city".into(), Value::string("NYC"));
        address.insert("zip".into(), Value::Int(10001));

        let mut fields = HashMap::new();
        fields.insert("name".into(), Value::string("Alice"));
        fields.insert("age".into(), Value::Int(30));
        fields.insert("address".into(), Value::Map(Box::new(address)));
        fields.insert("tags".into(), Value::List(Box::new(vec![
            Value::string("rust"),
            Value::string("graph"),
        ])));
        fields
    }

    #[test]
    fn test_document_get_field_flat() {
        let doc = Document { id: DocumentId(1), fields: sample_fields(), linked_node: None };
        assert_eq!(doc.get_field("name"), Some(&Value::string("Alice")));
        assert_eq!(doc.get_field("age"), Some(&Value::Int(30)));
    }

    #[test]
    fn test_document_get_field_nested() {
        let doc = Document { id: DocumentId(1), fields: sample_fields(), linked_node: None };
        assert_eq!(doc.get_field("address.city"), Some(&Value::string("NYC")));
        assert_eq!(doc.get_field("address.zip"), Some(&Value::Int(10001)));
    }

    #[test]
    fn test_document_get_field_list_index() {
        let doc = Document { id: DocumentId(1), fields: sample_fields(), linked_node: None };
        assert_eq!(doc.get_field("tags.0"), Some(&Value::string("rust")));
        assert_eq!(doc.get_field("tags.1"), Some(&Value::string("graph")));
    }

    #[test]
    fn test_collection_insert_and_get() {
        let col = DocumentCollection::new("users");
        let id = col.insert(sample_fields());
        assert_eq!(col.len(), 1);
        let doc = col.get(id).unwrap();
        assert_eq!(doc.get_field("name"), Some(&Value::string("Alice")));
    }

    #[test]
    fn test_collection_update() {
        let col = DocumentCollection::new("users");
        let id = col.insert(sample_fields());
        let mut updates = HashMap::new();
        updates.insert("age".into(), Value::Int(31));
        assert!(col.update(id, updates));
        let doc = col.get(id).unwrap();
        assert_eq!(doc.get_field("age"), Some(&Value::Int(31)));
    }

    #[test]
    fn test_collection_set_field_nested() {
        let col = DocumentCollection::new("users");
        let id = col.insert(sample_fields());
        assert!(col.set_field(id, "address.city", Value::string("LA")));
        let doc = col.get(id).unwrap();
        assert_eq!(doc.get_field("address.city"), Some(&Value::string("LA")));
    }

    #[test]
    fn test_collection_remove() {
        let col = DocumentCollection::new("users");
        let id = col.insert(sample_fields());
        assert_eq!(col.len(), 1);
        col.remove(id);
        assert_eq!(col.len(), 0);
        assert!(col.get(id).is_none());
    }

    #[test]
    fn test_collection_find_eq() {
        let col = DocumentCollection::new("users");
        col.insert(sample_fields());

        let mut fields2 = HashMap::new();
        fields2.insert("name".into(), Value::string("Bob"));
        fields2.insert("age".into(), Value::Int(25));
        col.insert(fields2);

        let results = col.find(&DocFilter::Eq {
            path: "name".into(),
            value: Value::string("Alice"),
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get_field("name"), Some(&Value::string("Alice")));
    }

    #[test]
    fn test_collection_find_nested_field() {
        let col = DocumentCollection::new("users");
        col.insert(sample_fields());

        let results = col.find(&DocFilter::Eq {
            path: "address.city".into(),
            value: Value::string("NYC"),
        });
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_collection_find_gt() {
        let col = DocumentCollection::new("users");
        col.insert(sample_fields());

        let mut fields2 = HashMap::new();
        fields2.insert("name".into(), Value::string("Bob"));
        fields2.insert("age".into(), Value::Int(25));
        col.insert(fields2);

        let results = col.find(&DocFilter::Gt {
            path: "age".into(),
            value: Value::Int(26),
        });
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_filter_contains() {
        let col = DocumentCollection::new("users");
        col.insert(sample_fields());

        let results = col.find(&DocFilter::Contains {
            path: "tags".into(),
            value: Value::string("rust"),
        });
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_filter_and_or() {
        let col = DocumentCollection::new("users");
        col.insert(sample_fields());

        let mut fields2 = HashMap::new();
        fields2.insert("name".into(), Value::string("Bob"));
        fields2.insert("age".into(), Value::Int(25));
        col.insert(fields2);

        // AND: name="Alice" AND age>20
        let filter = DocFilter::And(vec![
            DocFilter::Eq { path: "name".into(), value: Value::string("Alice") },
            DocFilter::Gt { path: "age".into(), value: Value::Int(20) },
        ]);
        assert_eq!(col.find(&filter).len(), 1);

        // OR: age=25 OR age=30
        let filter = DocFilter::Or(vec![
            DocFilter::Eq { path: "age".into(), value: Value::Int(25) },
            DocFilter::Eq { path: "age".into(), value: Value::Int(30) },
        ]);
        assert_eq!(col.find(&filter).len(), 2);
    }

    #[test]
    fn test_filter_exists() {
        let col = DocumentCollection::new("users");
        col.insert(sample_fields());

        let results = col.find(&DocFilter::Exists {
            path: "address.city".into(),
        });
        assert_eq!(results.len(), 1);

        let results = col.find(&DocFilter::Exists {
            path: "nonexistent".into(),
        });
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_collection_find_project() {
        let col = DocumentCollection::new("users");
        col.insert(sample_fields());

        let results = col.find_project(
            &DocFilter::Eq { path: "name".into(), value: Value::string("Alice") },
            &["name", "address.city"],
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("name"), Some(&Value::string("Alice")));
        assert_eq!(results[0].get("address.city"), Some(&Value::string("NYC")));
        assert!(results[0].get("age").is_none());
    }

    #[test]
    fn test_collection_count() {
        let col = DocumentCollection::new("users");
        col.insert(sample_fields());

        let mut fields2 = HashMap::new();
        fields2.insert("name".into(), Value::string("Bob"));
        col.insert(fields2);

        assert_eq!(col.count(&DocFilter::Exists { path: "name".into() }), 2);
    }

    #[test]
    fn test_collection_linked_node() {
        let col = DocumentCollection::new("users");
        let id = col.insert_linked(sample_fields(), NodeId(42));
        let doc = col.get(id).unwrap();
        assert_eq!(doc.linked_node, Some(NodeId(42)));
    }

    // ── DocumentStore tests ─────────────────────────────────────────────

    #[test]
    fn test_store_create_and_list_collections() {
        let store = DocumentStore::new();
        store.collection("users");
        store.collection("products");
        let names = store.collection_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"users".to_string()));
        assert!(names.contains(&"products".to_string()));
    }

    #[test]
    fn test_store_drop_collection() {
        let store = DocumentStore::new();
        let col = store.collection("temp");
        col.insert(sample_fields());
        assert!(store.drop_collection("temp"));
        assert_eq!(store.collection_names().len(), 0);
    }

    #[test]
    fn test_store_total_documents() {
        let store = DocumentStore::new();
        let c1 = store.collection("a");
        let c2 = store.collection("b");
        c1.insert(sample_fields());
        c1.insert(sample_fields());
        c2.insert(sample_fields());
        assert_eq!(store.total_documents(), 3);
    }

    #[test]
    fn test_store_collection_reuse() {
        let store = DocumentStore::new();
        let c1 = store.collection("users");
        c1.insert(sample_fields());
        // Second call returns same collection
        let c2 = store.collection("users");
        assert_eq!(c2.len(), 1);
    }

    #[test]
    fn test_filter_not() {
        let col = DocumentCollection::new("users");
        col.insert(sample_fields());

        let mut fields2 = HashMap::new();
        fields2.insert("name".into(), Value::string("Bob"));
        col.insert(fields2);

        let results = col.find(&DocFilter::Not(Box::new(DocFilter::Eq {
            path: "name".into(),
            value: Value::string("Alice"),
        })));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get_field("name"), Some(&Value::string("Bob")));
    }
}
