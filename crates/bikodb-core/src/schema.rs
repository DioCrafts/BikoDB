// =============================================================================
// bikodb-core::schema — Definiciones de schema (tipos, propiedades)
// =============================================================================
// El schema define la estructura del Knowledge Graph:
//   - Qué tipos de nodos existen ("Person", "Company", "Document")
//   - Qué tipos de aristas existen ("KNOWS", "WORKS_AT", "SIMILAR_TO")
//   - Qué propiedades tiene cada tipo y sus tipos de dato
//   - Qué índices existen
//
// Inspirado en ArcadeDB LocalSchema (persiste a schema.json) y
// Neo4j schema constraints/indexes.
// =============================================================================

use crate::types::TypeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── PropertyType ───────────────────────────────────────────────────────────

/// Tipos de propiedad soportados (como Type enum de ArcadeDB).
///
/// Mapea 1:1 con los variantes de Value, pero a nivel de schema
/// (para validación y serialización tipada).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyType {
    Bool,
    Int,
    Float,
    String,
    Binary,
    List,
    Map,
    DateTime,
    Link,
    /// Vector de f32 para embeddings de IA
    Embedding,
}

// ── PropertyDef ────────────────────────────────────────────────────────────

/// Definición de una propiedad en el schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyDef {
    /// ID numérico de la propiedad (usado en PropertyEntry)
    pub id: u16,
    /// Nombre legible ("name", "age", "embedding")
    pub name: String,
    /// Tipo de dato
    pub property_type: PropertyType,
    /// ¿Es obligatoria? (NOT NULL)
    pub required: bool,
    /// ¿Tiene valor por defecto?
    pub default: Option<crate::value::Value>,
}

// ── IndexType ──────────────────────────────────────────────────────────────

/// Tipos de índice soportados (como Schema.INDEX_TYPE en ArcadeDB).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexType {
    /// B-tree / LSM-Tree para range queries y lookups
    LsmTree,
    /// Hash index para lookups exactos O(1)
    Hash,
    /// Full-text search (tokenización + ranking)
    FullText,
    /// Vector index HNSW para KNN búsquedas de similaridad
    Vector,
    /// Geospatial index para queries espaciales
    Geospatial,
}

// ── IndexDef ───────────────────────────────────────────────────────────────

/// Definición de un índice en el schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDef {
    /// Nombre del índice
    pub name: String,
    /// Tipo de índice
    pub index_type: IndexType,
    /// Tipo de record al que aplica
    pub type_id: TypeId,
    /// IDs de las propiedades indexadas
    pub property_ids: Vec<u16>,
    /// ¿Es un índice único (no permite duplicados)?
    pub unique: bool,
}

// ── TypeKind ───────────────────────────────────────────────────────────────

/// Clase de tipo (Document, Vertex o Edge).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeKind {
    Document,
    Vertex,
    Edge,
}

// ── TypeDef ────────────────────────────────────────────────────────────────

/// Definición completa de un tipo en el schema.
///
/// Equivalente a DocumentType/VertexType/EdgeType de ArcadeDB,
/// unificado en una sola struct con un discriminante `kind`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDef {
    /// ID numérico del tipo
    pub id: TypeId,
    /// Nombre legible ("Person", "KNOWS", etc.)
    pub name: String,
    /// Clase: Document, Vertex o Edge
    pub kind: TypeKind,
    /// Propiedades definidas para este tipo
    pub properties: Vec<PropertyDef>,
    /// Tipo padre (para herencia, como en ArcadeDB)
    pub super_type: Option<TypeId>,
}

impl TypeDef {
    /// Busca una propiedad por nombre.
    pub fn find_property(&self, name: &str) -> Option<&PropertyDef> {
        self.properties.iter().find(|p| p.name == name)
    }

    /// Busca una propiedad por ID.
    pub fn find_property_by_id(&self, id: u16) -> Option<&PropertyDef> {
        self.properties.iter().find(|p| p.id == id)
    }

    /// Siguiente ID de propiedad disponible.
    pub fn next_property_id(&self) -> u16 {
        self.properties.iter().map(|p| p.id).max().unwrap_or(0) + 1
    }
}

// ── Schema ─────────────────────────────────────────────────────────────────

/// Schema completo de la base de datos.
///
/// Contiene todos los tipos, propiedades e índices.
/// Se serializa a JSON (como schema.json en ArcadeDB).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Schema {
    /// Mapa de TypeId → TypeDef
    pub types: HashMap<TypeId, TypeDef>,
    /// Mapa de nombre → TypeId (índice inverso para lookup por nombre)
    pub type_names: HashMap<String, TypeId>,
    /// Índices definidos
    pub indexes: Vec<IndexDef>,
    /// Siguiente TypeId disponible
    next_type_id: u16,
}

impl Schema {
    /// Crea un schema vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra un nuevo tipo y devuelve su TypeId.
    ///
    /// # Errores
    /// Retorna None si ya existe un tipo con el mismo nombre.
    pub fn create_type(&mut self, name: &str, kind: TypeKind) -> Option<TypeId> {
        if self.type_names.contains_key(name) {
            return None; // Tipo duplicado
        }

        let id = TypeId(self.next_type_id);
        self.next_type_id += 1;

        let type_def = TypeDef {
            id,
            name: name.to_string(),
            kind,
            properties: Vec::new(),
            super_type: None,
        };

        self.types.insert(id, type_def);
        self.type_names.insert(name.to_string(), id);
        Some(id)
    }

    /// Busca un tipo por nombre.
    pub fn get_type_by_name(&self, name: &str) -> Option<&TypeDef> {
        self.type_names
            .get(name)
            .and_then(|id| self.types.get(id))
    }

    /// Busca un tipo por ID.
    pub fn get_type(&self, id: TypeId) -> Option<&TypeDef> {
        self.types.get(&id)
    }

    /// Obtiene referencia mutable a un tipo por ID.
    pub fn get_type_mut(&mut self, id: TypeId) -> Option<&mut TypeDef> {
        self.types.get_mut(&id)
    }

    /// Añade una propiedad a un tipo existente.
    ///
    /// Retorna el property_id asignado, o None si el tipo no existe.
    pub fn add_property(
        &mut self,
        type_id: TypeId,
        name: &str,
        prop_type: PropertyType,
        required: bool,
    ) -> Option<u16> {
        let type_def = self.types.get_mut(&type_id)?;
        let prop_id = type_def.next_property_id();
        type_def.properties.push(PropertyDef {
            id: prop_id,
            name: name.to_string(),
            property_type: prop_type,
            required,
            default: None,
        });
        Some(prop_id)
    }

    /// Añade un índice al schema.
    pub fn add_index(&mut self, index: IndexDef) {
        self.indexes.push(index);
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_create_types() {
        let mut schema = Schema::new();

        let person_id = schema.create_type("Person", TypeKind::Vertex).unwrap();
        let knows_id = schema.create_type("KNOWS", TypeKind::Edge).unwrap();

        assert_eq!(person_id, TypeId(0));
        assert_eq!(knows_id, TypeId(1));

        // Tipo duplicado retorna None
        assert!(schema.create_type("Person", TypeKind::Vertex).is_none());
    }

    #[test]
    fn test_schema_lookup_by_name() {
        let mut schema = Schema::new();
        schema.create_type("Person", TypeKind::Vertex);

        let td = schema.get_type_by_name("Person").unwrap();
        assert_eq!(td.name, "Person");
        assert_eq!(td.kind, TypeKind::Vertex);

        assert!(schema.get_type_by_name("NonExistent").is_none());
    }

    #[test]
    fn test_schema_add_properties() {
        let mut schema = Schema::new();
        let person_id = schema.create_type("Person", TypeKind::Vertex).unwrap();

        let name_id = schema
            .add_property(person_id, "name", PropertyType::String, true)
            .unwrap();
        let age_id = schema
            .add_property(person_id, "age", PropertyType::Int, false)
            .unwrap();

        assert_eq!(name_id, 1); // next_property_id starts from 0+1
        assert_eq!(age_id, 2);

        let td = schema.get_type(person_id).unwrap();
        assert_eq!(td.properties.len(), 2);
        assert_eq!(td.find_property("name").unwrap().property_type, PropertyType::String);
    }

    #[test]
    fn test_schema_serialization() {
        let mut schema = Schema::new();
        schema.create_type("Person", TypeKind::Vertex);

        let json = serde_json::to_string(&schema).unwrap();
        let restored: Schema = serde_json::from_str(&json).unwrap();

        assert!(restored.get_type_by_name("Person").is_some());
    }
}
