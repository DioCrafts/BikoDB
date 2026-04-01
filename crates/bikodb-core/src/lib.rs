// =============================================================================
// bikodb-core — Módulo raíz
// =============================================================================
// Tipos fundamentales compartidos por todo el motor BikoDB.
//
// Este crate es la base de la pirámide de dependencias. NO depende de ningún
// otro crate de BikoDB, y todos los demás crates dependen de éste.
//
// Módulos:
//   types   → NodeId, EdgeId, RID (Record ID), TypeId
//   value   → Value enum (tipado dinámico para propiedades)
//   record  → Record, Document, Vertex, Edge (traits + structs)
//   schema  → PropertyDef, TypeDef, SchemaInfo
//   error   → BikoError (errores unificados)
//   config  → Constantes y configuración global
//   plugin  → Traits de extensibilidad (Plugin, Hook, UDF, etc.)
// =============================================================================

pub mod config;
pub mod error;
pub mod plugin;
pub mod record;
pub mod schema;
pub mod types;
pub mod value;

// Re-exports de conveniencia para import rápido:
//   use bikodb_core::prelude::*;
pub mod prelude {
    pub use crate::error::{BikoError, BikoResult};
    pub use crate::record::{Direction, Edge, Record, RecordType, Vertex};
    pub use crate::types::{BucketId, EdgeId, NodeId, PageId, TypeId, RID};
    pub use crate::value::Value;
}
