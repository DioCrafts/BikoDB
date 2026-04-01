// =============================================================================
// bikodb-core::error — Errores unificados del motor
// =============================================================================
// Un solo tipo de error para todo BikoDB, con variantes por subsistema.
// Usar thiserror para derivar Display y Error automáticamente.
// =============================================================================

use crate::types::{NodeId, TypeId, RID};
use thiserror::Error;

/// Tipo de resultado estándar de BikoDB.
pub type BikoResult<T> = Result<T, BikoError>;

/// Error unificado de BikoDB.
///
/// Cada módulo añade sus variantes aquí. Esto permite que los errores
/// se propaguen con `?` a través de todo el stack sin conversiones manuales.
#[derive(Debug, Error)]
pub enum BikoError {
    // ── Storage ────────────────────────────────────────────────────────
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Page not found: file={file_id} page={page_number}")]
    PageNotFound { file_id: u32, page_number: u32 },

    #[error("Page corrupted: {details}")]
    PageCorrupted { details: String },

    // ── Records ────────────────────────────────────────────────────────
    #[error("Record not found: {0}")]
    RecordNotFound(RID),

    #[error("Node not found: {0}")]
    NodeNotFound(NodeId),

    #[error("Duplicate key: type={type_id} key={key}")]
    DuplicateKey { type_id: TypeId, key: String },

    // ── Schema ─────────────────────────────────────────────────────────
    #[error("Type not found: {0}")]
    TypeNotFound(String),

    #[error("Type already exists: {0}")]
    TypeAlreadyExists(String),

    #[error("Property not found: {property} on type {type_name}")]
    PropertyNotFound {
        type_name: String,
        property: String,
    },

    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        expected: String,
        actual: String,
    },

    // ── Transaction ────────────────────────────────────────────────────
    #[error("Transaction conflict: {details}")]
    TransactionConflict { details: String },

    #[error("Transaction not active")]
    TransactionNotActive,

    #[error("WAL corrupted: {details}")]
    WalCorrupted { details: String },

    // ── Query ──────────────────────────────────────────────────────────
    #[error("{0}")]
    Generic(String),

    #[error("Query parse error: {0}")]
    QueryParse(String),

    #[error("Query execution error: {0}")]
    QueryExecution(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    // ── Index ──────────────────────────────────────────────────────────
    #[error("Index not found: {0}")]
    IndexNotFound(String),

    // ── Cluster ────────────────────────────────────────────────────────
    #[error("Cluster error: {0}")]
    Cluster(String),

    // ── Generic ────────────────────────────────────────────────────────
    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

// ── Conversiones desde errores de dependencias ────────────────────────────

impl From<serde_json::Error> for BikoError {
    fn from(e: serde_json::Error) -> Self {
        BikoError::Serialization(e.to_string())
    }
}

impl From<bincode::Error> for BikoError {
    fn from(e: bincode::Error) -> Self {
        BikoError::Serialization(e.to_string())
    }
}
