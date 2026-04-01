// =============================================================================
// bikodb-core::types — Identificadores fundamentales
// =============================================================================
// Diseño cache-friendly:
//   - NodeId y EdgeId son wrappers de u64 (8 bytes, cabe en registro de CPU)
//   - RID (Record ID) = BucketId + offset, empaquetado en u64
//   - Copy + Clone + Eq + Hash para uso eficiente en HashMaps y arrays
//
// Inspirado en ArcadeDB (#bucketId:offset) y Neo4j (internal long id).
// =============================================================================

use serde::{Deserialize, Serialize};
use std::fmt;

// ── NodeId ─────────────────────────────────────────────────────────────────
/// Identificador único de un vértice (nodo) en el grafo.
///
/// Internamente es un `u64` secuencial asignado por el storage engine.
/// Es Copy para poder pasarlo por valor sin overhead.
///
/// # Ejemplo
/// ```
/// use bikodb_core::types::NodeId;
/// let id = NodeId(42);
/// assert_eq!(id.0, 42);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(pub u64);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "N:{}", self.0)
    }
}

// ── EdgeId ─────────────────────────────────────────────────────────────────
/// Identificador único de una arista en el grafo.
///
/// Separado de NodeId para type-safety: no puedes pasar un EdgeId
/// donde se espera un NodeId sin conversión explícita.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EdgeId(pub u64);

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E:{}", self.0)
    }
}

// ── TypeId ─────────────────────────────────────────────────────────────────
/// Identificador del tipo/label de un nodo o arista en el schema.
///
/// Ejemplos: "Person" → TypeId(0), "KNOWS" → TypeId(1), "Company" → TypeId(2)
/// Se usa u16 porque raramente hay más de 65K tipos distintos.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TypeId(pub u16);

impl fmt::Display for TypeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T:{}", self.0)
    }
}

// ── BucketId ───────────────────────────────────────────────────────────────
/// Identificador de un bucket de almacenamiento.
///
/// Cada tipo de record puede tener múltiples buckets para distribuir
/// datos entre threads (inspirado en ArcadeDB BucketSelectionStrategy).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BucketId(pub u32);

impl fmt::Display for BucketId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "B:{}", self.0)
    }
}

// ── PageId ─────────────────────────────────────────────────────────────────
/// Identificador de una página en el storage engine.
///
/// Combina el ID del archivo (file/bucket) con el número de página
/// dentro de ese archivo. Esto permite al PageManager localizar
/// cualquier página en O(1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageId {
    /// ID del archivo/componente que contiene la página
    pub file_id: u32,
    /// Número de página dentro del archivo (0-indexed)
    pub page_number: u32,
}

impl fmt::Display for PageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P:{}:{}", self.file_id, self.page_number)
    }
}

// ── RID (Record ID) ───────────────────────────────────────────────────────
/// Record ID: identificador único de un registro en el almacenamiento.
///
/// Formato inspirado en ArcadeDB: `#bucket_id:offset`
/// Empaquetado en 8 bytes: [bucket_id: u16 | offset: u48]
///
/// Esto permite direccionar hasta 65K buckets × 281 trillones de registros
/// por bucket, más que suficiente para cualquier escala práctica.
///
/// # Layout en memoria
/// ```text
/// |  bucket_id (16 bits) | offset (48 bits)               |
/// |  0xFFFF               | 0xFFFF_FFFF_FFFF               |
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RID {
    /// ID del bucket que contiene el registro
    pub bucket_id: u16,
    /// Offset dentro del bucket (48 bits efectivos)
    pub offset: u64,
}

impl RID {
    /// Crea un nuevo RID a partir de bucket_id y offset.
    ///
    /// # Panics
    /// Panic si el offset excede 48 bits (> 0xFFFF_FFFF_FFFF).
    pub fn new(bucket_id: u16, offset: u64) -> Self {
        debug_assert!(
            offset <= 0x0000_FFFF_FFFF_FFFF,
            "RID offset excede 48 bits: {offset}"
        );
        Self { bucket_id, offset }
    }

    /// Empaqueta el RID en un u64 para almacenamiento compacto.
    ///
    /// Layout: [bucket_id: 16 bits MSB | offset: 48 bits LSB]
    #[inline]
    pub fn to_packed(&self) -> u64 {
        ((self.bucket_id as u64) << 48) | (self.offset & 0x0000_FFFF_FFFF_FFFF)
    }

    /// Desempaqueta un u64 en un RID.
    #[inline]
    pub fn from_packed(packed: u64) -> Self {
        Self {
            bucket_id: (packed >> 48) as u16,
            offset: packed & 0x0000_FFFF_FFFF_FFFF,
        }
    }

    /// RID inválido/nulo, usado como sentinel.
    pub const INVALID: RID = RID {
        bucket_id: u16::MAX,
        offset: u64::MAX,
    };

    /// Comprueba si este RID es válido (no es el sentinel).
    #[inline]
    pub fn is_valid(&self) -> bool {
        *self != Self::INVALID
    }
}

impl fmt::Display for RID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}:{}", self.bucket_id, self.offset)
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rid_pack_unpack_roundtrip() {
        let rid = RID::new(42, 123456);
        let packed = rid.to_packed();
        let unpacked = RID::from_packed(packed);
        assert_eq!(rid, unpacked);
    }

    #[test]
    fn test_rid_display_format() {
        let rid = RID::new(10, 345);
        assert_eq!(format!("{rid}"), "#10:345");
    }

    #[test]
    fn test_rid_invalid_sentinel() {
        assert!(!RID::INVALID.is_valid());
        assert!(RID::new(0, 0).is_valid());
    }

    #[test]
    fn test_node_id_display() {
        assert_eq!(format!("{}", NodeId(7)), "N:7");
    }

    #[test]
    fn test_edge_id_display() {
        assert_eq!(format!("{}", EdgeId(99)), "E:99");
    }

    #[test]
    fn test_type_id_display() {
        assert_eq!(format!("{}", TypeId(3)), "T:3");
    }

    #[test]
    fn test_rid_boundary_48bit() {
        let max_offset = 0x0000_FFFF_FFFF_FFFF;
        let rid = RID::new(u16::MAX - 1, max_offset);
        let roundtrip = RID::from_packed(rid.to_packed());
        assert_eq!(rid, roundtrip);
    }
}
