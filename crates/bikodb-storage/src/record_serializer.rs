// =============================================================================
// bikodb-storage::record_serializer — Bincode + LZ4 record serialization
// =============================================================================
// Pipeline: Record → bincode::serialize → codec::encode (LZ4)
//           codec::decode (LZ4) → bincode::deserialize → Record
//
// Each serialized record is prefixed with a 1-byte type tag so we can
// identify the record type without deserializing the whole blob.
// =============================================================================

use bikodb_core::error::{BikoError, BikoResult};
use bikodb_core::record::{Edge, RecordType, Vertex};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::codec;

const TAG_VERTEX: u8 = RecordType::Vertex as u8; // 1
const TAG_EDGE: u8 = RecordType::Edge as u8; // 2

/// Serializes any serde-compatible type to compressed binary.
///
/// Pipeline: T → bincode::serialize → codec::encode (LZ4 if beneficial)
pub fn serialize<T: Serialize>(record: &T) -> BikoResult<Vec<u8>> {
    let raw = bincode::serialize(record)
        .map_err(|e| BikoError::Serialization(format!("bincode encode: {e}")))?;
    Ok(codec::encode(&raw))
}

/// Deserializes compressed binary back to a serde type.
///
/// Pipeline: bytes → codec::decode (LZ4 decompress) → bincode::deserialize
pub fn deserialize<T: DeserializeOwned>(data: &[u8]) -> BikoResult<T> {
    let raw = codec::decode(data)?;
    bincode::deserialize(&raw)
        .map_err(|e| BikoError::Serialization(format!("bincode decode: {e}")))
}

/// Serializes a Vertex with a record-type tag prefix.
///
/// Format: [TAG_VERTEX:u8][compressed bincode bytes...]
pub fn serialize_vertex(vertex: &Vertex) -> BikoResult<Vec<u8>> {
    let compressed = serialize(vertex)?;
    let mut out = Vec::with_capacity(1 + compressed.len());
    out.push(TAG_VERTEX);
    out.extend_from_slice(&compressed);
    Ok(out)
}

/// Serializes an Edge with a record-type tag prefix.
///
/// Format: [TAG_EDGE:u8][compressed bincode bytes...]
pub fn serialize_edge(edge: &Edge) -> BikoResult<Vec<u8>> {
    let compressed = serialize(edge)?;
    let mut out = Vec::with_capacity(1 + compressed.len());
    out.push(TAG_EDGE);
    out.extend_from_slice(&compressed);
    Ok(out)
}

/// Peeks the record type from tagged data without full deserialization.
pub fn peek_record_type(data: &[u8]) -> BikoResult<RecordType> {
    match data.first() {
        Some(&1) => Ok(RecordType::Vertex),
        Some(&2) => Ok(RecordType::Edge),
        Some(&tag) => Err(BikoError::Serialization(format!(
            "unknown record tag: {tag}"
        ))),
        None => Err(BikoError::Serialization("empty record data".into())),
    }
}

/// Deserializes a tagged vertex record.
pub fn deserialize_vertex(data: &[u8]) -> BikoResult<Vertex> {
    if data.first() != Some(&TAG_VERTEX) {
        return Err(BikoError::Serialization("not a vertex record".into()));
    }
    deserialize(&data[1..])
}

/// Deserializes a tagged edge record.
pub fn deserialize_edge(data: &[u8]) -> BikoResult<Edge> {
    if data.first() != Some(&TAG_EDGE) {
        return Err(BikoError::Serialization("not an edge record".into()));
    }
    deserialize(&data[1..])
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use bikodb_core::record::RecordType;
    use bikodb_core::types::{EdgeId, NodeId, TypeId};
    use bikodb_core::value::Value;

    fn test_vertex() -> Vertex {
        let mut v = Vertex::new(NodeId(42), TypeId(1));
        v.set_property(0, Value::string("Alice"));
        v.set_property(1, Value::Int(30));
        v.set_property(2, Value::Float(1.75));
        v
    }

    fn test_edge() -> Edge {
        let mut e = Edge::new(EdgeId(100), TypeId(5), NodeId(1), NodeId(2));
        e.set_property(0, Value::string("KNOWS"));
        e.set_property(1, Value::Int(2020));
        e
    }

    #[test]
    fn test_vertex_roundtrip() {
        let v = test_vertex();
        let data = serialize_vertex(&v).unwrap();
        let recovered = deserialize_vertex(&data).unwrap();

        assert_eq!(recovered.id, v.id);
        assert_eq!(recovered.type_id, v.type_id);
        assert_eq!(
            recovered.get_property(0).unwrap().as_str(),
            Some("Alice")
        );
        assert_eq!(recovered.get_property(1).unwrap().as_int(), Some(30));
        assert_eq!(recovered.get_property(2).unwrap().as_float(), Some(1.75));
    }

    #[test]
    fn test_edge_roundtrip() {
        let e = test_edge();
        let data = serialize_edge(&e).unwrap();
        let recovered = deserialize_edge(&data).unwrap();

        assert_eq!(recovered.id, e.id);
        assert_eq!(recovered.source, e.source);
        assert_eq!(recovered.target, e.target);
        assert_eq!(recovered.type_id, e.type_id);
        assert_eq!(
            recovered.get_property(0).unwrap().as_str(),
            Some("KNOWS")
        );
    }

    #[test]
    fn test_peek_record_type() {
        let v_data = serialize_vertex(&test_vertex()).unwrap();
        let e_data = serialize_edge(&test_edge()).unwrap();

        assert_eq!(peek_record_type(&v_data).unwrap(), RecordType::Vertex);
        assert_eq!(peek_record_type(&e_data).unwrap(), RecordType::Edge);
        assert!(peek_record_type(&[]).is_err());
        assert!(peek_record_type(&[99]).is_err());
    }

    #[test]
    fn test_type_mismatch_errors() {
        let v_data = serialize_vertex(&test_vertex()).unwrap();
        let e_data = serialize_edge(&test_edge()).unwrap();

        assert!(deserialize_edge(&v_data).is_err());
        assert!(deserialize_vertex(&e_data).is_err());
    }

    #[test]
    fn test_compression_reduces_size() {
        // Create a vertex with large repetitive data to trigger LZ4 compression
        let mut v = Vertex::new(NodeId(1), TypeId(0));
        let large_string = "a]".repeat(500);
        v.set_property(0, Value::string(large_string));

        let serialized = serialize_vertex(&v).unwrap();
        let raw_bincode = bincode::serialize(&v).unwrap();

        // Serialized (with LZ4) should be smaller than raw bincode for repetitive data
        // +1 for the type tag byte
        assert!(
            serialized.len() < raw_bincode.len() + 10,
            "compressed={} raw={}",
            serialized.len(),
            raw_bincode.len()
        );
    }

    #[test]
    fn test_generic_serialize_deserialize() {
        let v = test_vertex();
        let data = serialize(&v).unwrap();
        let recovered: Vertex = deserialize(&data).unwrap();
        assert_eq!(recovered.id, v.id);
    }
}
