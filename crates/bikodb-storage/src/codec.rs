// =============================================================================
// bikodb-storage::codec — Serialización binaria + compresión LZ4
// =============================================================================
// Módulo de codificación/decodificación para registros y páginas.
//
// Diseño:
// - Formato binario compacto: [type_tag:u8][payload...]
// - Compresión LZ4 opcional para bloques > umbral
// - Integridad: checksum xxhash para verificación rápida
//
// Inspirado en ArcadeDB Binary serializer (compact, versionado).
// =============================================================================

use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use bikodb_core::error::{BikoError, BikoResult};

/// Umbral mínimo para activar compresión (bytes).
/// Bloques menores se almacenan sin comprimir.
const COMPRESSION_THRESHOLD: usize = 256;

/// Flag byte: datos sin comprimir
const FLAG_RAW: u8 = 0x00;
/// Flag byte: datos comprimidos con LZ4
const FLAG_LZ4: u8 = 0x01;

/// Codifica un bloque de bytes, aplicando compresión LZ4 si vale la pena.
///
/// Formato output: [flag:u8][payload...]
/// - flag=0x00: payload es raw
/// - flag=0x01: payload es LZ4 (con size prepended por lz4_flex)
///
/// Solo comprime si el input supera el umbral Y la compresión reduce tamaño.
pub fn encode(data: &[u8]) -> Vec<u8> {
    if data.len() < COMPRESSION_THRESHOLD {
        let mut out = Vec::with_capacity(1 + data.len());
        out.push(FLAG_RAW);
        out.extend_from_slice(data);
        return out;
    }

    let compressed = compress_prepend_size(data);

    // Solo usar compresión si realmente reduce tamaño
    if compressed.len() < data.len() {
        let mut out = Vec::with_capacity(1 + compressed.len());
        out.push(FLAG_LZ4);
        out.extend_from_slice(&compressed);
        out
    } else {
        let mut out = Vec::with_capacity(1 + data.len());
        out.push(FLAG_RAW);
        out.extend_from_slice(data);
        out
    }
}

/// Decodifica un bloque producido por `encode`.
///
/// Lee el flag byte para determinar si descomprimir.
pub fn decode(data: &[u8]) -> BikoResult<Vec<u8>> {
    if data.is_empty() {
        return Err(BikoError::Generic("codec: empty input".into()));
    }

    let flag = data[0];
    let payload = &data[1..];

    match flag {
        FLAG_RAW => Ok(payload.to_vec()),
        FLAG_LZ4 => decompress_size_prepended(payload).map_err(|e| {
            BikoError::Generic(format!("LZ4 decompression failed: {e}"))
        }),
        _ => Err(BikoError::Generic(format!(
            "codec: unknown flag byte 0x{flag:02X}"
        ))),
    }
}

/// Calcula xxHash64 de un bloque de bytes (para checksums de página).
pub fn checksum(data: &[u8]) -> u32 {
    // Usamos los 32 bits bajos de un hash simple para PageHeader.checksum
    let mut h: u32 = 0x811c_9dc5; // FNV offset basis 32
    for &byte in data {
        h ^= byte as u32;
        h = h.wrapping_mul(0x0100_0193); // FNV prime 32
    }
    h
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_small() {
        let data = b"hello world";
        let encoded = encode(data);
        assert_eq!(encoded[0], FLAG_RAW); // No comprime datos pequeños
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_encode_decode_large() {
        // Datos repetitivos que comprimen bien
        let data = vec![0xAA; 1024];
        let encoded = encode(&data);
        assert_eq!(encoded[0], FLAG_LZ4); // Debería comprimir
        assert!(encoded.len() < data.len()); // Más pequeño

        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_encode_decode_incompressible() {
        // Datos aleatorios que no comprimen
        let data: Vec<u8> = (0..512).map(|i| (i * 37 + 13) as u8).collect();
        let encoded = encode(&data);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_decode_empty_error() {
        assert!(decode(&[]).is_err());
    }

    #[test]
    fn test_decode_unknown_flag() {
        assert!(decode(&[0xFF, 0x01, 0x02]).is_err());
    }

    #[test]
    fn test_checksum_deterministic() {
        let data = b"test data for checksum";
        let c1 = checksum(data);
        let c2 = checksum(data);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_checksum_different_data() {
        let c1 = checksum(b"aaa");
        let c2 = checksum(b"bbb");
        assert_ne!(c1, c2);
    }
}
