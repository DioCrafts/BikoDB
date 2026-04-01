// =============================================================================
// bikodb-storage::delta — Property-level delta encoding
// =============================================================================
// Instead of storing the full new record for updates, we compute a binary
// diff (delta) between the old and new serialized representations.
//
// Encoding: a list of (offset, length, new_bytes) hunks.
// Format:  [hunk_count:u16]([offset:u32][len:u16][new_bytes...])*
//
// For small property changes on large records, this dramatically reduces
// WAL write amplification and disk I/O.
//
// If the delta is larger than the full record, we fall back to storing
// the full record (signaled by hunk_count == 0xFFFF).
// =============================================================================

use bikodb_core::error::{BikoError, BikoResult};

/// A single change hunk: "at `offset`, replace `len` bytes with `data`".
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaHunk {
    pub offset: u32,
    pub data: Vec<u8>,
}

/// A delta between two serialized byte sequences.
#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    /// Property-level diff: list of (offset, new_bytes) hunks.
    Hunks(Vec<DeltaHunk>),
    /// Full replacement (delta was larger than full record).
    Full(Vec<u8>),
}

/// Sentinel value for hunk_count indicating a full replacement.
const FULL_REPLACEMENT: u16 = 0xFFFF;

/// Maximum number of hunks before we fall back to full replacement.
const MAX_HUNKS: usize = 256;

/// Computes a byte-level delta between `old` and `new` byte slices.
///
/// Groups contiguous changed bytes into hunks. Falls back to full
/// replacement if the delta encoding would be larger.
pub fn compute_delta(old: &[u8], new: &[u8]) -> Delta {
    // If lengths differ significantly or old is empty, use full replacement
    if old.is_empty() {
        return Delta::Full(new.to_vec());
    }

    let min_len = old.len().min(new.len());
    let mut hunks: Vec<DeltaHunk> = Vec::new();
    let mut i = 0;

    // Find changed regions in the overlapping part
    while i < min_len {
        if old[i] != new[i] {
            // Start of a changed region
            let start = i;
            while i < min_len && old[i] != new[i] {
                i += 1;
            }
            hunks.push(DeltaHunk {
                offset: start as u32,
                data: new[start..i].to_vec(),
            });

            if hunks.len() > MAX_HUNKS {
                return Delta::Full(new.to_vec());
            }
        } else {
            i += 1;
        }
    }

    // If new is longer, append the tail as a hunk
    if new.len() > old.len() {
        hunks.push(DeltaHunk {
            offset: min_len as u32,
            data: new[min_len..].to_vec(),
        });
    }

    // If no changes, return empty hunks
    if hunks.is_empty() && new.len() == old.len() {
        return Delta::Hunks(Vec::new());
    }

    // Compute encoded sizes and choose the smaller representation
    let delta_size = delta_encoded_size(&hunks);
    let full_size = 2 + new.len(); // hunk_count(2) + full data

    if delta_size >= full_size {
        Delta::Full(new.to_vec())
    } else {
        Delta::Hunks(hunks)
    }
}

/// Applies a delta to an old byte sequence, producing the new version.
pub fn apply_delta(old: &[u8], delta: &Delta) -> Vec<u8> {
    match delta {
        Delta::Full(data) => data.clone(),
        Delta::Hunks(hunks) => {
            let max_end = hunks
                .iter()
                .map(|h| h.offset as usize + h.data.len())
                .max()
                .unwrap_or(0);
            let result_len = old.len().max(max_end);
            let mut result = Vec::with_capacity(result_len);
            result.extend_from_slice(old);
            // Extend if new data goes beyond old length
            if result_len > result.len() {
                result.resize(result_len, 0);
            }
            for hunk in hunks {
                let start = hunk.offset as usize;
                let end = start + hunk.data.len();
                if end > result.len() {
                    result.resize(end, 0);
                }
                result[start..end].copy_from_slice(&hunk.data);
            }
            result
        }
    }
}

/// Serializes a Delta to bytes for WAL storage.
///
/// Format:
/// - Full: [0xFFFF:u16][data...]
/// - Hunks: [count:u16]([offset:u32][len:u16][data...])*
pub fn encode_delta(delta: &Delta) -> Vec<u8> {
    match delta {
        Delta::Full(data) => {
            let mut buf = Vec::with_capacity(2 + data.len());
            buf.extend_from_slice(&FULL_REPLACEMENT.to_le_bytes());
            buf.extend_from_slice(data);
            buf
        }
        Delta::Hunks(hunks) => {
            let mut buf = Vec::with_capacity(delta_encoded_size(hunks));
            buf.extend_from_slice(&(hunks.len() as u16).to_le_bytes());
            for hunk in hunks {
                buf.extend_from_slice(&hunk.offset.to_le_bytes());
                buf.extend_from_slice(&(hunk.data.len() as u16).to_le_bytes());
                buf.extend_from_slice(&hunk.data);
            }
            buf
        }
    }
}

/// Deserializes a Delta from bytes.
pub fn decode_delta(data: &[u8]) -> BikoResult<Delta> {
    if data.len() < 2 {
        return Err(BikoError::Serialization("delta too short".into()));
    }

    let count = u16::from_le_bytes([data[0], data[1]]);

    if count == FULL_REPLACEMENT {
        return Ok(Delta::Full(data[2..].to_vec()));
    }

    let mut offset = 2usize;
    let mut hunks = Vec::with_capacity(count as usize);

    for _ in 0..count {
        if offset + 6 > data.len() {
            return Err(BikoError::Serialization("truncated delta hunk".into()));
        }
        let hunk_offset =
            u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        let hunk_len =
            u16::from_le_bytes(data[offset + 4..offset + 6].try_into().unwrap())
                as usize;
        offset += 6;

        if offset + hunk_len > data.len() {
            return Err(BikoError::Serialization(
                "truncated delta hunk data".into(),
            ));
        }
        hunks.push(DeltaHunk {
            offset: hunk_offset,
            data: data[offset..offset + hunk_len].to_vec(),
        });
        offset += hunk_len;
    }

    Ok(Delta::Hunks(hunks))
}

fn delta_encoded_size(hunks: &[DeltaHunk]) -> usize {
    2 + hunks.iter().map(|h| 4 + 2 + h.data.len()).sum::<usize>()
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_data_empty_delta() {
        let old = b"hello world";
        let new = b"hello world";
        let delta = compute_delta(old, new);
        assert_eq!(delta, Delta::Hunks(Vec::new()));

        let result = apply_delta(old, &delta);
        assert_eq!(result, new);
    }

    #[test]
    fn test_single_byte_change() {
        let old = b"hello world";
        let new = b"hello World";
        let delta = compute_delta(old, new);

        match &delta {
            Delta::Hunks(hunks) => {
                assert_eq!(hunks.len(), 1);
                assert_eq!(hunks[0].offset, 6);
                assert_eq!(hunks[0].data, b"W");
            }
            Delta::Full(_) => panic!("expected hunks"),
        }

        let result = apply_delta(old, &delta);
        assert_eq!(result, new);
    }

    #[test]
    fn test_multiple_hunks() {
        // Use a record large enough that delta encoding beats full replacement.
        // 100 bytes with 2 small changes → hunks overhead (2 + 2×10) = 22 < full (102)
        let mut old = vec![0u8; 100];
        let mut new = old.clone();
        // Change bytes 10..14
        new[10] = 0xFF;
        new[11] = 0xFF;
        new[12] = 0xFF;
        new[13] = 0xFF;
        // Change bytes 80..84
        new[80] = 0xAA;
        new[81] = 0xAA;
        new[82] = 0xAA;
        new[83] = 0xAA;

        let delta = compute_delta(&old, &new);

        match &delta {
            Delta::Hunks(hunks) => {
                assert_eq!(hunks.len(), 2);
                assert_eq!(hunks[0].offset, 10);
                assert_eq!(hunks[0].data, vec![0xFF; 4]);
                assert_eq!(hunks[1].offset, 80);
                assert_eq!(hunks[1].data, vec![0xAA; 4]);
            }
            Delta::Full(_) => panic!("expected hunks"),
        }

        let result = apply_delta(&old, &delta);
        assert_eq!(result, new);
    }

    #[test]
    fn test_new_longer_than_old() {
        let old = b"short";
        let new = b"short and longer";
        let delta = compute_delta(old, new);

        let result = apply_delta(old, &delta);
        assert_eq!(result, new);
    }

    #[test]
    fn test_empty_old_full_replacement() {
        let old = b"";
        let new = b"brand new data";
        let delta = compute_delta(old, new);

        assert!(matches!(delta, Delta::Full(_)));
        let result = apply_delta(old, &delta);
        assert_eq!(result, new);
    }

    #[test]
    fn test_encode_decode_roundtrip_hunks() {
        let old = b"property: old_value_here";
        let new = b"property: new_value_here";
        let delta = compute_delta(old, new);

        let encoded = encode_delta(&delta);
        let decoded = decode_delta(&encoded).unwrap();
        assert_eq!(delta, decoded);

        let result = apply_delta(old, &decoded);
        assert_eq!(result, new);
    }

    #[test]
    fn test_encode_decode_roundtrip_full() {
        let delta = Delta::Full(b"full replacement data".to_vec());

        let encoded = encode_delta(&delta);
        let decoded = decode_delta(&encoded).unwrap();
        assert_eq!(delta, decoded);
    }

    #[test]
    fn test_delta_smaller_than_full() {
        // Large record with small change — delta should be much smaller
        let mut old = vec![0xAA; 1000];
        let mut new = old.clone();
        new[500] = 0xBB; // Change 1 byte

        let delta = compute_delta(&old, &new);
        let encoded = encode_delta(&delta);

        // Delta should be much smaller than full record (1000+ bytes)
        assert!(
            encoded.len() < 20,
            "encoded delta should be tiny, got {} bytes",
            encoded.len()
        );

        let result = apply_delta(&old, &delta);
        assert_eq!(result, new);
    }

    #[test]
    fn test_completely_different_data_full_replacement() {
        let old = vec![0x00; 100];
        let new = vec![0xFF; 100];

        let delta = compute_delta(&old, &new);
        // Single big hunk is still smaller than full, so it could be either
        let result = apply_delta(&old, &delta);
        assert_eq!(result, new);
    }

    #[test]
    fn test_decode_truncated_errors() {
        assert!(decode_delta(&[]).is_err());
        // Declare 2 hunks but only provide header
        let bad = [2u8, 0];
        assert!(decode_delta(&bad).is_err());
    }
}
