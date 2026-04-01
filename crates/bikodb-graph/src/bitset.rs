// =============================================================================
// bikodb-graph::bitset — Bitset compacto para tracking de nodos visitados
// =============================================================================
// Un bit por nodo en vez de un HashSet<NodeId>.
//
// ## Ventajas sobre HashSet:
// - 8x menos RAM: 1 bit vs ~40 bytes por entry en HashSet
// - Cache-friendly: acceso secuencial a bloques de 64 bits
// - Sin hashing: O(1) directo por index, zero overhead
// - Soporte para operaciones atómicas (BFS paralelo)
//
// Para 10M nodos: BitSet = 1.25 MB vs HashSet ≈ 400 MB
// =============================================================================

use std::sync::atomic::{AtomicU64, Ordering};

/// Bitset no-atómico, óptimo para BFS single-thread.
pub struct BitSet {
    words: Vec<u64>,
    len: usize,
}

impl BitSet {
    /// Crea un bitset con capacidad para `n` bits, todos en 0.
    pub fn new(n: usize) -> Self {
        let num_words = (n + 63) / 64;
        Self {
            words: vec![0u64; num_words],
            len: n,
        }
    }

    /// Marca bit `i` y retorna `true` si NO estaba marcado (insert nuevo).
    #[inline]
    pub fn set_and_check(&mut self, i: u32) -> bool {
        let word = (i / 64) as usize;
        let bit = 1u64 << (i % 64);
        let was_clear = (self.words[word] & bit) == 0;
        self.words[word] |= bit;
        was_clear
    }

    /// ¿Está marcado el bit `i`?
    #[inline]
    pub fn get(&self, i: u32) -> bool {
        let word = (i / 64) as usize;
        let bit = 1u64 << (i % 64);
        (self.words[word] & bit) != 0
    }

    /// Número de bits marcados.
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Capacidad total en bits.
    pub fn len(&self) -> usize {
        self.len
    }

    /// ¿Está vacío (ningún bit marcado)?
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }
}

/// Bitset atómico para BFS paralelo (múltiples threads marcan concurrentemente).
pub struct AtomicBitSet {
    words: Vec<AtomicU64>,
    len: usize,
}

impl AtomicBitSet {
    /// Crea un bitset atómico con capacidad para `n` bits, todos en 0.
    pub fn new(n: usize) -> Self {
        let num_words = (n + 63) / 64;
        let words = (0..num_words).map(|_| AtomicU64::new(0)).collect();
        Self { words, len: n }
    }

    /// Intenta marcar bit `i`. Retorna `true` si este thread lo marcó primero.
    ///
    /// Usa fetch_or atómico: lock-free, sin CAS retry loops.
    /// Múltiples threads pueden llamar esto concurrentemente sobre bits
    /// en el mismo word sin data races.
    #[inline]
    pub fn set_and_check(&self, i: u32) -> bool {
        let word = (i / 64) as usize;
        let bit = 1u64 << (i % 64);
        let prev = self.words[word].fetch_or(bit, Ordering::Relaxed);
        (prev & bit) == 0
    }

    /// ¿Está marcado el bit `i`?
    #[inline]
    pub fn get(&self, i: u32) -> bool {
        let word = (i / 64) as usize;
        let bit = 1u64 << (i % 64);
        (self.words[word].load(Ordering::Relaxed) & bit) != 0
    }

    /// Número de bits marcados.
    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as usize)
            .sum()
    }

    /// Capacidad total en bits.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|w| w.load(Ordering::Relaxed) == 0)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitset_basic() {
        let mut bs = BitSet::new(128);
        assert!(bs.is_empty());

        assert!(bs.set_and_check(0));   // nuevo
        assert!(!bs.set_and_check(0));  // ya existía
        assert!(bs.get(0));
        assert!(!bs.get(1));

        assert!(bs.set_and_check(127));
        assert_eq!(bs.count_ones(), 2);
    }

    #[test]
    fn test_bitset_all_bits() {
        let n = 200;
        let mut bs = BitSet::new(n);
        for i in 0..n as u32 {
            assert!(bs.set_and_check(i));
        }
        assert_eq!(bs.count_ones(), n);
        for i in 0..n as u32 {
            assert!(!bs.set_and_check(i)); // ya todos marcados
        }
    }

    #[test]
    fn test_atomic_bitset_basic() {
        let bs = AtomicBitSet::new(256);
        assert!(bs.set_and_check(0));
        assert!(!bs.set_and_check(0));
        assert!(bs.get(0));
        assert!(!bs.get(1));
        assert!(bs.set_and_check(255));
        assert_eq!(bs.count_ones(), 2);
    }

    #[test]
    fn test_atomic_bitset_concurrent() {
        use std::sync::Arc;
        use std::thread;

        let n = 10_000;
        let bs = Arc::new(AtomicBitSet::new(n));

        // 4 threads, cada uno intenta marcar todos los bits
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let bs = Arc::clone(&bs);
                thread::spawn(move || {
                    let mut first_count = 0usize;
                    for i in 0..n as u32 {
                        if bs.set_and_check(i) {
                            first_count += 1;
                        }
                    }
                    first_count
                })
            })
            .collect();

        let total_firsts: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

        // Exactamente n bits marcados: cada bit fue "first" exactamente una vez
        assert_eq!(total_firsts, n);
        assert_eq!(bs.count_ones(), n);
    }

    #[test]
    fn test_bitset_memory_efficiency() {
        // 10M bits = ~1.25 MB (vs HashSet ~400MB)
        let n = 10_000_000;
        let bs = BitSet::new(n);
        let bytes = bs.words.len() * 8; // Vec<u64>
        assert!(bytes < 2_000_000); // < 2MB
        assert_eq!(bs.len(), n);
    }
}
