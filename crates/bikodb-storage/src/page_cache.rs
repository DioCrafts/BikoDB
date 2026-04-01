// =============================================================================
// bikodb-storage::page_cache — Cache de páginas con eviction Clock
// =============================================================================
// Cache concurrente de páginas en RAM.
//
// Diseño:
// - DashMap para reads/writes lock-free (sharded por PageId hash)
// - Eviction via Clock algorithm (second-chance) for better cache hit rate
// - Stats: hits, misses, evictions para monitoreo
// - Budget-aware: can evict down to a target size for memory backpressure
//
// Inspirado en ArcadeDB PageManager (ConcurrentHashMap cache).
// =============================================================================

use dashmap::DashMap;
use bikodb_core::types::PageId;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use parking_lot::Mutex;

use crate::page::Page;

/// Estadísticas del cache de páginas.
///
/// Todos los contadores son atómicos para acceso concurrente sin lock.
#[derive(Debug)]
pub struct CacheStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    pub current_size: AtomicU64,
}

impl CacheStats {
    fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            current_size: AtomicU64::new(0),
        }
    }

    /// Hit rate como porcentaje (0.0 - 100.0).
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            return 0.0;
        }
        (hits as f64 / total as f64) * 100.0
    }
}

impl Default for CacheStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache concurrente de páginas.
///
/// Almacena páginas indexadas por PageId en un DashMap (lock-free sharded).
/// Usa Clock algorithm (second-chance) para eviction:
/// - Cada página tiene un "referenced" bit (set on access)
/// - Clock hand rotates through pages; referenced pages get a second chance
/// - Unclean pages are skipped (they need flush first)
///
/// # Concurrencia
/// - Reads: completamente lock-free (DashMap sharding)
/// - Writes: lock per shard (solo bloquea 1/N del cache)
/// - Stats: atómicos, sin locks
///
/// # Ejemplo
/// ```
/// use bikodb_storage::page_cache::PageCache;
/// use bikodb_storage::page::Page;
/// use bikodb_core::types::PageId;
///
/// let cache = PageCache::new(1024);
/// let pid = PageId { file_id: 0, page_number: 0 };
/// let page = Page::new(4096);
///
/// cache.put(pid, page);
/// assert!(cache.get(&pid).is_some());
/// ```
pub struct PageCache {
    /// Mapa concurrente: PageId → Page
    pages: DashMap<PageId, Page>,
    /// Referenced bits for clock algorithm (PageId → referenced)
    referenced: DashMap<PageId, AtomicBool>,
    /// Clock hand: ordered list of page IDs for sweep
    clock_ring: Mutex<Vec<PageId>>,
    /// Current position of clock hand
    clock_hand: Mutex<usize>,
    /// Número máximo de páginas en cache
    max_pages: usize,
    /// Estadísticas
    pub stats: CacheStats,
}

impl PageCache {
    /// Crea un cache con capacidad máxima de `max_pages`.
    pub fn new(max_pages: usize) -> Self {
        Self {
            pages: DashMap::with_capacity(max_pages),
            referenced: DashMap::with_capacity(max_pages),
            clock_ring: Mutex::new(Vec::with_capacity(max_pages)),
            clock_hand: Mutex::new(0),
            max_pages,
            stats: CacheStats::new(),
        }
    }

    /// Obtiene una página del cache.
    ///
    /// Sets the referenced bit (second-chance for clock eviction).
    pub fn get(&self, page_id: &PageId) -> Option<dashmap::mapref::one::Ref<'_, PageId, Page>> {
        match self.pages.get(page_id) {
            Some(entry) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                // Set referenced bit for clock algorithm
                if let Some(r) = self.referenced.get(page_id) {
                    r.value().store(true, Ordering::Relaxed);
                }
                Some(entry)
            }
            None => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Inserta una página en el cache.
    ///
    /// Si el cache está lleno, evicta via clock algorithm.
    pub fn put(&self, page_id: PageId, page: Page) {
        // Evicción si estamos al límite
        if self.pages.len() >= self.max_pages && !self.pages.contains_key(&page_id) {
            self.clock_evict();
        }

        let is_new = !self.pages.contains_key(&page_id);
        self.pages.insert(page_id, page);
        self.referenced
            .insert(page_id, AtomicBool::new(true));

        if is_new {
            self.clock_ring.lock().push(page_id);
        }

        self.stats
            .current_size
            .store(self.pages.len() as u64, Ordering::Relaxed);
    }

    /// Obtiene una referencia mutable a una página (para modificarla in-place).
    pub fn get_mut(
        &self,
        page_id: &PageId,
    ) -> Option<dashmap::mapref::one::RefMut<'_, PageId, Page>> {
        let result = self.pages.get_mut(page_id);
        if result.is_some() {
            if let Some(r) = self.referenced.get(page_id) {
                r.value().store(true, Ordering::Relaxed);
            }
        }
        result
    }

    /// Elimina una página del cache.
    pub fn remove(&self, page_id: &PageId) -> Option<Page> {
        self.referenced.remove(page_id);
        self.pages.remove(page_id).map(|(_, page)| {
            // Remove from clock ring
            self.clock_ring.lock().retain(|p| p != page_id);
            self.stats
                .current_size
                .store(self.pages.len() as u64, Ordering::Relaxed);
            page
        })
    }

    /// Clock algorithm eviction: sweeps the clock hand looking for a victim.
    ///
    /// - If referenced bit is set → clear it (second chance) → advance
    /// - If referenced bit is clear AND page is clean → evict
    /// - Dirty pages are always skipped (need flush first)
    /// - Scans at most 2×ring_size before giving up
    fn clock_evict(&self) {
        let ring = self.clock_ring.lock();
        let ring_len = ring.len();
        if ring_len == 0 {
            return;
        }

        let mut hand = self.clock_hand.lock();
        let max_scans = ring_len * 2;

        for _ in 0..max_scans {
            let idx = *hand % ring_len;
            let pid = ring[idx];
            *hand = (*hand + 1) % ring_len;

            // Skip dirty pages
            if let Some(entry) = self.pages.get(&pid) {
                if entry.value().is_dirty() {
                    continue;
                }
            } else {
                continue;
            }

            // Check referenced bit
            if let Some(r) = self.referenced.get(&pid) {
                if r.value().load(Ordering::Relaxed) {
                    // Second chance: clear referenced bit, move on
                    r.value().store(false, Ordering::Relaxed);
                    continue;
                }
            }

            // Victim found: evict
            drop(ring);
            drop(hand);
            self.pages.remove(&pid);
            self.referenced.remove(&pid);
            self.clock_ring.lock().retain(|p| *p != pid);
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // If all pages have second chance or are dirty, evict first clean page
        drop(ring);
        drop(hand);
        self.evict_first_clean();
    }

    /// Fallback: evict first clean page found (no referenced check).
    fn evict_first_clean(&self) {
        let mut to_evict = None;
        for entry in self.pages.iter() {
            if !entry.value().is_dirty() {
                to_evict = Some(*entry.key());
                break;
            }
        }
        if let Some(pid) = to_evict {
            self.pages.remove(&pid);
            self.referenced.remove(&pid);
            self.clock_ring.lock().retain(|p| *p != pid);
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Evicts clean pages down to `target_count` pages.
    ///
    /// Used by budget enforcement to free memory under pressure.
    /// Returns the number of pages evicted.
    pub fn evict_to_target(&self, target_count: usize) -> usize {
        let mut evicted = 0;
        while self.pages.len() > target_count {
            let before = self.pages.len();
            self.clock_evict();
            if self.pages.len() >= before {
                break; // No progress (all dirty)
            }
            evicted += 1;
        }
        evicted
    }

    /// Retorna todas las páginas dirty para flush a disco.
    pub fn dirty_pages(&self) -> Vec<PageId> {
        self.pages
            .iter()
            .filter(|entry| entry.value().is_dirty())
            .map(|entry| *entry.key())
            .collect()
    }

    /// Número actual de páginas en cache.
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// ¿Cache vacío?
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn make_pid(file: u32, page: u32) -> PageId {
        PageId {
            file_id: file,
            page_number: page,
        }
    }

    #[test]
    fn test_cache_put_get() {
        let cache = PageCache::new(100);
        let pid = make_pid(0, 0);
        let mut page = Page::new(4096);
        page.write(0, b"data");

        cache.put(pid, page);
        assert!(cache.get(&pid).is_some());
        assert!(cache.get(&make_pid(0, 1)).is_none());
    }

    #[test]
    fn test_cache_stats() {
        let cache = PageCache::new(100);
        let pid = make_pid(0, 0);
        cache.put(pid, Page::new(4096));

        cache.get(&pid); // hit
        cache.get(&pid); // hit
        cache.get(&make_pid(1, 0)); // miss

        assert_eq!(cache.stats.hits.load(Ordering::Relaxed), 2);
        assert_eq!(cache.stats.misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = PageCache::new(2);
        cache.put(make_pid(0, 0), Page::new(4096));
        cache.put(make_pid(0, 1), Page::new(4096));
        // Cache lleno, la siguiente put debería evictar una
        cache.put(make_pid(0, 2), Page::new(4096));

        assert!(cache.len() <= 2);
        assert!(cache.stats.evictions.load(Ordering::Relaxed) >= 1);
    }

    #[test]
    fn test_cache_dirty_pages() {
        let cache = PageCache::new(100);
        let pid = make_pid(0, 0);
        let mut page = Page::new(4096);
        page.write(0, b"dirty"); // Marca como dirty

        cache.put(pid, page);
        let dirty = cache.dirty_pages();
        assert!(dirty.contains(&pid));
    }

    #[test]
    fn test_cache_hit_rate() {
        let cache = PageCache::new(100);
        let pid = make_pid(0, 0);
        cache.put(pid, Page::new(4096));

        // 3 hits, 1 miss → 75%
        cache.get(&pid);
        cache.get(&pid);
        cache.get(&pid);
        cache.get(&make_pid(1, 0));

        let rate = cache.stats.hit_rate();
        assert!((rate - 75.0).abs() < 0.01);
    }
}
