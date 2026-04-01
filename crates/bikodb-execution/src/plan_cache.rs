// =============================================================================
// bikodb-execution::plan_cache — LRU plan cache
// =============================================================================
// Caches optimized LogicalOp plans keyed by a u64 hash of the query string.
// Uses a simple LRU eviction policy with configurable capacity.
// =============================================================================

use std::collections::HashMap;

use crate::plan::LogicalOp;

/// Entry in the plan cache: the optimized plan + insertion order.
#[derive(Debug, Clone)]
struct CacheEntry {
    plan: LogicalOp,
    /// Monotonic counter; higher = more recently used.
    last_used: u64,
}

/// LRU cache for optimized query plans.
///
/// Keyed by a `u64` hash of the query (caller is responsible for hashing).
/// When the cache exceeds `capacity`, the least-recently-used entry is evicted.
#[derive(Debug)]
pub struct PlanCache {
    entries: HashMap<u64, CacheEntry>,
    capacity: usize,
    counter: u64,
    hits: u64,
    misses: u64,
}

impl PlanCache {
    /// Creates a new cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "PlanCache capacity must be > 0");
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            counter: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Looks up a cached plan by query hash.
    pub fn get(&mut self, key: u64) -> Option<LogicalOp> {
        self.counter += 1;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used = self.counter;
            self.hits += 1;
            Some(entry.plan.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    /// Inserts an optimized plan into the cache. Evicts LRU if full.
    pub fn put(&mut self, key: u64, plan: LogicalOp) {
        self.counter += 1;
        if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) {
            self.evict_lru();
        }
        self.entries.insert(
            key,
            CacheEntry {
                plan,
                last_used: self.counter,
            },
        );
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cache hit count.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cache miss count.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Hit rate as a percentage (0.0–100.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Evicts the least-recently-used entry.
    fn evict_lru(&mut self) {
        if let Some((&lru_key, _)) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
        {
            self.entries.remove(&lru_key);
        }
    }
}

/// Computes a simple hash for a query string (FNV-1a 64-bit).
pub fn query_hash(query: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in query.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use bikodb_core::types::TypeId;

    fn scan(tid: u16) -> LogicalOp {
        LogicalOp::NodeScan {
            type_id: TypeId(tid),
        }
    }

    #[test]
    fn test_put_and_get() {
        let mut cache = PlanCache::new(4);
        let key = query_hash("SELECT * FROM Person");
        cache.put(key, scan(1));

        let result = cache.get(key);
        assert!(result.is_some());
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn test_miss() {
        let mut cache = PlanCache::new(4);
        let result = cache.get(12345);
        assert!(result.is_none());
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn test_eviction() {
        let mut cache = PlanCache::new(2);
        cache.put(1, scan(1));
        cache.put(2, scan(2));
        // Accessing key 1 makes it recently used
        let _ = cache.get(1);
        // Adding key 3 should evict key 2 (LRU)
        cache.put(3, scan(3));

        assert_eq!(cache.len(), 2);
        assert!(cache.get(2).is_none()); // evicted
        assert!(cache.get(1).is_some()); // kept
        assert!(cache.get(3).is_some()); // added
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = PlanCache::new(4);
        cache.put(1, scan(1));
        let _ = cache.get(1); // hit
        let _ = cache.get(1); // hit
        let _ = cache.get(2); // miss

        assert!((cache.hit_rate() - 66.666).abs() < 1.0);
    }

    #[test]
    fn test_query_hash_deterministic() {
        let h1 = query_hash("MATCH (p:Person) RETURN p");
        let h2 = query_hash("MATCH (p:Person) RETURN p");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_query_hash_different() {
        let h1 = query_hash("query A");
        let h2 = query_hash("query B");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_clear() {
        let mut cache = PlanCache::new(4);
        cache.put(1, scan(1));
        cache.put(2, scan(2));
        cache.clear();
        assert!(cache.is_empty());
    }
}
