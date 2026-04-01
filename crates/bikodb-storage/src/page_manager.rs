// =============================================================================
// bikodb-storage::page_manager — PageCache ↔ MmapFile integration
// =============================================================================
// Provides transparent disk-backed paging:
//   - Load-on-miss: pages not in cache are loaded from the mmap file
//   - Flush: dirty pages in cache are written back to mmap + fsync
//   - Allocate: grows the mmap file for new pages
//
// Thread-safe: MmapFile is wrapped in RwLock, PageCache uses DashMap.
// =============================================================================

use bikodb_core::config::DEFAULT_PAGE_SIZE;
use bikodb_core::error::{BikoError, BikoResult};
use bikodb_core::types::PageId;
use parking_lot::RwLock;
use std::path::Path;

use crate::mmap::MmapFile;
use crate::page::{Page, PageHeader};
use crate::page_cache::{CacheStats, PageCache};

/// Integrates PageCache with MmapFile for transparent disk-backed paging.
pub struct PageManager {
    mmap: RwLock<MmapFile>,
    cache: PageCache,
    page_size: usize,
    file_id: u32,
}

impl PageManager {
    /// Opens or creates a PageManager backed by the given file (default 64KB pages).
    pub fn open(path: impl AsRef<Path>, max_cached_pages: usize) -> BikoResult<Self> {
        Self::open_with_page_size(path, DEFAULT_PAGE_SIZE, max_cached_pages)
    }

    /// Opens with a custom page size (useful for testing with small pages).
    pub fn open_with_page_size(
        path: impl AsRef<Path>,
        page_size: usize,
        max_cached_pages: usize,
    ) -> BikoResult<Self> {
        let mmap = MmapFile::open(path, page_size)?;
        let cache = PageCache::new(max_cached_pages);
        Ok(Self {
            mmap: RwLock::new(mmap),
            cache,
            page_size,
            file_id: 0,
        })
    }

    fn pid(&self, page_number: u32) -> PageId {
        PageId {
            file_id: self.file_id,
            page_number,
        }
    }

    /// Loads a page from the mmap file (does NOT check cache first).
    fn load_from_disk(&self, page_number: u32) -> BikoResult<Page> {
        let mmap = self.mmap.read();
        if page_number >= mmap.page_count() {
            return Err(BikoError::PageNotFound {
                file_id: self.file_id,
                page_number,
            });
        }
        let raw = mmap.read_page(page_number)?;
        Ok(Page::from_bytes(raw, self.page_size))
    }

    /// Reads bytes from a page at the given offset. Loads from disk on cache miss.
    pub fn read_from_page(
        &self,
        page_number: u32,
        offset: usize,
        len: usize,
    ) -> BikoResult<Vec<u8>> {
        let pid = self.pid(page_number);

        // Fast path: cache hit
        if let Some(entry) = self.cache.get(&pid) {
            return Ok(entry.value().read(offset, len).to_vec());
        }

        // Slow path: load from disk, cache it, read
        let page = self.load_from_disk(page_number)?;
        let result = page.read(offset, len).to_vec();
        self.cache.put(pid, page);
        Ok(result)
    }

    /// Writes bytes to a page at the given offset. Loads from disk on cache miss.
    pub fn write_to_page(
        &self,
        page_number: u32,
        offset: usize,
        data: &[u8],
    ) -> BikoResult<()> {
        let pid = self.pid(page_number);

        // Ensure the page is in cache
        if self.cache.get(&pid).is_none() {
            let page = {
                let mmap = self.mmap.read();
                if page_number < mmap.page_count() {
                    drop(mmap);
                    self.load_from_disk(page_number)?
                } else {
                    Page::new(self.page_size)
                }
            };
            self.cache.put(pid, page);
        }

        // Write through cache (marks page dirty)
        match self.cache.get_mut(&pid) {
            Some(mut entry) => {
                entry.value_mut().write(offset, data);
                Ok(())
            }
            None => Err(BikoError::PageNotFound {
                file_id: self.file_id,
                page_number,
            }),
        }
    }

    /// Allocates a new empty page, returning its page number.
    pub fn allocate_page(&self) -> BikoResult<u32> {
        let page_num = {
            let mut mmap = self.mmap.write();
            mmap.allocate_page()?
        };
        let pid = self.pid(page_num);
        let page = Page::new(self.page_size);
        self.cache.put(pid, page);
        Ok(page_num)
    }

    /// Flushes all dirty pages from cache to the mmap file + fsync.
    pub fn flush(&self) -> BikoResult<()> {
        let dirty = self.cache.dirty_pages();
        if dirty.is_empty() {
            return Ok(());
        }

        let mut mmap = self.mmap.write();
        for pid in &dirty {
            if let Some(entry) = self.cache.get(pid) {
                let bytes = entry.value().to_bytes();
                mmap.write_page(pid.page_number, &bytes)?;
            }
        }
        // Mark pages clean after writing
        for pid in &dirty {
            if let Some(mut entry) = self.cache.get_mut(pid) {
                entry.value_mut().mark_clean();
            }
        }
        mmap.flush()
    }

    /// Returns the content_size of a cached page (loads from disk if needed).
    pub fn page_content_size(&self, page_number: u32) -> BikoResult<usize> {
        let pid = self.pid(page_number);

        if let Some(entry) = self.cache.get(&pid) {
            return Ok(entry.value().header.content_size as usize);
        }

        let page = self.load_from_disk(page_number)?;
        let size = page.header.content_size as usize;
        self.cache.put(pid, page);
        Ok(size)
    }

    /// Usable data bytes per page (page_size minus header).
    pub fn data_capacity(&self) -> usize {
        self.page_size - PageHeader::SIZE
    }

    /// Number of pages currently on disk.
    pub fn page_count(&self) -> u32 {
        self.mmap.read().page_count()
    }

    /// Page size in bytes.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Cache statistics (hits, misses, evictions).
    pub fn cache_stats(&self) -> &CacheStats {
        &self.cache.stats
    }

    /// Evicts clean pages down to `target_count` via clock algorithm.
    ///
    /// Used by budget enforcement to keep cache within memory limits.
    /// Returns the number of pages evicted.
    pub fn evict_to_target(&self, target_count: usize) -> usize {
        self.cache.evict_to_target(target_count)
    }

    /// Current number of pages in cache.
    pub fn cached_page_count(&self) -> usize {
        self.cache.len()
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("bikodb_test_page_mgr");
        fs::create_dir_all(&dir).ok();
        let p = dir.join(name);
        let _ = fs::remove_file(&p);
        p
    }

    #[test]
    fn test_allocate_and_write() {
        let path = temp_path("test_alloc_write.dat");
        let mgr = PageManager::open_with_page_size(&path, 256, 64).unwrap();

        let pn = mgr.allocate_page().unwrap();
        assert_eq!(pn, 0);

        mgr.write_to_page(pn, 0, b"hello").unwrap();

        let data = mgr.read_from_page(pn, 0, 5).unwrap();
        assert_eq!(&data, b"hello");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_flush_persists_to_disk() {
        let path = temp_path("test_flush_persist.dat");

        // Write and flush
        {
            let mgr = PageManager::open_with_page_size(&path, 256, 64).unwrap();
            let pn = mgr.allocate_page().unwrap();
            mgr.write_to_page(pn, 0, b"persistent data").unwrap();
            mgr.flush().unwrap();
        }

        // Reopen and read
        {
            let mgr = PageManager::open_with_page_size(&path, 256, 64).unwrap();
            let data = mgr.read_from_page(0, 0, 15).unwrap();
            assert_eq!(&data, b"persistent data");
        }

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_cache_hit_on_second_read() {
        let path = temp_path("test_cache_hit.dat");
        let mgr = PageManager::open_with_page_size(&path, 256, 64).unwrap();
        let pn = mgr.allocate_page().unwrap();
        mgr.write_to_page(pn, 0, b"data").unwrap();

        // First read (should be cache hit since we just wrote)
        let _ = mgr.read_from_page(pn, 0, 4).unwrap();
        // Second read (cache hit)
        let _ = mgr.read_from_page(pn, 0, 4).unwrap();

        let hits = mgr.cache_stats().hits.load(Ordering::Relaxed);
        assert!(hits >= 2, "Expected at least 2 cache hits, got {hits}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_multiple_pages() {
        let path = temp_path("test_multi_page.dat");
        let mgr = PageManager::open_with_page_size(&path, 256, 64).unwrap();

        let p0 = mgr.allocate_page().unwrap();
        let p1 = mgr.allocate_page().unwrap();
        let p2 = mgr.allocate_page().unwrap();

        mgr.write_to_page(p0, 0, b"page-0").unwrap();
        mgr.write_to_page(p1, 0, b"page-1").unwrap();
        mgr.write_to_page(p2, 0, b"page-2").unwrap();

        assert_eq!(&mgr.read_from_page(p0, 0, 6).unwrap(), b"page-0");
        assert_eq!(&mgr.read_from_page(p1, 0, 6).unwrap(), b"page-1");
        assert_eq!(&mgr.read_from_page(p2, 0, 6).unwrap(), b"page-2");

        assert_eq!(mgr.page_count(), 3);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_content_size_tracking() {
        let path = temp_path("test_content_size.dat");
        let mgr = PageManager::open_with_page_size(&path, 256, 64).unwrap();
        let pn = mgr.allocate_page().unwrap();

        assert_eq!(mgr.page_content_size(pn).unwrap(), 0);

        mgr.write_to_page(pn, 0, b"12345").unwrap();
        assert_eq!(mgr.page_content_size(pn).unwrap(), 5);

        mgr.write_to_page(pn, 100, b"abc").unwrap();
        assert_eq!(mgr.page_content_size(pn).unwrap(), 103);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_data_capacity() {
        let path = temp_path("test_capacity.dat");
        let mgr = PageManager::open_with_page_size(&path, 256, 64).unwrap();
        assert_eq!(mgr.data_capacity(), 256 - PageHeader::SIZE);
        let _ = fs::remove_file(&path);
    }
}
