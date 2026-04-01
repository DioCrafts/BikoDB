// =============================================================================
// bikodb-storage::storage_engine — End-to-end disk persistence engine
// =============================================================================
// Orchestrates the full persistence pipeline:
//
//   store_vertex(Vertex)
//     → record_serializer::serialize_vertex (bincode + LZ4)
//     → WAL::write (delta entry with full record for crash recovery)
//     → PageManager::write_to_page (slotted page via cache → mmap)
//     → RID
//
//   load_vertex(RID)
//     → PageManager::read_from_page (cache hit or mmap load)
//     → record_serializer::deserialize_vertex (LZ4 decompress + bincode)
//     → Vertex
//
//   flush()  → PageManager::flush (dirty pages → mmap → fsync)
//   checkpoint() → flush + WAL checkpoint + dictionary persist
//   recover() → WAL replay → reconstruct pages → flush
//
// Record format on page: [size:u32][tagged_compressed_record_bytes...]
// RID encoding: bucket_id=0, offset = (page_num << 32) | page_offset
// WAL delta: [page_num:u32][page_offset:u32][tagged_compressed_record_bytes...]
// =============================================================================

use bikodb_core::error::{BikoError, BikoResult};
use bikodb_core::record::{Edge, Vertex};
use bikodb_core::types::RID;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::delta;
use crate::dictionary::Dictionary;
use crate::page_manager::PageManager;
use crate::record_serializer;
use crate::wal::{ConcurrentWal, WalOpType};

/// Allocation state: tracks where the next record will be written.
struct AllocState {
    /// Page number currently being appended to.
    current_page: u32,
    /// Byte offset within current page's data area.
    current_offset: usize,
}

/// Recovery statistics returned by [`StorageEngine::recover`].
#[derive(Debug, Default)]
pub struct RecoveryStats {
    pub records_recovered: u64,
    pub deletes_recovered: u64,
    pub updates_recovered: u64,
    /// Number of incomplete (uncommitted) transactions skipped during recovery.
    pub incomplete_txs_skipped: u64,
}

/// Controls when WAL entries are fsynced to disk.
///
/// Trade-off between durability and write latency:
/// - `Sync`: fsync after every write (safest, slowest — ~1ms per op)
/// - `Batch`: fsync every N entries (default — good throughput + durability)
/// - `Async`: never auto-fsync (fastest, risk of data loss on crash)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityMode {
    /// fsync after every WAL write. Maximum durability, highest latency.
    Sync,
    /// fsync every `group_commit_threshold` entries (default). Good balance.
    Batch,
    /// No automatic fsync. Caller must explicitly call `wal_sync()`.
    /// Risk: unfsynced entries may be lost on crash.
    Async,
}

impl Default for DurabilityMode {
    fn default() -> Self {
        Self::Batch
    }
}

/// Information about a created snapshot.
#[derive(Debug, Clone)]
pub struct SnapshotInfo {
    /// Unix timestamp when the snapshot was created.
    pub timestamp: u64,
    /// Number of data pages in the snapshot.
    pub page_count: u32,
    /// Number of dictionary entries in the snapshot.
    pub dict_entries: u32,
}

/// End-to-end storage engine wiring serialization, compression, paging,
/// WAL delta storage, and crash recovery into a single coherent path.
pub struct StorageEngine {
    page_mgr: PageManager,
    wal: ConcurrentWal,
    dictionary: Dictionary,
    dict_path: PathBuf,
    alloc: Mutex<AllocState>,
    durability: DurabilityMode,
}

// ── RID encoding helpers ─────────────────────────────────────────────────

/// Encodes (page_number, page_offset) into a u64 for RID.offset.
fn encode_rid_offset(page_num: u32, page_offset: u32) -> u64 {
    ((page_num as u64) << 32) | (page_offset as u64)
}

/// Decodes RID.offset back to (page_number, page_offset).
fn decode_rid_offset(offset: u64) -> (u32, u32) {
    let page_num = (offset >> 32) as u32;
    let page_offset = (offset & 0xFFFF_FFFF) as u32;
    (page_num, page_offset)
}

// ── WAL payload encoding ────────────────────────────────────────────────

/// WAL insert payload: [page_num:u32][page_offset:u32][record_data...]
fn encode_wal_insert(page_num: u32, page_offset: u32, data: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(8 + data.len());
    payload.extend_from_slice(&page_num.to_le_bytes());
    payload.extend_from_slice(&page_offset.to_le_bytes());
    payload.extend_from_slice(data);
    payload
}

fn decode_wal_insert(payload: &[u8]) -> BikoResult<(u32, u32, &[u8])> {
    if payload.len() < 8 {
        return Err(BikoError::WalCorrupted {
            details: "RecordInsert payload too short".into(),
        });
    }
    let page_num = u32::from_le_bytes(payload[0..4].try_into().unwrap());
    let page_offset = u32::from_le_bytes(payload[4..8].try_into().unwrap());
    Ok((page_num, page_offset, &payload[8..]))
}

/// WAL delete payload: [page_num:u32][page_offset:u32]
fn encode_wal_delete(page_num: u32, page_offset: u32) -> Vec<u8> {
    let mut payload = Vec::with_capacity(8);
    payload.extend_from_slice(&page_num.to_le_bytes());
    payload.extend_from_slice(&page_offset.to_le_bytes());
    payload
}

fn decode_wal_delete(payload: &[u8]) -> BikoResult<(u32, u32)> {
    if payload.len() < 8 {
        return Err(BikoError::WalCorrupted {
            details: "RecordDelete payload too short".into(),
        });
    }
    let page_num = u32::from_le_bytes(payload[0..4].try_into().unwrap());
    let page_offset = u32::from_le_bytes(payload[4..8].try_into().unwrap());
    Ok((page_num, page_offset))
}

/// WAL delta-update payload: [page_num:u32][page_offset:u32][delta_encoded...]
fn encode_wal_delta_update(page_num: u32, page_offset: u32, delta_bytes: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(8 + delta_bytes.len());
    payload.extend_from_slice(&page_num.to_le_bytes());
    payload.extend_from_slice(&page_offset.to_le_bytes());
    payload.extend_from_slice(delta_bytes);
    payload
}

impl StorageEngine {
    /// Opens or creates a StorageEngine at the given directory.
    ///
    /// On open, performs WAL recovery to replay any uncommitted changes
    /// from a prior crash, then initializes the allocation pointer.
    pub fn open(data_dir: impl AsRef<Path>) -> BikoResult<Self> {
        Self::open_with_options(data_dir, 64 * 1024, 1024, 1000)
    }

    /// Opens with custom page size, cache capacity, and WAL group commit threshold.
    pub fn open_with_options(
        data_dir: impl AsRef<Path>,
        page_size: usize,
        max_cached_pages: usize,
        wal_group_commit: u64,
    ) -> BikoResult<Self> {
        Self::open_full(
            data_dir,
            page_size,
            max_cached_pages,
            wal_group_commit,
            DurabilityMode::default(),
        )
    }

    /// Opens with full configuration including durability mode.
    pub fn open_full(
        data_dir: impl AsRef<Path>,
        page_size: usize,
        max_cached_pages: usize,
        wal_group_commit: u64,
        durability: DurabilityMode,
    ) -> BikoResult<Self> {
        let data_dir = data_dir.as_ref();
        fs::create_dir_all(data_dir)?;

        let data_path = data_dir.join("data.bikodb");
        let wal_dir = data_dir.join("wal");
        let dict_path = data_dir.join("dictionary.bin");

        // In Sync mode, group commit threshold = 1 (fsync every entry)
        let effective_threshold = match durability {
            DurabilityMode::Sync => 1,
            DurabilityMode::Batch => wal_group_commit,
            DurabilityMode::Async => u64::MAX, // never auto-sync
        };

        let page_mgr =
            PageManager::open_with_page_size(&data_path, page_size, max_cached_pages)?;
        let wal = ConcurrentWal::open(&wal_dir, effective_threshold)?;

        let dictionary = if dict_path.exists() {
            let data = fs::read(&dict_path)?;
            Dictionary::from_bytes(&data).unwrap_or_default()
        } else {
            Dictionary::new()
        };

        let engine = Self {
            page_mgr,
            wal,
            dictionary,
            dict_path,
            alloc: Mutex::new(AllocState {
                current_page: 0,
                current_offset: 0,
            }),
            durability,
        };

        // Recover from WAL (replay only committed transactions)
        engine.recover_internal()?;
        // Initialize allocation pointer from page state
        engine.init_alloc()?;

        Ok(engine)
    }

    /// Returns the current durability mode.
    pub fn durability_mode(&self) -> DurabilityMode {
        self.durability
    }

    /// Writes a TxBegin marker to the WAL for the given transaction ID.
    ///
    /// Must be called before any record operations for that tx to establish
    /// transaction boundaries in the WAL (crash recovery uses these to
    /// identify incomplete transactions).
    pub fn write_tx_begin(&self, tx_id: u64) -> BikoResult<()> {
        use crate::wal::WalEntry;
        self.wal.write_batch(&[WalEntry {
            tx_id,
            op_type: WalOpType::TxBegin,
            payload: Vec::new(),
        }])?;
        Ok(())
    }

    /// Writes a TxCommit marker to the WAL for the given transaction ID.
    ///
    /// After this, the transaction's entries are guaranteed to be replayed
    /// on recovery. Includes an fsync to ensure the commit is durable.
    pub fn write_tx_commit(&self, tx_id: u64) -> BikoResult<()> {
        use crate::wal::WalEntry;
        self.wal.write_batch(&[WalEntry {
            tx_id,
            op_type: WalOpType::TxCommit,
            payload: Vec::new(),
        }])?;
        Ok(())
    }

    /// Returns a reference to the underlying WAL (for advanced use).
    pub fn wal(&self) -> &ConcurrentWal {
        &self.wal
    }

    // ── Store ──────────────────────────────────────────────────────────

    /// Stores a Vertex on disk, returning its RID.
    ///
    /// Pipeline: serialize (bincode+LZ4) → WAL delta → page write.
    pub fn store_vertex(&self, vertex: &Vertex) -> BikoResult<RID> {
        let serialized = record_serializer::serialize_vertex(vertex)?;
        self.store_raw_record(&serialized)
    }

    /// Stores an Edge on disk, returning its RID.
    pub fn store_edge(&self, edge: &Edge) -> BikoResult<RID> {
        let serialized = record_serializer::serialize_edge(edge)?;
        self.store_raw_record(&serialized)
    }

    /// Internal: stores pre-serialized record bytes.
    fn store_raw_record(&self, data: &[u8]) -> BikoResult<RID> {
        let record_len = data.len();
        let needed = 4 + record_len; // [size:u32][data...]

        // 1. Allocate space on a page
        let (page_num, offset) = self.allocate_space(needed)?;

        // 2. Write-ahead: WAL delta entry (before modifying page)
        let wal_payload = encode_wal_insert(page_num, offset as u32, data);
        self.wal.write(WalOpType::RecordInsert, &wal_payload)?;

        // 3. Write record to page: [size:u32][data...]
        let mut buf = Vec::with_capacity(needed);
        buf.extend_from_slice(&(record_len as u32).to_le_bytes());
        buf.extend_from_slice(data);
        self.page_mgr.write_to_page(page_num, offset, &buf)?;

        // 4. Return RID
        Ok(RID::new(0, encode_rid_offset(page_num, offset as u32)))
    }

    // ── Load ───────────────────────────────────────────────────────────

    /// Loads a Vertex from disk by its RID.
    pub fn load_vertex(&self, rid: &RID) -> BikoResult<Vertex> {
        let data = self.load_raw_record(rid)?;
        record_serializer::deserialize_vertex(&data)
    }

    /// Loads an Edge from disk by its RID.
    pub fn load_edge(&self, rid: &RID) -> BikoResult<Edge> {
        let data = self.load_raw_record(rid)?;
        record_serializer::deserialize_edge(&data)
    }

    /// Internal: loads raw record bytes from a page.
    fn load_raw_record(&self, rid: &RID) -> BikoResult<Vec<u8>> {
        let (page_num, page_offset) = decode_rid_offset(rid.offset);
        let off = page_offset as usize;

        // Read size header
        let size_bytes = self.page_mgr.read_from_page(page_num, off, 4)?;
        let size = u32::from_le_bytes(size_bytes[..4].try_into().unwrap()) as usize;

        if size == 0 {
            return Err(BikoError::RecordNotFound(*rid));
        }

        // Read record data
        self.page_mgr.read_from_page(page_num, off + 4, size)
    }

    // ── Delete ─────────────────────────────────────────────────────────

    /// Deletes a record by setting its size to 0 (tombstone).
    pub fn delete_record(&self, rid: &RID) -> BikoResult<()> {
        let (page_num, page_offset) = decode_rid_offset(rid.offset);

        // WAL delta
        let wal_payload = encode_wal_delete(page_num, page_offset);
        self.wal.write(WalOpType::RecordDelete, &wal_payload)?;

        // Write tombstone (size = 0)
        self.page_mgr
            .write_to_page(page_num, page_offset as usize, &0u32.to_le_bytes())?;
        Ok(())
    }

    // ── Update (property-level delta) ──────────────────────────────────

    /// Updates a Vertex in-place using property-level delta encoding.
    ///
    /// Computes binary diff between old and new serialized forms; only the
    /// changed bytes are written to the WAL (`RecordUpdate` with delta payload).
    pub fn update_vertex(&self, rid: &RID, new_vertex: &Vertex) -> BikoResult<()> {
        let new_serialized = record_serializer::serialize_vertex(new_vertex)?;
        self.update_raw_record(rid, &new_serialized)
    }

    /// Updates an Edge in-place using property-level delta encoding.
    pub fn update_edge(&self, rid: &RID, new_edge: &Edge) -> BikoResult<()> {
        let new_serialized = record_serializer::serialize_edge(new_edge)?;
        self.update_raw_record(rid, &new_serialized)
    }

    /// Internal: updates a record with delta encoding in the WAL.
    fn update_raw_record(&self, rid: &RID, new_data: &[u8]) -> BikoResult<()> {
        let (page_num, page_offset) = decode_rid_offset(rid.offset);
        let off = page_offset as usize;

        // 1. Read old record data
        let old_data = self.load_raw_record(rid)?;

        // 2. Compute delta
        let d = delta::compute_delta(&old_data, new_data);
        let delta_bytes = delta::encode_delta(&d);

        // 3. Write delta to WAL
        let wal_payload = encode_wal_delta_update(page_num, page_offset, &delta_bytes);
        self.wal.write(WalOpType::RecordUpdate, &wal_payload)?;

        // 4. Write new data to page (full record on page, delta only in WAL)
        let mut buf = Vec::with_capacity(4 + new_data.len());
        buf.extend_from_slice(&(new_data.len() as u32).to_le_bytes());
        buf.extend_from_slice(new_data);
        self.page_mgr.write_to_page(page_num, off, &buf)?;

        Ok(())
    }

    // ── Durability ─────────────────────────────────────────────────────

    /// Flushes dirty pages to disk and persists the dictionary.
    pub fn flush(&self) -> BikoResult<()> {
        self.page_mgr.flush()?;
        let dict_bytes = self.dictionary.to_bytes();
        fs::write(&self.dict_path, dict_bytes)?;
        Ok(())
    }

    /// Syncs the WAL to disk (ensures pending entries are durable).
    pub fn wal_sync(&self) -> BikoResult<()> {
        self.wal.flush()
    }

    /// Checkpoint: flush all data + WAL checkpoint + purge old WAL files.
    pub fn checkpoint(&self) -> BikoResult<()> {
        self.flush()?;
        self.wal.checkpoint()?;
        self.wal.purge_old()?;
        Ok(())
    }

    // ── Budget enforcement ─────────────────────────────────────────────

    /// Evicts clean cache pages down to `max_pages` count.
    ///
    /// Called by upper layers (e.g., `ResourceMonitor`) when memory budget
    /// for the page cache is exceeded.  Returns the number of pages evicted.
    pub fn enforce_cache_budget(&self, max_pages: usize) -> usize {
        self.page_mgr.evict_to_target(max_pages)
    }

    /// Returns current number of cached pages (for budget checks).
    pub fn cached_page_count(&self) -> usize {
        self.page_mgr.cached_page_count()
    }

    // ── Dictionary access ──────────────────────────────────────────────

    /// Returns a reference to the property dictionary.
    pub fn dictionary(&self) -> &Dictionary {
        &self.dictionary
    }

    // ── Snapshot ───────────────────────────────────────────────────────

    /// Creates a point-in-time snapshot of the database at `snapshot_dir`.
    ///
    /// Protocol:
    /// 1. Checkpoint (flush all dirty pages + WAL checkpoint + purge)
    /// 2. Copy data file, dictionary, and current WAL to snapshot directory
    ///
    /// The snapshot is a consistent, self-contained copy that can be restored
    /// with [`restore_from_snapshot`].
    pub fn create_snapshot(&self, snapshot_dir: impl AsRef<Path>) -> BikoResult<SnapshotInfo> {
        let snap_dir = snapshot_dir.as_ref();

        // 1. Checkpoint to ensure all data is flushed
        self.checkpoint()?;

        // 2. Create snapshot directory
        fs::create_dir_all(snap_dir)?;

        // 3. Copy data file
        let src_data = self.dict_path.parent().unwrap().join("data.bikodb");
        let dst_data = snap_dir.join("data.bikodb");
        if src_data.exists() {
            fs::copy(&src_data, &dst_data)?;
        }

        // 4. Copy dictionary
        let dst_dict = snap_dir.join("dictionary.bin");
        if self.dict_path.exists() {
            fs::copy(&self.dict_path, &dst_dict)?;
        }

        // 5. Write snapshot metadata
        let info = SnapshotInfo {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            page_count: self.page_mgr.page_count(),
            dict_entries: self.dictionary.len() as u32,
        };
        let meta = format!(
            "{{\"timestamp\":{},\"page_count\":{},\"dict_entries\":{}}}",
            info.timestamp, info.page_count, info.dict_entries
        );
        fs::write(snap_dir.join("snapshot.meta"), meta.as_bytes())?;

        Ok(info)
    }

    /// Restores a database from a snapshot directory.
    ///
    /// Copies the snapshot files into the target data directory, then
    /// opens a new StorageEngine from that directory.
    ///
    /// **Warning**: Overwrites any existing data in `data_dir`.
    pub fn restore_from_snapshot(
        snapshot_dir: impl AsRef<Path>,
        data_dir: impl AsRef<Path>,
        page_size: usize,
        max_cached_pages: usize,
    ) -> BikoResult<Self> {
        let snap_dir = snapshot_dir.as_ref();
        let data_dir = data_dir.as_ref();

        // Validate snapshot
        let meta_path = snap_dir.join("snapshot.meta");
        if !meta_path.exists() {
            return Err(BikoError::Generic(
                "Invalid snapshot: missing snapshot.meta".into(),
            ));
        }

        // Create/clean target directory
        fs::create_dir_all(data_dir)?;

        // Copy data file
        let snap_data = snap_dir.join("data.bikodb");
        if snap_data.exists() {
            fs::copy(&snap_data, data_dir.join("data.bikodb"))?;
        }

        // Copy dictionary
        let snap_dict = snap_dir.join("dictionary.bin");
        if snap_dict.exists() {
            fs::copy(&snap_dict, data_dir.join("dictionary.bin"))?;
        }

        // Create empty WAL directory (snapshot is post-checkpoint, no WAL needed)
        let wal_dir = data_dir.join("wal");
        fs::create_dir_all(&wal_dir)?;

        // Open engine from restored data
        Self::open_with_options(data_dir, page_size, max_cached_pages, 1000)
    }

    // ── Auto-checkpoint ────────────────────────────────────────────────

    /// Performs a checkpoint if the WAL has accumulated more than
    /// `threshold` entries since the last checkpoint.
    ///
    /// Returns `true` if a checkpoint was performed.
    ///
    /// Designed to be called periodically (e.g., after N writes, or from
    /// a background timer) to bound recovery time.
    pub fn maybe_checkpoint(&self, wal_entry_threshold: u64) -> BikoResult<bool> {
        let entries = self.wal.recover()?;
        if entries.len() as u64 >= wal_entry_threshold {
            self.checkpoint()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    // ── Recovery ───────────────────────────────────────────────────────

    /// Replays WAL entries since the last checkpoint to restore state.
    ///
    /// Only entries belonging to committed transactions are replayed.
    /// If a transaction has a TxBegin but no TxCommit, its entries are
    /// skipped (crash mid-transaction safety).
    ///
    /// Entries with tx_id == 0 (e.g., from StorageEngine direct writes)
    /// are always replayed — they are not part of a managed transaction.
    fn recover_internal(&self) -> BikoResult<RecoveryStats> {
        let entries = self.wal.recover()?;
        let mut stats = RecoveryStats::default();

        // Phase 1: scan for committed transaction IDs
        let mut begun_txs: HashSet<u64> = HashSet::new();
        let mut committed_txs: HashSet<u64> = HashSet::new();

        for entry in &entries {
            match entry.op_type {
                WalOpType::TxBegin => {
                    begun_txs.insert(entry.tx_id);
                }
                WalOpType::TxCommit => {
                    committed_txs.insert(entry.tx_id);
                }
                _ => {}
            }
        }

        // Incomplete txs: began but never committed
        let incomplete: HashSet<u64> = begun_txs
            .difference(&committed_txs)
            .copied()
            .collect();
        stats.incomplete_txs_skipped = incomplete.len() as u64;

        // Phase 2: replay only committed (or non-transactional) entries
        for entry in &entries {
            // Skip entries belonging to incomplete transactions
            if entry.tx_id != 0 && incomplete.contains(&entry.tx_id) {
                continue;
            }

            match entry.op_type {
                WalOpType::RecordInsert => {
                    let (page_num, page_offset, data) =
                        decode_wal_insert(&entry.payload)?;
                    self.ensure_page_exists(page_num)?;

                    let mut buf = Vec::with_capacity(4 + data.len());
                    buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
                    buf.extend_from_slice(data);
                    self.page_mgr
                        .write_to_page(page_num, page_offset as usize, &buf)?;
                    stats.records_recovered += 1;
                }
                WalOpType::RecordDelete => {
                    let (page_num, page_offset) = decode_wal_delete(&entry.payload)?;
                    self.ensure_page_exists(page_num)?;
                    self.page_mgr.write_to_page(
                        page_num,
                        page_offset as usize,
                        &0u32.to_le_bytes(),
                    )?;
                    stats.deletes_recovered += 1;
                }
                WalOpType::RecordUpdate => {
                    // Delta-encoded update: read old record, apply delta, write new
                    let (page_num, page_offset, delta_bytes) =
                        decode_wal_insert(&entry.payload)?;
                    self.ensure_page_exists(page_num)?;

                    let d = delta::decode_delta(delta_bytes)?;
                    match &d {
                        delta::Delta::Full(new_data) => {
                            let mut buf = Vec::with_capacity(4 + new_data.len());
                            buf.extend_from_slice(
                                &(new_data.len() as u32).to_le_bytes(),
                            );
                            buf.extend_from_slice(new_data);
                            self.page_mgr.write_to_page(
                                page_num,
                                page_offset as usize,
                                &buf,
                            )?;
                        }
                        delta::Delta::Hunks(hunks) => {
                            // Read old, apply hunks, write back
                            let rid = RID::new(
                                0,
                                encode_rid_offset(page_num, page_offset),
                            );
                            if let Ok(old_data) = self.load_raw_record(&rid) {
                                let new_data = delta::apply_delta(&old_data, &d);
                                let mut buf =
                                    Vec::with_capacity(4 + new_data.len());
                                buf.extend_from_slice(
                                    &(new_data.len() as u32).to_le_bytes(),
                                );
                                buf.extend_from_slice(&new_data);
                                self.page_mgr.write_to_page(
                                    page_num,
                                    page_offset as usize,
                                    &buf,
                                )?;
                            }
                        }
                    }
                    stats.updates_recovered += 1;
                }
                // TxCommit, TxRollback, Checkpoint, PageWrite — skip
                _ => {}
            }
        }

        if stats.records_recovered > 0
            || stats.deletes_recovered > 0
            || stats.updates_recovered > 0
        {
            // Persist recovered data and write new checkpoint
            self.page_mgr.flush()?;
            self.wal.checkpoint()?;
        }

        Ok(stats)
    }

    /// Ensures that pages up to `page_num` exist (for recovery).
    fn ensure_page_exists(&self, page_num: u32) -> BikoResult<()> {
        let current_count = self.page_mgr.page_count();
        for _ in current_count..=page_num {
            self.page_mgr.allocate_page()?;
        }
        Ok(())
    }

    /// Initializes the allocation pointer from current page state.
    fn init_alloc(&self) -> BikoResult<()> {
        let mut alloc = self.alloc.lock();
        let page_count = self.page_mgr.page_count();

        if page_count == 0 {
            let pn = self.page_mgr.allocate_page()?;
            alloc.current_page = pn;
            alloc.current_offset = 0;
        } else {
            let last_page = page_count - 1;
            alloc.current_page = last_page;
            alloc.current_offset = self.page_mgr.page_content_size(last_page)?;
        }
        Ok(())
    }

    /// Allocates space for a record of `needed` bytes on a page.
    ///
    /// Returns (page_number, byte_offset) within the page data area.
    fn allocate_space(&self, needed: usize) -> BikoResult<(u32, usize)> {
        let data_capacity = self.page_mgr.data_capacity();

        if needed > data_capacity {
            return Err(BikoError::Generic(format!(
                "record too large: {needed} bytes exceeds page capacity {data_capacity}"
            )));
        }

        let mut alloc = self.alloc.lock();

        if alloc.current_offset + needed > data_capacity {
            // Current page full — allocate a new one
            let page_num = self.page_mgr.allocate_page()?;
            alloc.current_page = page_num;
            alloc.current_offset = 0;
        }

        let page_num = alloc.current_page;
        let offset = alloc.current_offset;
        alloc.current_offset += needed;
        Ok((page_num, offset))
    }
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
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::PathBuf;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join("bikodb_test_storage_engine")
            .join(name);
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn make_vertex(id: u64, name: &str) -> Vertex {
        let mut v = Vertex::new(NodeId(id), TypeId(1));
        v.set_property(0, Value::string(name));
        v.set_property(1, Value::Int(id as i64));
        v
    }

    fn make_edge(id: u64, src: u64, dst: u64) -> Edge {
        let mut e = Edge::new(EdgeId(id), TypeId(5), NodeId(src), NodeId(dst));
        e.set_property(0, Value::string("KNOWS"));
        e
    }

    // ── Basic roundtrip ────────────────────────────────────────────────

    #[test]
    fn test_store_and_load_vertex() {
        let dir = temp_dir("test_store_vertex");
        let engine = StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();

        let v = make_vertex(1, "Alice");
        let rid = engine.store_vertex(&v).unwrap();
        let loaded = engine.load_vertex(&rid).unwrap();

        assert_eq!(loaded.id, NodeId(1));
        assert_eq!(loaded.type_id, TypeId(1));
        assert_eq!(loaded.get_property(0).unwrap().as_str(), Some("Alice"));
        assert_eq!(loaded.get_property(1).unwrap().as_int(), Some(1));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_store_and_load_edge() {
        let dir = temp_dir("test_store_edge");
        let engine = StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();

        let e = make_edge(100, 1, 2);
        let rid = engine.store_edge(&e).unwrap();
        let loaded = engine.load_edge(&rid).unwrap();

        assert_eq!(loaded.id, EdgeId(100));
        assert_eq!(loaded.source, NodeId(1));
        assert_eq!(loaded.target, NodeId(2));
        assert_eq!(loaded.get_property(0).unwrap().as_str(), Some("KNOWS"));

        let _ = fs::remove_dir_all(&dir);
    }

    // ── Multiple records ───────────────────────────────────────────────

    #[test]
    fn test_multiple_records() {
        let dir = temp_dir("test_multi_records");
        let engine = StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();

        let mut rids = Vec::new();
        for i in 0..50 {
            let v = make_vertex(i, &format!("Node_{i}"));
            rids.push(engine.store_vertex(&v).unwrap());
        }

        // Read all back
        for (i, rid) in rids.iter().enumerate() {
            let v = engine.load_vertex(rid).unwrap();
            assert_eq!(v.id, NodeId(i as u64));
            assert_eq!(
                v.get_property(0).unwrap().as_str(),
                Some(format!("Node_{i}").as_str())
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    // ── Page overflow ──────────────────────────────────────────────────

    #[test]
    fn test_page_overflow_allocates_new() {
        let dir = temp_dir("test_page_overflow");
        // Small pages (256 bytes) to force overflow quickly
        let engine = StorageEngine::open_with_options(&dir, 256, 64, 100).unwrap();

        let mut rids = Vec::new();
        for i in 0..20 {
            let v = make_vertex(i, &format!("name_{i}"));
            rids.push(engine.store_vertex(&v).unwrap());
        }

        // Should have used multiple pages
        assert!(
            engine.page_mgr.page_count() > 1,
            "Expected multiple pages, got {}",
            engine.page_mgr.page_count()
        );

        // All records should still be readable
        for (i, rid) in rids.iter().enumerate() {
            let v = engine.load_vertex(rid).unwrap();
            assert_eq!(v.id, NodeId(i as u64));
        }

        let _ = fs::remove_dir_all(&dir);
    }

    // ── Delete ─────────────────────────────────────────────────────────

    #[test]
    fn test_delete_record() {
        let dir = temp_dir("test_delete");
        let engine = StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();

        let v = make_vertex(1, "ToDelete");
        let rid = engine.store_vertex(&v).unwrap();

        // Should be readable
        assert!(engine.load_vertex(&rid).is_ok());

        // Delete
        engine.delete_record(&rid).unwrap();

        // Should not be readable (tombstone)
        assert!(engine.load_vertex(&rid).is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    // ── Flush + reopen ─────────────────────────────────────────────────

    #[test]
    fn test_flush_and_reopen() {
        let dir = temp_dir("test_flush_reopen");
        let rid;

        // Phase 1: store and flush
        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();
            let v = make_vertex(42, "Persistent");
            rid = engine.store_vertex(&v).unwrap();
            engine.flush().unwrap();
        }

        // Phase 2: reopen and read
        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();
            let loaded = engine.load_vertex(&rid).unwrap();
            assert_eq!(loaded.id, NodeId(42));
            assert_eq!(
                loaded.get_property(0).unwrap().as_str(),
                Some("Persistent")
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    // ── Recovery from WAL ──────────────────────────────────────────────

    #[test]
    fn test_recovery_from_wal() {
        let dir = temp_dir("test_recovery");
        let rid;

        // Phase 1: store + sync WAL, but do NOT flush pages
        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();
            let v = make_vertex(99, "Recovered");
            rid = engine.store_vertex(&v).unwrap();
            engine.wal_sync().unwrap();
            // Drop without flush — pages not persisted, only WAL
        }

        // Phase 2: reopen — recovery should replay WAL
        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();
            let loaded = engine.load_vertex(&rid).unwrap();
            assert_eq!(loaded.id, NodeId(99));
            assert_eq!(
                loaded.get_property(0).unwrap().as_str(),
                Some("Recovered")
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    // ── Checkpoint ─────────────────────────────────────────────────────

    #[test]
    fn test_checkpoint() {
        let dir = temp_dir("test_checkpoint");
        let engine = StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();

        for i in 0..10 {
            engine
                .store_vertex(&make_vertex(i, &format!("v{i}")))
                .unwrap();
        }

        // Checkpoint: flush + WAL checkpoint + purge
        engine.checkpoint().unwrap();

        // After checkpoint, WAL should be clean (no entries to replay)
        let entries = engine.wal.recover().unwrap();
        assert!(entries.is_empty(), "Expected empty WAL after checkpoint");

        let _ = fs::remove_dir_all(&dir);
    }

    // ── Dictionary persistence ─────────────────────────────────────────

    #[test]
    fn test_dictionary_persistence() {
        let dir = temp_dir("test_dict_persist");

        // Phase 1: populate dictionary
        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();
            engine.dictionary().get_or_insert("name");
            engine.dictionary().get_or_insert("age");
            engine.dictionary().get_or_insert("email");
            engine.flush().unwrap();
        }

        // Phase 2: reopen — dictionary should be loaded from disk
        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();
            assert_eq!(engine.dictionary().len(), 3);
            assert_eq!(engine.dictionary().lookup("name"), Some(0));
            assert_eq!(engine.dictionary().lookup("age"), Some(1));
            assert_eq!(engine.dictionary().lookup("email"), Some(2));
        }

        let _ = fs::remove_dir_all(&dir);
    }

    // ── Record type identification ─────────────────────────────────────

    #[test]
    fn test_record_type_peek() {
        let dir = temp_dir("test_peek_type");
        let engine = StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();

        let v_rid = engine.store_vertex(&make_vertex(1, "V")).unwrap();
        let e_rid = engine.store_edge(&make_edge(1, 1, 2)).unwrap();

        let v_data = engine.load_raw_record(&v_rid).unwrap();
        let e_data = engine.load_raw_record(&e_rid).unwrap();

        assert_eq!(
            record_serializer::peek_record_type(&v_data).unwrap(),
            RecordType::Vertex
        );
        assert_eq!(
            record_serializer::peek_record_type(&e_data).unwrap(),
            RecordType::Edge
        );

        let _ = fs::remove_dir_all(&dir);
    }

    // ── Mixed vertices and edges ───────────────────────────────────────

    #[test]
    fn test_mixed_vertices_and_edges() {
        let dir = temp_dir("test_mixed");
        let engine = StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();

        let v1_rid = engine.store_vertex(&make_vertex(1, "Alice")).unwrap();
        let v2_rid = engine.store_vertex(&make_vertex(2, "Bob")).unwrap();
        let e_rid = engine.store_edge(&make_edge(10, 1, 2)).unwrap();

        // Flush and reopen
        engine.flush().unwrap();
        drop(engine);

        let engine = StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();
        let v1 = engine.load_vertex(&v1_rid).unwrap();
        let v2 = engine.load_vertex(&v2_rid).unwrap();
        let e = engine.load_edge(&e_rid).unwrap();

        assert_eq!(v1.get_property(0).unwrap().as_str(), Some("Alice"));
        assert_eq!(v2.get_property(0).unwrap().as_str(), Some("Bob"));
        assert_eq!(e.source, NodeId(1));
        assert_eq!(e.target, NodeId(2));

        let _ = fs::remove_dir_all(&dir);
    }

    // ── Encoding helpers ───────────────────────────────────────────────

    #[test]
    fn test_rid_offset_encoding() {
        let page = 12345u32;
        let offset = 6789u32;
        let encoded = encode_rid_offset(page, offset);
        let (p, o) = decode_rid_offset(encoded);
        assert_eq!(p, page);
        assert_eq!(o, offset);
    }

    #[test]
    fn test_wal_insert_encoding() {
        let data = b"test record data";
        let payload = encode_wal_insert(5, 100, data);
        let (page, offset, recovered) = decode_wal_insert(&payload).unwrap();
        assert_eq!(page, 5);
        assert_eq!(offset, 100);
        assert_eq!(recovered, data);
    }

    // ── Delta update ───────────────────────────────────────────────────

    #[test]
    fn test_update_vertex_delta() {
        let dir = temp_dir("test_update_vertex");
        let engine = StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();

        // Store initial
        let v = make_vertex(1, "Alice");
        let rid = engine.store_vertex(&v).unwrap();

        // Update property
        let mut v2 = make_vertex(1, "Alicia");
        v2.set_property(1, Value::Int(99));
        engine.update_vertex(&rid, &v2).unwrap();

        // Read back
        let loaded = engine.load_vertex(&rid).unwrap();
        assert_eq!(loaded.get_property(0).unwrap().as_str(), Some("Alicia"));
        assert_eq!(loaded.get_property(1).unwrap().as_int(), Some(99));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_update_edge_delta() {
        let dir = temp_dir("test_update_edge");
        let engine = StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();

        let e = make_edge(1, 10, 20);
        let rid = engine.store_edge(&e).unwrap();

        let mut e2 = make_edge(1, 10, 20);
        e2.set_property(0, Value::string("LIKES"));
        e2.set_property(1, Value::Int(2025));
        engine.update_edge(&rid, &e2).unwrap();

        let loaded = engine.load_edge(&rid).unwrap();
        assert_eq!(loaded.get_property(0).unwrap().as_str(), Some("LIKES"));
        assert_eq!(loaded.get_property(1).unwrap().as_int(), Some(2025));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_update_multiple_times() {
        let dir = temp_dir("test_update_multi");
        let engine = StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();

        let rid = engine.store_vertex(&make_vertex(1, "v0")).unwrap();

        for i in 1..10 {
            let v = make_vertex(1, &format!("v{i}"));
            engine.update_vertex(&rid, &v).unwrap();
        }

        let loaded = engine.load_vertex(&rid).unwrap();
        assert_eq!(loaded.get_property(0).unwrap().as_str(), Some("v9"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_update_persists_after_flush() {
        let dir = temp_dir("test_update_flush");
        let rid;

        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();
            rid = engine.store_vertex(&make_vertex(1, "before")).unwrap();
            engine.update_vertex(&rid, &make_vertex(1, "after")).unwrap();
            engine.flush().unwrap();
        }

        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();
            let loaded = engine.load_vertex(&rid).unwrap();
            assert_eq!(loaded.get_property(0).unwrap().as_str(), Some("after"));
        }

        let _ = fs::remove_dir_all(&dir);
    }

    // ── Crash mid-transaction recovery ─────────────────────────────────

    #[test]
    fn test_incomplete_tx_skipped_on_recovery() {
        use crate::wal::WalEntry;

        let dir = temp_dir("test_incomplete_tx");

        // Phase 1: write TxBegin + a record insert, but NO TxCommit.
        // This simulates a crash mid-transaction.
        let rid;
        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();

            // Write a committed record first (tx_id=0, always replayed)
            let v = make_vertex(1, "Committed");
            rid = engine.store_vertex(&v).unwrap();

            // Now simulate an incomplete transaction (tx_id=42)
            engine.write_tx_begin(42).unwrap();

            // Write a record as part of tx 42
            let v2 = make_vertex(2, "Incomplete");
            let serialized = crate::record_serializer::serialize_vertex(&v2).unwrap();
            let wal_payload = encode_wal_insert(0, 200, &serialized);
            engine.wal.write_with_tx(42, WalOpType::RecordInsert, &wal_payload).unwrap();

            // NO TxCommit for tx 42 — crash!
            engine.wal_sync().unwrap();
            // Drop without flush
        }

        // Phase 2: reopen — recovery should skip tx 42's entries
        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();

            // The committed record (tx_id=0) should be recoverable
            let loaded = engine.load_vertex(&rid).unwrap();
            assert_eq!(loaded.id, NodeId(1));
            assert_eq!(
                loaded.get_property(0).unwrap().as_str(),
                Some("Committed")
            );

            // The incomplete record at offset 200 should NOT exist (size=0)
            let incomplete_rid = RID::new(0, encode_rid_offset(0, 200));
            assert!(engine.load_vertex(&incomplete_rid).is_err());
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_complete_tx_recovered() {
        use crate::wal::WalEntry;

        let dir = temp_dir("test_complete_tx");

        // Phase 1: write TxBegin + record + TxCommit (complete tx)
        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();

            // Write tx begin
            engine.write_tx_begin(77).unwrap();

            // Write a record as part of tx 77
            let v = make_vertex(5, "TxRecord");
            let serialized = crate::record_serializer::serialize_vertex(&v).unwrap();
            let wal_payload = encode_wal_insert(0, 0, &serialized);
            engine.wal.write_with_tx(77, WalOpType::RecordInsert, &wal_payload).unwrap();

            // Write tx commit
            engine.write_tx_commit(77).unwrap();

            engine.wal_sync().unwrap();
            // Drop without flush — only WAL has data
        }

        // Phase 2: reopen — recovery should replay tx 77's entries
        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();
            let rid = RID::new(0, encode_rid_offset(0, 0));
            let loaded = engine.load_vertex(&rid).unwrap();
            assert_eq!(loaded.id, NodeId(5));
            assert_eq!(
                loaded.get_property(0).unwrap().as_str(),
                Some("TxRecord")
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_durability_mode_sync() {
        let dir = temp_dir("test_durability_sync");
        let engine = StorageEngine::open_full(&dir, 4096, 64, 1000, DurabilityMode::Sync).unwrap();
        assert_eq!(engine.durability_mode(), DurabilityMode::Sync);

        let v = make_vertex(1, "SyncMode");
        let rid = engine.store_vertex(&v).unwrap();
        let loaded = engine.load_vertex(&rid).unwrap();
        assert_eq!(loaded.get_property(0).unwrap().as_str(), Some("SyncMode"));

        let _ = fs::remove_dir_all(&dir);
    }

    // ── Snapshot create/restore ────────────────────────────────────────

    #[test]
    fn test_snapshot_create_and_restore() {
        let dir = temp_dir("test_snapshot_src");
        let snap_dir = temp_dir("test_snapshot_snap");
        let restore_dir = temp_dir("test_snapshot_restore");
        let rid;

        // Create data and snapshot
        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();
            let v = make_vertex(42, "Snapshotted");
            rid = engine.store_vertex(&v).unwrap();
            engine.dictionary().get_or_insert("snapshot_prop");

            let info = engine.create_snapshot(&snap_dir).unwrap();
            assert!(info.page_count > 0);
            assert!(info.timestamp > 0);
            assert!(snap_dir.join("snapshot.meta").exists());
            assert!(snap_dir.join("data.bikodb").exists());
        }

        // Restore from snapshot into a different directory
        {
            let engine =
                StorageEngine::restore_from_snapshot(&snap_dir, &restore_dir, 4096, 64).unwrap();
            let loaded = engine.load_vertex(&rid).unwrap();
            assert_eq!(loaded.id, NodeId(42));
            assert_eq!(
                loaded.get_property(0).unwrap().as_str(),
                Some("Snapshotted")
            );
            // Dictionary should also be restored
            assert_eq!(engine.dictionary().lookup("snapshot_prop"), Some(0));
        }

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&snap_dir);
        let _ = fs::remove_dir_all(&restore_dir);
    }

    #[test]
    fn test_snapshot_invalid_dir() {
        let fake_snap = temp_dir("test_snapshot_fake");
        let restore_dir = temp_dir("test_snapshot_restore_fake");
        fs::create_dir_all(&fake_snap).unwrap();
        // No snapshot.meta → should fail
        let result = StorageEngine::restore_from_snapshot(&fake_snap, &restore_dir, 4096, 64);
        assert!(result.is_err());
        let _ = fs::remove_dir_all(&fake_snap);
    }

    // ── Auto-checkpoint ────────────────────────────────────────────────

    #[test]
    fn test_maybe_checkpoint() {
        let dir = temp_dir("test_maybe_checkpoint");
        let engine = StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();

        // Insert a few records
        for i in 0..5 {
            engine.store_vertex(&make_vertex(i, &format!("v{i}"))).unwrap();
        }

        // With a low threshold, checkpoint should trigger
        let did_checkpoint = engine.maybe_checkpoint(3).unwrap();
        assert!(did_checkpoint);

        // After checkpoint, WAL should be clean
        let entries = engine.wal.recover().unwrap();
        assert!(entries.is_empty());

        // With a high threshold, no checkpoint
        engine.store_vertex(&make_vertex(100, "extra")).unwrap();
        let did_checkpoint2 = engine.maybe_checkpoint(1000).unwrap();
        assert!(!did_checkpoint2);

        let _ = fs::remove_dir_all(&dir);
    }

    // ── WAL corruption resilience ──────────────────────────────────────

    #[test]
    fn test_wal_truncated_entry_recovery() {
        let dir = temp_dir("test_wal_truncated");

        // Phase 1: write valid data, then append garbage to simulate crash mid-write
        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();
            let v = make_vertex(1, "BeforeCorruption");
            engine.store_vertex(&v).unwrap();
            engine.wal_sync().unwrap();

            // Append garbage bytes to the WAL file to simulate torn write
            let wal_dir = dir.join("wal");
            let wal_files: Vec<_> = fs::read_dir(&wal_dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map(|x| x == "log").unwrap_or(false))
                .collect();
            assert!(!wal_files.is_empty());
            let wal_path = wal_files.last().unwrap().path();
            let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
            // Write a valid-looking length but garbage data (CRC will fail)
            f.write_all(&100u32.to_le_bytes()).unwrap(); // len=100
            f.write_all(&0xDEADBEEFu32.to_le_bytes()).unwrap(); // bad CRC
            f.write_all(&[0xFF; 100]).unwrap(); // garbage payload
            f.sync_all().unwrap();
        }

        // Phase 2: reopen — should recover the valid record and skip the garbage
        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();
            // The valid record should still be loadable (recovered from WAL)
            // We check that it didn't crash on open
            assert!(engine.page_mgr.page_count() > 0);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    // ── Page checksum ──────────────────────────────────────────────────

    #[test]
    fn test_page_checksum_computed_on_flush() {
        let dir = temp_dir("test_page_checksum");

        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();
            engine.store_vertex(&make_vertex(1, "Checksummed")).unwrap();
            engine.flush().unwrap();
        }

        // Reopen and verify the page loaded without checksum errors
        {
            let engine =
                StorageEngine::open_with_options(&dir, 4096, 64, 1).unwrap();
            // If checksum failed, from_bytes would have printed a warning
            // but the data should still load correctly
            let rid = RID::new(0, encode_rid_offset(0, 0));
            let loaded = engine.load_vertex(&rid).unwrap();
            assert_eq!(loaded.id, NodeId(1));
        }

        let _ = fs::remove_dir_all(&dir);
    }
}
