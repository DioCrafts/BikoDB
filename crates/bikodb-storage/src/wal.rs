// =============================================================================
// bikodb-storage::wal — Write-Ahead Log con delta storage
// =============================================================================
// WAL append-only para crash recovery y durabilidad.
//
// Diseño:
// - Cada entrada: [len:u32][crc:u32][tx_id:u64][op_type:u8][payload...]
// - CRC32 covers tx_id + op_type + payload for integrity
// - Rotación al alcanzar MAX_WAL_FILE_SIZE (64MB)
// - Recovery: replay secuencial desde último checkpoint
// - fsync configurable (per-commit o batch)
// - Corruption resilience: skip-and-continue on CRC mismatch
//
// Inspirado en ArcadeDB WAL (append-only, checkpointing, delta format).
// =============================================================================

use bikodb_core::config::MAX_WAL_FILE_SIZE;
use bikodb_core::error::{BikoError, BikoResult};
use parking_lot::Mutex;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// WAL file magic number: "OxWL" in little-endian.
const WAL_MAGIC: u32 = 0x4C57_784F; // "OxWL"
/// WAL format version.
const WAL_VERSION: u32 = 2;
/// WAL file header size: magic(4) + version(4).
const WAL_HEADER_SIZE: usize = 8;

/// Tipos de operación registrados en el WAL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalOpType {
    /// Escritura de página (page_id + data delta)
    PageWrite = 0,
    /// Commit de transacción
    TxCommit = 1,
    /// Rollback de transacción
    TxRollback = 2,
    /// Checkpoint (marca de consistencia)
    Checkpoint = 3,
    /// Inserción de registro: payload = [page_num:u32][offset:u32][record_data...]
    RecordInsert = 4,
    /// Eliminación de registro: payload = [page_num:u32][offset:u32]
    RecordDelete = 5,
    /// Actualización de registro: payload = [page_num:u32][offset:u32][new_record_data...]
    RecordUpdate = 6,
    /// Inicio de transacción (boundary marker for crash recovery)
    TxBegin = 7,
}

impl WalOpType {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::PageWrite),
            1 => Some(Self::TxCommit),
            2 => Some(Self::TxRollback),
            3 => Some(Self::Checkpoint),
            4 => Some(Self::RecordInsert),
            5 => Some(Self::RecordDelete),
            6 => Some(Self::RecordUpdate),
            7 => Some(Self::TxBegin),
            _ => None,
        }
    }
}

/// Una entrada individual del WAL.
#[derive(Debug, Clone)]
pub struct WalEntry {
    pub tx_id: u64,
    pub op_type: WalOpType,
    pub payload: Vec<u8>,
}

/// Computes FNV-1a 32-bit checksum (same as codec::checksum).
fn wal_crc(data: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &byte in data {
        h ^= byte as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

impl WalEntry {
    /// Serializa la entrada a bytes.
    ///
    /// Formato: [total_len:u32][crc:u32][tx_id:u64][op_type:u8][payload...]
    /// CRC covers the bytes after the crc field (tx_id + op_type + payload).
    pub fn to_bytes(&self) -> Vec<u8> {
        let payload_len = self.payload.len();
        let total_len = 8 + 1 + payload_len; // tx_id + op_type + payload
        let mut buf = Vec::with_capacity(4 + 4 + total_len);

        buf.extend_from_slice(&(total_len as u32).to_le_bytes());

        // Build the data to CRC (tx_id + op_type + payload)
        let mut crc_data = Vec::with_capacity(total_len);
        crc_data.extend_from_slice(&self.tx_id.to_le_bytes());
        crc_data.push(self.op_type as u8);
        crc_data.extend_from_slice(&self.payload);

        let crc = wal_crc(&crc_data);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf.extend_from_slice(&crc_data);
        buf
    }

    /// Deserializa una entrada desde un reader.
    ///
    /// Returns `Ok(None)` on clean EOF.
    /// Returns `Err(WalCorrupted)` on CRC mismatch (caller may skip-and-continue).
    fn read_from(reader: &mut impl Read) -> BikoResult<Option<Self>> {
        let mut len_buf = [0u8; 4];
        match reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(BikoError::Io(e)),
        }
        let total_len = u32::from_le_bytes(len_buf) as usize;
        if total_len < 9 {
            return Err(BikoError::WalCorrupted {
                details: format!("WAL entry too short: len={total_len}"),
            });
        }

        // Read CRC
        let mut crc_buf = [0u8; 4];
        match reader.read_exact(&mut crc_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(BikoError::Io(e)),
        }
        let stored_crc = u32::from_le_bytes(crc_buf);

        // Read the data (tx_id + op_type + payload)
        let mut data = vec![0u8; total_len];
        match reader.read_exact(&mut data) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(BikoError::Io(e)),
        }

        // Verify CRC
        let computed_crc = wal_crc(&data);
        if computed_crc != stored_crc {
            return Err(BikoError::WalCorrupted {
                details: format!(
                    "CRC mismatch: stored=0x{stored_crc:08X}, computed=0x{computed_crc:08X}"
                ),
            });
        }

        let tx_id = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let op_type = WalOpType::from_u8(data[8])
            .ok_or_else(|| BikoError::WalCorrupted {
                details: format!("unknown op type {}", data[8]),
            })?;
        let payload = data[9..].to_vec();

        Ok(Some(WalEntry {
            tx_id,
            op_type,
            payload,
        }))
    }
}

/// Write-Ahead Log.
///
/// Append-only log para durabilidad y crash recovery.
/// Cuando el archivo actual excede `MAX_WAL_FILE_SIZE`, rota a uno nuevo.
///
/// # Uso típico
/// ```ignore
/// let mut wal = WriteAheadLog::open("data/wal")?;
/// wal.append(WalEntry { tx_id: 1, op_type: WalOpType::PageWrite, payload: data })?;
/// wal.sync()?; // Flush a disco
/// ```
pub struct WriteAheadLog {
    dir: PathBuf,
    writer: BufWriter<File>,
    current_file_idx: u64,
    current_size: u64,
}

impl WriteAheadLog {
    /// Abre o crea un WAL en el directorio dado.
    ///
    /// Si ya existen archivos WAL, continúa desde el último.
    /// New WAL files get a magic/version header for format detection.
    pub fn open(dir: impl AsRef<Path>) -> BikoResult<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;

        let current_file_idx = Self::find_latest_wal_idx(&dir);
        let wal_path = Self::wal_file_path(&dir, current_file_idx);
        let is_new = !wal_path.exists() || fs::metadata(&wal_path).map(|m| m.len() == 0).unwrap_or(true);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)?;

        let current_size = file.metadata()?.len();

        let mut wal = Self {
            dir,
            writer: BufWriter::new(file),
            current_file_idx,
            current_size,
        };

        // Write magic header for new files
        if is_new {
            wal.write_header()?;
        }

        Ok(wal)
    }

    /// Writes the WAL file header (magic + version).
    fn write_header(&mut self) -> BikoResult<()> {
        self.writer.write_all(&WAL_MAGIC.to_le_bytes())?;
        self.writer.write_all(&WAL_VERSION.to_le_bytes())?;
        self.current_size += WAL_HEADER_SIZE as u64;
        Ok(())
    }

    /// Validates the WAL file header. Returns false if not a valid WAL file.
    fn validate_header(file: &mut File) -> BikoResult<bool> {
        let file_len = file.metadata()?.len();
        if file_len < WAL_HEADER_SIZE as u64 {
            // Legacy file without header — allow reading from start
            file.seek(SeekFrom::Start(0))?;
            return Ok(true); // tolerate headerless files
        }

        file.seek(SeekFrom::Start(0))?;
        let mut magic_buf = [0u8; 4];
        match file.read_exact(&mut magic_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(false),
            Err(e) => return Err(BikoError::Io(e)),
        }
        let magic = u32::from_le_bytes(magic_buf);

        if magic != WAL_MAGIC {
            // Legacy file — seek back to start and read as headerless
            file.seek(SeekFrom::Start(0))?;
            return Ok(true);
        }

        let mut ver_buf = [0u8; 4];
        file.read_exact(&mut ver_buf)?;
        let version = u32::from_le_bytes(ver_buf);
        if version > WAL_VERSION {
            return Err(BikoError::WalCorrupted {
                details: format!("WAL version {version} is newer than supported {WAL_VERSION}"),
            });
        }

        // File pointer is now past the header — ready to read entries
        Ok(true)
    }

    /// Agrega una entrada al WAL.
    pub fn append(&mut self, entry: WalEntry) -> BikoResult<()> {
        let bytes = entry.to_bytes();

        // Rotar si excedemos el tamaño máximo
        if self.current_size + bytes.len() as u64 > MAX_WAL_FILE_SIZE as u64 {
            self.rotate()?;
        }

        self.writer.write_all(&bytes)?;
        self.current_size += bytes.len() as u64;
        Ok(())
    }

    /// Flush del buffer + fsync a disco.
    pub fn sync(&mut self) -> BikoResult<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }

    /// Escribe un checkpoint marker.
    pub fn checkpoint(&mut self) -> BikoResult<()> {
        self.append(WalEntry {
            tx_id: 0,
            op_type: WalOpType::Checkpoint,
            payload: Vec::new(),
        })
    }

    /// Lee todas las entradas desde el último checkpoint para recovery.
    ///
    /// Resilience: if a WAL entry fails CRC validation, it and all subsequent
    /// entries in that file are skipped (tail corruption from crash mid-write).
    /// Valid entries from prior files are still returned.
    pub fn recover(&self) -> BikoResult<Vec<WalEntry>> {
        let mut all_entries = Vec::new();

        for idx in 0..=self.current_file_idx {
            let path = Self::wal_file_path(&self.dir, idx);
            if !path.exists() {
                continue;
            }

            let mut file = File::open(&path)?;

            // Validate header (tolerates legacy headerless files)
            if !Self::validate_header(&mut file)? {
                continue; // skip invalid files
            }

            // Read entries with skip-on-corruption
            loop {
                match WalEntry::read_from(&mut file) {
                    Ok(Some(entry)) => all_entries.push(entry),
                    Ok(None) => break, // clean EOF
                    Err(BikoError::WalCorrupted { .. }) => {
                        // Tail corruption (crash mid-write) — stop reading this file.
                        // Entries already read from this file are still valid.
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        // Buscar el último checkpoint y devolver todo lo que viene después
        let last_checkpoint = all_entries
            .iter()
            .rposition(|e| e.op_type == WalOpType::Checkpoint);

        match last_checkpoint {
            Some(pos) => Ok(all_entries[pos + 1..].to_vec()),
            None => Ok(all_entries),
        }
    }

    /// Elimina archivos WAL anteriores al actual (después de un checkpoint exitoso).
    pub fn purge_old(&self) -> BikoResult<()> {
        for idx in 0..self.current_file_idx {
            let path = Self::wal_file_path(&self.dir, idx);
            if path.exists() {
                fs::remove_file(&path)?;
            }
        }
        Ok(())
    }

    /// Rota al siguiente archivo WAL.
    fn rotate(&mut self) -> BikoResult<()> {
        self.writer.flush()?;

        self.current_file_idx += 1;
        let new_path = Self::wal_file_path(&self.dir, self.current_file_idx);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&new_path)?;

        self.writer = BufWriter::new(file);
        self.current_size = 0;
        self.write_header()?;
        Ok(())
    }

    pub(crate) fn wal_file_path(dir: &Path, idx: u64) -> PathBuf {
        dir.join(format!("wal_{idx:08}.log"))
    }

    fn find_latest_wal_idx(dir: &Path) -> u64 {
        let mut max_idx = 0u64;
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(rest) = name.strip_prefix("wal_") {
                        if let Some(num_str) = rest.strip_suffix(".log") {
                            if let Ok(idx) = num_str.parse::<u64>() {
                                max_idx = max_idx.max(idx);
                            }
                        }
                    }
                }
            }
        }
        max_idx
    }
}

// =============================================================================
// ConcurrentWal — Thread-safe WAL with group commit
// =============================================================================
// Wrapper sobre WriteAheadLog que permite acceso concurrente desde múltiples
// threads. Usa parking_lot::Mutex (más rápido que std::Mutex) y group commit
// para amortizar el costo de fsync sobre múltiples escrituras.
//
// Diseño:
// - Mutex protege el WAL subyacente
// - `append_batch()` escribe N entradas + fsync en una sola lock acquisition
// - `group_commit_threshold` contrола cuántas entradas se acumulan antes de fsync
// - tx_counter atómico para IDs de transacción
// =============================================================================

/// WAL concurrente con group commit para inserción masiva.
///
/// Thread-safe: múltiples threads pueden escribir concurrentemente.
/// Group commit: acumula escrituras antes de hacer fsync para amortizar I/O.
///
/// # Ejemplo
/// ```ignore
/// let wal = ConcurrentWal::open("data/wal", 1000)?;
/// wal.write(WalOpType::PageWrite, &payload)?;
/// wal.write_batch(&entries)?; // N entries + 1 fsync
/// ```
pub struct ConcurrentWal {
    inner: Mutex<WriteAheadLog>,
    tx_counter: AtomicU64,
    /// Número de entradas entre fsyncs automáticos.
    group_commit_threshold: u64,
    /// Conteo de entradas desde el último sync.
    pending_count: AtomicU64,
}

impl ConcurrentWal {
    /// Abre un ConcurrentWal. `group_commit_threshold`: entradas entre fsyncs.
    pub fn open(dir: impl AsRef<Path>, group_commit_threshold: u64) -> BikoResult<Self> {
        let wal = WriteAheadLog::open(dir)?;
        Ok(Self {
            inner: Mutex::new(wal),
            tx_counter: AtomicU64::new(1),
            group_commit_threshold,
            pending_count: AtomicU64::new(0),
        })
    }

    /// Genera un nuevo transaction ID atómicamente.
    pub fn next_tx_id(&self) -> u64 {
        self.tx_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Escribe una entrada al WAL. Hace fsync si se alcanza el threshold.
    pub fn write(&self, op_type: WalOpType, payload: &[u8]) -> BikoResult<u64> {
        let tx_id = self.next_tx_id();
        self.write_with_tx(tx_id, op_type, payload)?;
        Ok(tx_id)
    }

    /// Escribe una entrada al WAL con un tx_id específico.
    pub fn write_with_tx(&self, tx_id: u64, op_type: WalOpType, payload: &[u8]) -> BikoResult<()> {
        let entry = WalEntry {
            tx_id,
            op_type,
            payload: payload.to_vec(),
        };

        let mut wal = self.inner.lock();
        wal.append(entry)?;

        let pending = self.pending_count.fetch_add(1, Ordering::Relaxed) + 1;
        if pending >= self.group_commit_threshold {
            wal.sync()?;
            self.pending_count.store(0, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Escribe un batch de entradas al WAL con un solo lock + un solo fsync.
    ///
    /// Esto es la operación clave para inserción masiva:
    /// - 1 lock acquisition para todo el batch
    /// - N appends secuenciales (in-memory BufWriter, sin I/O por entrada)
    /// - 1 fsync al final del batch
    ///
    /// Throughput: amortiza el costo de fsync (~1ms) sobre N entradas.
    pub fn write_batch(&self, entries: &[WalEntry]) -> BikoResult<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let mut wal = self.inner.lock();
        for entry in entries {
            wal.append(entry.clone())?;
        }
        wal.sync()?;
        self.pending_count.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Fuerza fsync de las entradas pendientes.
    pub fn flush(&self) -> BikoResult<()> {
        let mut wal = self.inner.lock();
        wal.sync()?;
        self.pending_count.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Checkpoint: marca de consistencia en el WAL.
    pub fn checkpoint(&self) -> BikoResult<()> {
        let mut wal = self.inner.lock();
        wal.checkpoint()?;
        wal.sync()?;
        self.pending_count.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Recover: lee entradas desde el último checkpoint.
    pub fn recover(&self) -> BikoResult<Vec<WalEntry>> {
        let wal = self.inner.lock();
        wal.recover()
    }

    /// Purga archivos WAL antiguos.
    pub fn purge_old(&self) -> BikoResult<()> {
        let wal = self.inner.lock();
        wal.purge_old()
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn temp_wal_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("bikodb_test_wal").join(name);
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn test_wal_append_and_recover() {
        let dir = temp_wal_dir("basic");
        let mut wal = WriteAheadLog::open(&dir).unwrap();

        wal.append(WalEntry {
            tx_id: 1,
            op_type: WalOpType::PageWrite,
            payload: vec![0xAB; 100],
        })
        .unwrap();

        wal.append(WalEntry {
            tx_id: 1,
            op_type: WalOpType::TxCommit,
            payload: vec![],
        })
        .unwrap();

        wal.sync().unwrap();

        let entries = wal.recover().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].tx_id, 1);
        assert_eq!(entries[0].op_type, WalOpType::PageWrite);
        assert_eq!(entries[1].op_type, WalOpType::TxCommit);
    }

    #[test]
    fn test_wal_checkpoint_recovery() {
        let dir = temp_wal_dir("checkpoint");
        let mut wal = WriteAheadLog::open(&dir).unwrap();

        // Datos antes del checkpoint (deberían ignorarse en recovery)
        wal.append(WalEntry {
            tx_id: 1,
            op_type: WalOpType::PageWrite,
            payload: vec![1],
        })
        .unwrap();

        wal.checkpoint().unwrap();

        // Datos después del checkpoint (estos sí se recuperan)
        wal.append(WalEntry {
            tx_id: 2,
            op_type: WalOpType::PageWrite,
            payload: vec![2],
        })
        .unwrap();

        wal.sync().unwrap();

        let entries = wal.recover().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].tx_id, 2);
    }

    #[test]
    fn test_wal_entry_serialization() {
        let entry = WalEntry {
            tx_id: 42,
            op_type: WalOpType::PageWrite,
            payload: b"hello".to_vec(),
        };

        let bytes = entry.to_bytes();
        let mut cursor = io::Cursor::new(bytes);
        let recovered = WalEntry::read_from(&mut cursor).unwrap().unwrap();

        assert_eq!(recovered.tx_id, 42);
        assert_eq!(recovered.op_type, WalOpType::PageWrite);
        assert_eq!(recovered.payload, b"hello");
    }

    #[test]
    fn test_wal_reopen() {
        let dir = temp_wal_dir("reopen");

        {
            let mut wal = WriteAheadLog::open(&dir).unwrap();
            wal.append(WalEntry {
                tx_id: 1,
                op_type: WalOpType::TxCommit,
                payload: vec![],
            })
            .unwrap();
            wal.sync().unwrap();
        }

        // Re-open y verificar que los datos persisten
        let wal2 = WriteAheadLog::open(&dir).unwrap();
        let entries = wal2.recover().unwrap();
        assert_eq!(entries.len(), 1);
    }

    // ── ConcurrentWal tests ─────────────────────────────────────────────

    #[test]
    fn test_concurrent_wal_write() {
        let dir = temp_wal_dir("concurrent_basic");
        let wal = ConcurrentWal::open(&dir, 100).unwrap();

        let tx1 = wal.write(WalOpType::PageWrite, b"data1").unwrap();
        let tx2 = wal.write(WalOpType::PageWrite, b"data2").unwrap();
        assert_ne!(tx1, tx2);

        wal.flush().unwrap();

        let entries = wal.recover().unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_concurrent_wal_write_batch() {
        let dir = temp_wal_dir("concurrent_batch");
        let wal = ConcurrentWal::open(&dir, 1000).unwrap();

        let entries: Vec<WalEntry> = (0..100)
            .map(|i| WalEntry {
                tx_id: i,
                op_type: WalOpType::PageWrite,
                payload: vec![i as u8; 10],
            })
            .collect();

        wal.write_batch(&entries).unwrap();

        let recovered = wal.recover().unwrap();
        assert_eq!(recovered.len(), 100);
    }

    #[test]
    fn test_concurrent_wal_group_commit() {
        let dir = temp_wal_dir("concurrent_group");
        // Threshold = 5: auto-sync every 5 writes
        let wal = ConcurrentWal::open(&dir, 5).unwrap();

        for _ in 0..10 {
            wal.write(WalOpType::PageWrite, b"x").unwrap();
        }
        // After 10 writes with threshold 5, should have auto-synced twice
        // Verify by recovering
        let entries = wal.recover().unwrap();
        assert_eq!(entries.len(), 10);
    }

    #[test]
    fn test_concurrent_wal_checkpoint() {
        let dir = temp_wal_dir("concurrent_checkpoint");
        let wal = ConcurrentWal::open(&dir, 1000).unwrap();

        wal.write(WalOpType::PageWrite, b"before").unwrap();
        wal.checkpoint().unwrap();
        wal.write(WalOpType::PageWrite, b"after").unwrap();
        wal.flush().unwrap();

        let entries = wal.recover().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].payload, b"after");
    }

    #[test]
    fn test_concurrent_wal_multithread() {
        let dir = temp_wal_dir("concurrent_mt");
        let wal = std::sync::Arc::new(ConcurrentWal::open(&dir, 100).unwrap());

        std::thread::scope(|s| {
            for thread_id in 0..4u8 {
                let w = std::sync::Arc::clone(&wal);
                s.spawn(move || {
                    for i in 0..25u8 {
                        let payload = vec![thread_id, i];
                        w.write(WalOpType::PageWrite, &payload).unwrap();
                    }
                });
            }
        });

        wal.flush().unwrap();
        let entries = wal.recover().unwrap();
        assert_eq!(entries.len(), 100);
    }

    // ── CRC integrity ───────────────────────────────────────────────────

    #[test]
    fn test_wal_entry_crc_integrity() {
        let entry = WalEntry {
            tx_id: 99,
            op_type: WalOpType::RecordInsert,
            payload: b"record_data".to_vec(),
        };
        let bytes = entry.to_bytes();

        // Verify the entry includes a CRC (bytes[4..8] is the CRC field)
        let stored_crc = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_ne!(stored_crc, 0); // CRC should not be zero for non-trivial data

        // Tamper with one byte in the payload area and verify CRC catches it
        let mut tampered = bytes.clone();
        tampered[12] ^= 0xFF; // Flip a byte in the tx_id/op area
        let mut cursor = io::Cursor::new(tampered);
        let result = WalEntry::read_from(&mut cursor);
        assert!(result.is_err() || result.unwrap().is_none());
    }

    #[test]
    fn test_wal_corruption_skip_and_continue() {
        let dir = temp_wal_dir("corruption_skip");
        let mut wal = WriteAheadLog::open(&dir).unwrap();

        // Write 2 valid entries
        wal.append(WalEntry {
            tx_id: 1,
            op_type: WalOpType::PageWrite,
            payload: vec![0xAA; 10],
        }).unwrap();
        wal.append(WalEntry {
            tx_id: 2,
            op_type: WalOpType::TxCommit,
            payload: vec![],
        }).unwrap();
        wal.sync().unwrap();

        // Append garbage to simulate crash mid-write
        let wal_path = WriteAheadLog::wal_file_path(&dir, 0);
        {
            let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
            f.write_all(&50u32.to_le_bytes()).unwrap(); // len
            f.write_all(&0xBADC_0000u32.to_le_bytes()).unwrap(); // bad CRC
            f.write_all(&[0xFF; 50]).unwrap(); // garbage
            f.sync_all().unwrap();
        }

        // Recovery should return the 2 valid entries, skipping the corrupt tail
        let wal2 = WriteAheadLog::open(&dir).unwrap();
        let entries = wal2.recover().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].tx_id, 1);
        assert_eq!(entries[1].tx_id, 2);
    }

    #[test]
    fn test_wal_magic_header() {
        let dir = temp_wal_dir("magic_header");
        {
            let mut wal = WriteAheadLog::open(&dir).unwrap();
            wal.append(WalEntry {
                tx_id: 1,
                op_type: WalOpType::PageWrite,
                payload: vec![1],
            }).unwrap();
            wal.sync().unwrap();
        }

        // Verify the file starts with magic bytes
        let wal_path = WriteAheadLog::wal_file_path(&dir, 0);
        let file_bytes = fs::read(&wal_path).unwrap();
        let magic = u32::from_le_bytes(file_bytes[0..4].try_into().unwrap());
        let version = u32::from_le_bytes(file_bytes[4..8].try_into().unwrap());
        assert_eq!(magic, WAL_MAGIC);
        assert_eq!(version, WAL_VERSION);

        // And recovery still works
        let wal2 = WriteAheadLog::open(&dir).unwrap();
        let entries = wal2.recover().unwrap();
        assert_eq!(entries.len(), 1);
    }
}
