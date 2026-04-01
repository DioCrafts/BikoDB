// =============================================================================
// bikodb-storage — Storage Engine
// =============================================================================
// Gestiona la persistencia de datos en disco con las siguientes capas:
//
//   page      → Páginas de tamaño fijo (64KB), unit de I/O
//   page_cache→ Cache de páginas en RAM (LRU eviction, lock-free reads)
//   bucket    → Colección de registros (nodos, edges) almacenados en páginas
//   wal       → Write-Ahead Log para durabilidad ACID (delta storage)
//   mmap      → Memory-mapped files para lecturas zero-copy
//   codec     → Serialización binaria + compresión LZ4
//   dictionary→ Mapeo string ↔ u16 para propiedades (compactación)
//
// ## Inspiración
// - ArcadeDB: PageManager, BasePage (65KB), WALFile (delta, 64MB max),
//   Bucket, Dictionary, FileManager, TransactionManager
// - Neo4j: Store files, property chains, relationship chains
//
// ## Diseño clave
// 1. Páginas son la unidad mínima de I/O — todo se lee/escribe en páginas
// 2. WAL almacena solo deltas (no páginas completas) para eficiencia
// 3. PageCache usa DashMap (lock-free concurrent HashMap) para reads
// 4. Compresión LZ4 en páginas antes de escribir a disco
// =============================================================================

pub mod bucket;
pub mod codec;
pub mod delta;
pub mod dictionary;
pub mod mmap;
pub mod page;
pub mod page_cache;
pub mod page_manager;
pub mod record_serializer;
pub mod storage_engine;
pub mod wal;

// Re-exports
pub use bucket::Bucket;
pub use dictionary::Dictionary;
pub use page::{Page, PageHeader};
pub use page_cache::PageCache;
pub use page_manager::PageManager;
pub use storage_engine::{DurabilityMode, SnapshotInfo, StorageEngine};
pub use wal::{ConcurrentWal, WalEntry, WalOpType, WriteAheadLog};
