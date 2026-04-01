// =============================================================================
// bikodb-core::config — Constantes y configuración global
// =============================================================================
// Valores por defecto del motor. Pueden ser overridden en runtime.
// Centralizados aquí para facilitar tuning y benchmarks.
// =============================================================================

/// Tamaño de página por defecto: 64KB (como ArcadeDB 2^16).
/// Páginas grandes mejoran throughput secuencial; páginas pequeñas
/// mejoran random access. 64KB es un buen compromiso.
pub const DEFAULT_PAGE_SIZE: usize = 64 * 1024; // 64 KB

/// Tamaño de página para índices: 256KB (como LSM-Tree de ArcadeDB).
/// Índices se leen secuencialmente con más frecuencia, así que
/// páginas más grandes reducen overhead de I/O.
pub const INDEX_PAGE_SIZE: usize = 256 * 1024; // 256 KB

/// Tamaño máximo de un WAL file antes de rotar: 64MB.
pub const MAX_WAL_FILE_SIZE: usize = 64 * 1024 * 1024; // 64 MB

/// Número máximo de páginas en cache antes de eviction.
/// Con 64KB por página, 16K páginas = ~1 GB de cache.
pub const DEFAULT_PAGE_CACHE_SIZE: usize = 16_384;

/// Número de buckets por defecto para un nuevo tipo.
/// Más buckets = mejor paralelismo de escritura (un bucket por thread).
pub const DEFAULT_BUCKETS_PER_TYPE: u32 = 8;

/// Número de threads del async executor por defecto.
/// Se puede ajustar según el hardware.
pub const DEFAULT_ASYNC_THREADS: usize = 4;

/// Máximo de transacciones concurrentes antes de backpressure.
pub const MAX_CONCURRENT_TRANSACTIONS: usize = 1024;

/// Dimensiones por defecto para embeddings de IA.
/// 384 es el tamaño de sentence-transformers/all-MiniLM-L6-v2.
pub const DEFAULT_EMBEDDING_DIMENSIONS: usize = 384;

/// Tamaño del buffer para SmallVec de edges por nodo.
/// La mayoría de nodos en grafos power-law tienen ≤ 4 edges,
/// así que inline 4 en stack (32 bytes) para minimizar RAM por nodo.
pub const INLINE_EDGES_PER_NODE: usize = 4;

/// Tamaño del buffer para SmallVec de propiedades por record.
/// La mayoría de records tienen ≤ 4 propiedades.
pub const INLINE_PROPERTIES_PER_RECORD: usize = 4;

/// Configuración runtime del motor (overrides de las constantes).
///
/// Se crea con defaults y se modifica antes de abrir la base de datos.
/// Una vez abierta, la configuración es inmutable.
#[derive(Debug, Clone)]
pub struct OxiConfig {
    /// Tamaño de página de datos en bytes
    pub page_size: usize,
    /// Tamaño de página de índices en bytes
    pub index_page_size: usize,
    /// Número máximo de páginas en cache
    pub page_cache_size: usize,
    /// Tamaño máximo de WAL file en bytes
    pub max_wal_file_size: usize,
    /// Número de buckets por defecto para tipos nuevos
    pub buckets_per_type: u32,
    /// ¿Usar WAL? (desactivar para benchmarks)
    pub use_wal: bool,
    /// ¿Sync WAL a disco en cada commit?
    pub wal_sync: bool,
    /// Directorio de datos
    pub data_dir: String,
}

impl Default for OxiConfig {
    fn default() -> Self {
        Self {
            page_size: DEFAULT_PAGE_SIZE,
            index_page_size: INDEX_PAGE_SIZE,
            page_cache_size: DEFAULT_PAGE_CACHE_SIZE,
            max_wal_file_size: MAX_WAL_FILE_SIZE,
            buckets_per_type: DEFAULT_BUCKETS_PER_TYPE,
            use_wal: true,
            wal_sync: true,
            data_dir: "./data".to_string(),
        }
    }
}
