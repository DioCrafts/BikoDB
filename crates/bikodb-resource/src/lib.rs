// =============================================================================
// bikodb-resource — Resource Manager
// =============================================================================
// Monitoreo y gestión adaptativa de recursos del sistema.
//
//   monitor    → CPU, RAM, disk usage tracking
//   budget     → Distribución de memoria entre componentes (cache, WAL, etc.)
//
// ## Diseño
// - Polling periódico de métricas del sistema
// - Ajuste automático del page cache según RAM disponible
// - Alertas cuando se exceden umbrales configurables
//
// ## Inspiración
// - ArcadeDB: PerformanceStats, profiler, resource reporting
// - Neo4j: JVM memory tracking, adaptive cache
// =============================================================================

pub mod budget;
pub mod monitor;
pub mod thread_pool;
