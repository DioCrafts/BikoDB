// =============================================================================
// bikodb-resource::monitor — Métricas del sistema
// =============================================================================

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Snapshot de métricas del sistema en un momento dado.
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    /// RAM usada por el proceso (bytes, estimada)
    pub memory_used_bytes: u64,
    /// Número de páginas en cache
    pub cached_pages: u64,
    /// Número de queries ejecutadas desde el inicio
    pub queries_executed: u64,
    /// Nodes totales en el grafo
    pub total_nodes: u64,
    /// Edges totales en el grafo
    pub total_edges: u64,
    /// Uptime en segundos
    pub uptime_secs: f64,
}

/// Monitor de recursos del motor.
///
/// Registra métricas clave para observabilidad y decisiones adaptativas.
///
/// # Ejemplo
/// ```
/// use bikodb_resource::monitor::ResourceMonitor;
///
/// let monitor = ResourceMonitor::new();
/// monitor.record_query();
/// monitor.record_query();
///
/// let metrics = monitor.snapshot(0, 0, 0);
/// assert_eq!(metrics.queries_executed, 2);
/// ```
pub struct ResourceMonitor {
    start_time: Instant,
    queries_executed: AtomicU64,
    bytes_written: AtomicU64,
    bytes_read: AtomicU64,
    /// Tracked graph memory in bytes (updated externally by graph engine).
    graph_memory_bytes: AtomicU64,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            queries_executed: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            graph_memory_bytes: AtomicU64::new(0),
        }
    }

    /// Registra una query ejecutada.
    pub fn record_query(&self) {
        self.queries_executed.fetch_add(1, Ordering::Relaxed);
    }

    /// Registra bytes escritos a disco.
    pub fn record_write(&self, bytes: u64) {
        self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Registra bytes leídos de disco.
    pub fn record_read(&self, bytes: u64) {
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Actualiza la memoria usada por el grafo (llamado por el graph engine).
    pub fn update_graph_memory(&self, bytes: u64) {
        self.graph_memory_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Memoria actual del grafo en bytes.
    pub fn graph_memory_bytes(&self) -> u64 {
        self.graph_memory_bytes.load(Ordering::Relaxed)
    }

    /// Genera un snapshot de métricas.
    ///
    /// Los contadores de nodos/edges se pasan externamente (del grafo).
    /// `graph_memory_bytes` se usa como estimación de RAM si está disponible.
    pub fn snapshot(&self, total_nodes: u64, total_edges: u64, cached_pages: u64) -> SystemMetrics {
        SystemMetrics {
            memory_used_bytes: self.graph_memory_bytes.load(Ordering::Relaxed),
            cached_pages,
            queries_executed: self.queries_executed.load(Ordering::Relaxed),
            total_nodes,
            total_edges,
            uptime_secs: self.start_time.elapsed().as_secs_f64(),
        }
    }

    /// Bytes totales escritos.
    pub fn total_bytes_written(&self) -> u64 {
        self.bytes_written.load(Ordering::Relaxed)
    }

    /// Bytes totales leídos.
    pub fn total_bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_basic() {
        let m = ResourceMonitor::new();
        m.record_query();
        m.record_query();
        m.record_write(4096);
        m.record_read(8192);

        let snap = m.snapshot(100, 200, 0);
        assert_eq!(snap.queries_executed, 2);
        assert_eq!(snap.total_nodes, 100);
        assert_eq!(m.total_bytes_written(), 4096);
        assert_eq!(m.total_bytes_read(), 8192);
        assert!(snap.uptime_secs >= 0.0);
    }

    #[test]
    fn test_monitor_graph_memory() {
        let m = ResourceMonitor::new();
        assert_eq!(m.graph_memory_bytes(), 0);

        m.update_graph_memory(1_000_000);
        assert_eq!(m.graph_memory_bytes(), 1_000_000);

        let snap = m.snapshot(50, 100, 10);
        assert_eq!(snap.memory_used_bytes, 1_000_000);
        assert_eq!(snap.cached_pages, 10);
    }
}
