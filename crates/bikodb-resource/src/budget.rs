// =============================================================================
// bikodb-resource::budget — Distribución de memoria entre componentes
// =============================================================================

/// Presupuesto de memoria del motor.
///
/// Distribuye la RAM disponible entre los distintos componentes:
/// page cache, WAL buffers, vector index cache, etc.
///
/// # Ejemplo
/// ```
/// use bikodb_resource::budget::MemoryBudget;
///
/// // 1 GB total
/// let budget = MemoryBudget::new(1024 * 1024 * 1024);
/// assert!(budget.page_cache_bytes > 0);
/// assert!(budget.wal_buffer_bytes > 0);
/// ```
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Total de memoria asignada al motor
    pub total_bytes: usize,
    /// Memoria para el page cache (50% por defecto)
    pub page_cache_bytes: usize,
    /// Memoria para el grafo in-memory (15%)
    pub graph_bytes: usize,
    /// Memoria para buffer WAL (10%)
    pub wal_buffer_bytes: usize,
    /// Memoria para vector index (10%)
    pub vector_index_bytes: usize,
    /// Memoria para query execution (10%)
    pub query_execution_bytes: usize,
    /// Reserva general (5%)
    pub reserved_bytes: usize,
}

impl MemoryBudget {
    /// Crea un presupuesto con distribución por defecto.
    pub fn new(total_bytes: usize) -> Self {
        Self {
            total_bytes,
            page_cache_bytes: total_bytes * 50 / 100,
            graph_bytes: total_bytes * 15 / 100,
            wal_buffer_bytes: total_bytes * 10 / 100,
            vector_index_bytes: total_bytes * 10 / 100,
            query_execution_bytes: total_bytes * 10 / 100,
            reserved_bytes: total_bytes * 5 / 100,
        }
    }

    /// Crea un presupuesto con distribución custom.
    pub fn custom(
        total_bytes: usize,
        page_cache_pct: u8,
        graph_pct: u8,
        wal_pct: u8,
        vector_pct: u8,
        query_pct: u8,
    ) -> Self {
        let reserved_pct = 100u8
            .saturating_sub(page_cache_pct)
            .saturating_sub(graph_pct)
            .saturating_sub(wal_pct)
            .saturating_sub(vector_pct)
            .saturating_sub(query_pct);

        Self {
            total_bytes,
            page_cache_bytes: total_bytes * page_cache_pct as usize / 100,
            graph_bytes: total_bytes * graph_pct as usize / 100,
            wal_buffer_bytes: total_bytes * wal_pct as usize / 100,
            vector_index_bytes: total_bytes * vector_pct as usize / 100,
            query_execution_bytes: total_bytes * query_pct as usize / 100,
            reserved_bytes: total_bytes * reserved_pct as usize / 100,
        }
    }

    /// Número máximo de páginas en cache dados el presupuesto y page_size.
    pub fn max_cached_pages(&self, page_size: usize) -> usize {
        self.page_cache_bytes / page_size
    }

    /// ¿El consumo de memoria del grafo excede su presupuesto?
    pub fn graph_over_budget(&self, current_graph_bytes: usize) -> bool {
        current_graph_bytes > self.graph_bytes
    }

    /// ¿El page cache excede su presupuesto?
    ///
    /// Returns `Some(target)` with the target page count to evict to
    /// if over budget, or `None` if within budget.
    pub fn page_cache_target(&self, cached_pages: usize, page_size: usize) -> Option<usize> {
        let max = self.max_cached_pages(page_size);
        if cached_pages > max {
            Some(max)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_budget() {
        let budget = MemoryBudget::new(1_000_000);
        assert_eq!(budget.page_cache_bytes, 500_000);
        assert_eq!(budget.graph_bytes, 150_000);
        assert_eq!(budget.wal_buffer_bytes, 100_000);
        assert_eq!(budget.vector_index_bytes, 100_000);
        assert_eq!(budget.query_execution_bytes, 100_000);
        assert_eq!(budget.reserved_bytes, 50_000);
    }

    #[test]
    fn test_max_cached_pages() {
        let budget = MemoryBudget::new(1024 * 1024 * 64); // 64MB
        let max_pages = budget.max_cached_pages(65536); // 64KB pages
        assert_eq!(max_pages, 512); // 50% of 64MB / 64KB
    }

    #[test]
    fn test_custom_budget() {
        let budget = MemoryBudget::custom(1_000_000, 70, 10, 5, 5, 5);
        assert_eq!(budget.page_cache_bytes, 700_000);
        assert_eq!(budget.graph_bytes, 100_000);
        assert_eq!(budget.reserved_bytes, 50_000);
    }

    #[test]
    fn test_graph_over_budget() {
        let budget = MemoryBudget::new(1_000_000);
        assert!(!budget.graph_over_budget(100_000));
        assert!(budget.graph_over_budget(200_000));
    }
}
