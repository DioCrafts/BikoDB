// =============================================================================
// bikodb-resource::thread_pool — Configurable Rayon thread pool
// =============================================================================
// Provides explicit control over the global Rayon thread pool used by
// parallel graph algorithms (BFS, SSSP, PageRank, LPA, etc.).
//
// By default, Rayon uses `num_cpus` threads. This module allows:
// - Setting a specific thread count (for resource-constrained environments)
// - Naming threads for debugging (e.g., `bikodb-worker-0`)
// - Stack size configuration (for deep recursion in DFS)
//
// MUST be called early (before any rayon parallel work).
// Calling after rayon is already initialized returns an error.
// =============================================================================

use std::sync::atomic::{AtomicBool, Ordering};

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Configuration for the BikoDB worker thread pool.
///
/// # Example
/// ```
/// use bikodb_resource::thread_pool::ThreadPoolConfig;
///
/// // Use 4 threads with custom stack size
/// let config = ThreadPoolConfig::new()
///     .num_threads(4)
///     .stack_size(4 * 1024 * 1024);
///
/// // In production, call config.init() once at startup.
/// ```
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of worker threads. `None` = Rayon default (num_cpus).
    threads: Option<usize>,
    /// Stack size per thread in bytes. `None` = Rayon default (8MB).
    stack: Option<usize>,
    /// Thread name prefix.
    prefix: String,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreadPoolConfig {
    /// Creates a config with defaults (num_cpus threads, 8MB stack, "bikodb-worker" prefix).
    pub fn new() -> Self {
        Self {
            threads: None,
            stack: None,
            prefix: "bikodb-worker".to_string(),
        }
    }

    /// Sets the number of worker threads.
    pub fn num_threads(mut self, n: usize) -> Self {
        self.threads = Some(n.max(1));
        self
    }

    /// Sets the stack size per thread (bytes).
    pub fn stack_size(mut self, bytes: usize) -> Self {
        self.stack = Some(bytes);
        self
    }

    /// Sets the thread name prefix.
    pub fn thread_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Returns the configured thread count (or system default).
    pub fn get_num_threads(&self) -> usize {
        self.threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
    }

    /// Initializes the global Rayon thread pool with this configuration.
    ///
    /// Must be called once, before any parallel work. Returns `Err` if
    /// the pool was already initialized (by a previous call or by Rayon
    /// auto-initialization).
    pub fn init(self) -> Result<(), ThreadPoolInitError> {
        if INITIALIZED.swap(true, Ordering::SeqCst) {
            return Err(ThreadPoolInitError::AlreadyInitialized);
        }

        let mut builder = rayon::ThreadPoolBuilder::new();

        if let Some(n) = self.threads {
            builder = builder.num_threads(n);
        }
        if let Some(s) = self.stack {
            builder = builder.stack_size(s);
        }

        let prefix = self.prefix;
        builder = builder.thread_name(move |idx| format!("{prefix}-{idx}"));

        builder
            .build_global()
            .map_err(|e| ThreadPoolInitError::RayonError(e.to_string()))
    }
}

/// Error initializing the thread pool.
#[derive(Debug, Clone)]
pub enum ThreadPoolInitError {
    /// The global pool was already initialized.
    AlreadyInitialized,
    /// Rayon returned an error during build.
    RayonError(String),
}

impl std::fmt::Display for ThreadPoolInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyInitialized => write!(f, "thread pool already initialized"),
            Self::RayonError(e) => write!(f, "rayon thread pool error: {e}"),
        }
    }
}

impl std::error::Error for ThreadPoolInitError {}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let cfg = ThreadPoolConfig::new();
        assert!(cfg.threads.is_none());
        assert!(cfg.stack.is_none());
        assert_eq!(cfg.prefix, "bikodb-worker");
        assert!(cfg.get_num_threads() >= 1);
    }

    #[test]
    fn test_config_builder() {
        let cfg = ThreadPoolConfig::new()
            .num_threads(8)
            .stack_size(2 * 1024 * 1024)
            .thread_prefix("test-worker");

        assert_eq!(cfg.threads, Some(8));
        assert_eq!(cfg.stack, Some(2 * 1024 * 1024));
        assert_eq!(cfg.prefix, "test-worker");
        assert_eq!(cfg.get_num_threads(), 8);
    }

    #[test]
    fn test_num_threads_min_one() {
        let cfg = ThreadPoolConfig::new().num_threads(0);
        assert_eq!(cfg.threads, Some(1));
    }
}
