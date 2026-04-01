// =============================================================================
// bikodb-server — Server Layer (Stubs v0.1)
// =============================================================================
// Punto de entrada para clientes. En v0.2 integrará HTTP/REST (axum),
// protocolo Bolt (Neo4j compatible), y WebSocket para subscriptions.
//
//   database   → Fachada principal que coordina todos los componentes
//
// ## v0.1
// Clase Database como API in-process (embedded mode).
// Sin networking — los clientes usan la API Rust directamente.
//
// ## v0.2
// - HTTP/REST con axum
// - Bolt wire protocol para compatibilidad con Neo4j drivers
// - WebSocket para graph change subscriptions
// =============================================================================

pub mod database;
pub mod http_api;
pub mod plugin_manager;

pub use database::{Database, LiveQueryCallback, SemanticSearchResult};
pub use http_api::build_router;
pub use plugin_manager::PluginManager;
