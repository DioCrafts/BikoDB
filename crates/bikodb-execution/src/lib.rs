// =============================================================================
// bikodb-execution — Execution Engine
// =============================================================================
// Ejecuta planes lógicos de query sobre el grafo.
//
//   plan       → Plan lógico (árbol de operadores lógicos)
//   operator   → Operadores físicos (scan, filter, expand, project, limit)
//   pipeline   → Pipeline de ejecución: conecta operadores en flujo pull-based
//
// ## Modelo de ejecución
// Pull-based (Volcano): cada operador implementa `next() → Option<Row>`.
// Ventajas: lazy evaluation, backpressure natural, bajo uso de RAM.
//
// ## Inspiración
// - ArcadeDB: SQLEngine, command executors
// - Neo4j: Morsel-driven pipeline (futuro), slot-based row
// =============================================================================

pub mod cost;
pub mod operator;
pub mod optimizer;
pub mod pipeline;
pub mod plan;
pub mod plan_cache;
pub mod tracker;
