// =============================================================================
// bikodb-query — Query Layer
// =============================================================================
// Parsers para SQL (graph dialect), Cypher y Gremlin que producen LogicalOp.
//
//   sql     → Parser SQL inspirado en ArcadeDB SQL (SELECT, INSERT, DELETE, COUNT, GROUP BY)
//   cypher  → Parser Cypher (MATCH, CREATE, DELETE, SET, RETURN, WHERE, ORDER BY)
//   gremlin → Parser Gremlin bytecode interpreter (g.V().has().out()...)
//
// ## Diseño
// Parser recursivo descendente escrito a mano (sin dependencia de PEG/nom).
// Esto permite mensajes de error claros y control total del AST.
// =============================================================================

pub mod cypher;
pub mod gremlin;
pub mod sql;
