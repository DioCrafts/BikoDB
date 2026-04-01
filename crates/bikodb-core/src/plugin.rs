// =============================================================================
// bikodb-core::plugin — Sistema de plugins y extensibilidad
// =============================================================================
// Define las interfaces (traits) que los plugins deben implementar para
// extender BikoDB sin recompilar el núcleo.
//
// ## Componentes
//   Plugin            → Trait principal con lifecycle (init/shutdown)
//   HookPoint         → Puntos de intercepción pre/post operación
//   HookHandler       → Trait para interceptores con capacidad de veto
//   UserDefinedFn     → Trait para funciones custom en queries (UDFs)
//   GraphAlgorithmExt → Trait para algoritmos de grafos custom
//   DistanceFnExt     → Trait para métricas de distancia custom
//   InferenceProvider → Trait para proveedores de inferencia custom
//
// ## Diseño
// Todo es trait-based: para añadir funcionalidad, se implementa un trait
// y se registra en el PluginManager (en bikodb-server). No requiere
// recompilar el core engine.
// =============================================================================

use crate::error::BikoResult;
use crate::types::{EdgeId, NodeId, TypeId};
use crate::value::Value;
use std::any::Any;
use std::collections::HashMap;
use std::fmt;

// ── Plugin Trait ────────────────────────────────────────────────────────────

/// Trait principal para plugins de BikoDB.
///
/// Cada plugin tiene un nombre, versión, y lifecycle gestionado por el
/// PluginManager. Puede registrar hooks, UDFs, algoritmos, etc.
///
/// # Ejemplo
/// ```
/// use bikodb_core::plugin::{Plugin, PluginContext};
/// use bikodb_core::error::BikoResult;
///
/// struct MyPlugin;
///
/// impl Plugin for MyPlugin {
///     fn name(&self) -> &str { "my-plugin" }
///     fn version(&self) -> &str { "1.0.0" }
///     fn init(&self, _ctx: &dyn PluginContext) -> BikoResult<()> {
///         println!("Plugin initialized!");
///         Ok(())
///     }
///     fn shutdown(&self) -> BikoResult<()> {
///         println!("Plugin shutdown!");
///         Ok(())
///     }
///     fn as_any(&self) -> &dyn std::any::Any { self }
/// }
/// ```
pub trait Plugin: Send + Sync {
    /// Nombre único del plugin.
    fn name(&self) -> &str;
    /// Versión semántica del plugin.
    fn version(&self) -> &str;
    /// Descripción breve.
    fn description(&self) -> &str { "" }
    /// Inicialización del plugin. Se llama al registrarlo.
    fn init(&self, ctx: &dyn PluginContext) -> BikoResult<()>;
    /// Shutdown limpio. Se llama al desregistrarlo.
    fn shutdown(&self) -> BikoResult<()>;
    /// Cast a Any para downcasting (acceso a estado interno).
    fn as_any(&self) -> &dyn Any;
}

impl fmt::Debug for dyn Plugin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Plugin({} v{})", self.name(), self.version())
    }
}

/// Contexto proporcionado al plugin durante init.
///
/// Permite al plugin consultar el estado de la base de datos.
pub trait PluginContext: Send + Sync {
    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;
    fn registered_types(&self) -> Vec<String>;
}

/// Metadata de un plugin registrado.
#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub active: bool,
}

// ── Hook System ─────────────────────────────────────────────────────────────

/// Puntos en el pipeline donde los hooks pueden interceptar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookPoint {
    /// Antes de insertar un nodo.
    PreInsertNode,
    /// Después de insertar un nodo.
    PostInsertNode,
    /// Antes de insertar un edge.
    PreInsertEdge,
    /// Después de insertar un edge.
    PostInsertEdge,
    /// Antes de eliminar un nodo.
    PreDeleteNode,
    /// Después de eliminar un nodo.
    PostDeleteNode,
    /// Antes de ejecutar un query.
    PreQuery,
    /// Después de ejecutar un query.
    PostQuery,
    /// Antes de modificar una propiedad.
    PreSetProperty,
    /// Después de modificar una propiedad.
    PostSetProperty,
}

/// Contexto pasado a un hook handler.
#[derive(Debug, Clone)]
pub struct HookContext {
    pub hook_point: HookPoint,
    pub node_id: Option<NodeId>,
    pub edge_id: Option<EdgeId>,
    pub type_id: Option<TypeId>,
    pub properties: Vec<(u16, Value)>,
    pub query_text: Option<String>,
    /// Datos custom del hook (extensible).
    pub metadata: HashMap<String, Value>,
}

impl HookContext {
    pub fn for_node(hook_point: HookPoint, node_id: NodeId, type_id: TypeId) -> Self {
        Self {
            hook_point,
            node_id: Some(node_id),
            edge_id: None,
            type_id: Some(type_id),
            properties: Vec::new(),
            query_text: None,
            metadata: HashMap::new(),
        }
    }

    pub fn for_edge(hook_point: HookPoint, edge_id: EdgeId, type_id: TypeId) -> Self {
        Self {
            hook_point,
            node_id: None,
            edge_id: Some(edge_id),
            type_id: Some(type_id),
            properties: Vec::new(),
            query_text: None,
            metadata: HashMap::new(),
        }
    }

    pub fn for_query(hook_point: HookPoint, query: &str) -> Self {
        Self {
            hook_point,
            node_id: None,
            edge_id: None,
            type_id: None,
            properties: Vec::new(),
            query_text: Some(query.to_string()),
            metadata: HashMap::new(),
        }
    }
}

/// Resultado de un hook: continuar o abortar la operación.
#[derive(Debug, Clone)]
pub enum HookResult {
    /// La operación debe continuar.
    Continue,
    /// La operación debe abortarse (con razón).
    Abort(String),
}

/// Trait para handlers de hooks.
///
/// Los hooks pre-operación pueden vetar (abortar) una operación.
/// Los hooks post-operación son informativos.
///
/// # Ejemplo
/// ```
/// use bikodb_core::plugin::{HookHandler, HookContext, HookResult};
///
/// struct AuditHook;
///
/// impl HookHandler for AuditHook {
///     fn name(&self) -> &str { "audit" }
///     fn handle(&self, ctx: &HookContext) -> HookResult {
///         println!("Operation: {:?} on node {:?}", ctx.hook_point, ctx.node_id);
///         HookResult::Continue
///     }
/// }
/// ```
pub trait HookHandler: Send + Sync {
    fn name(&self) -> &str;
    fn handle(&self, ctx: &HookContext) -> HookResult;
}

// ── User Defined Functions (UDFs) ───────────────────────────────────────────

/// Tipo de retorno de una UDF.
#[derive(Debug, Clone)]
pub enum UdfReturn {
    Scalar(Value),
    List(Vec<Value>),
}

/// Trait para funciones definidas por el usuario (UDFs).
///
/// Se registran en el motor y pueden usarse en queries SQL/Cypher/Gremlin.
///
/// # Ejemplo
/// ```
/// use bikodb_core::plugin::{UserDefinedFn, UdfReturn};
/// use bikodb_core::value::Value;
///
/// struct UpperCase;
///
/// impl UserDefinedFn for UpperCase {
///     fn name(&self) -> &str { "UPPER" }
///     fn call(&self, args: &[Value]) -> UdfReturn {
///         match args.first() {
///             Some(Value::String(s)) => UdfReturn::Scalar(Value::string(s.to_uppercase())),
///             _ => UdfReturn::Scalar(Value::Null),
///         }
///     }
///     fn description(&self) -> &str { "Converts string to upper case" }
///     fn param_count(&self) -> usize { 1 }
/// }
/// ```
pub trait UserDefinedFn: Send + Sync {
    /// Nombre de la función (usado en queries).
    fn name(&self) -> &str;
    /// Ejecuta la función con los argumentos dados.
    fn call(&self, args: &[Value]) -> UdfReturn;
    /// Descripción de la función.
    fn description(&self) -> &str { "" }
    /// Número esperado de parámetros (0 = variadic).
    fn param_count(&self) -> usize { 0 }
}

impl fmt::Debug for dyn UserDefinedFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UDF({}, params={})", self.name(), self.param_count())
    }
}

// ── Graph Algorithm Plugin ──────────────────────────────────────────────────

/// Resultado de un algoritmo de grafos.
#[derive(Debug, Clone)]
pub struct AlgorithmResult {
    /// Scores/valores por nodo (NodeId → f64).
    pub node_scores: HashMap<u64, f64>,
    /// Metadata adicional.
    pub metadata: HashMap<String, Value>,
}

/// Trait para algoritmos de grafos pluggables.
///
/// Permite registrar algoritmos custom (centralidad, community detection,
/// ranking, etc.) sin recompilar el motor.
///
/// # Ejemplo
/// ```
/// use bikodb_core::plugin::{GraphAlgorithmExt, AlgorithmResult, AlgorithmInput};
/// use bikodb_core::error::BikoResult;
///
/// struct DegreeCount;
///
/// impl GraphAlgorithmExt for DegreeCount {
///     fn name(&self) -> &str { "degree_count" }
///     fn execute(&self, input: &AlgorithmInput) -> BikoResult<AlgorithmResult> {
///         use std::collections::HashMap;
///         let mut scores = HashMap::new();
///         for &(src, dst) in &input.edges {
///             *scores.entry(src).or_insert(0.0) += 1.0;
///             *scores.entry(dst).or_insert(0.0) += 1.0;
///         }
///         Ok(AlgorithmResult { node_scores: scores, metadata: HashMap::new() })
///     }
///     fn description(&self) -> &str { "Counts degree per node" }
/// }
/// ```
pub trait GraphAlgorithmExt: Send + Sync {
    fn name(&self) -> &str;
    fn execute(&self, input: &AlgorithmInput) -> BikoResult<AlgorithmResult>;
    fn description(&self) -> &str { "" }
}

/// Input para ejecutar un algoritmo de grafos.
#[derive(Debug, Clone)]
pub struct AlgorithmInput {
    /// Lista de nodos: (node_id_raw, type_id).
    pub nodes: Vec<(u64, TypeId)>,
    /// Lista de edges: (source_id_raw, target_id_raw).
    pub edges: Vec<(u64, u64)>,
    /// Parámetros del algoritmo.
    pub params: HashMap<String, Value>,
}

impl fmt::Debug for dyn GraphAlgorithmExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GraphAlgorithm({})", self.name())
    }
}

// ── Distance Function Extension ─────────────────────────────────────────────

/// Trait para métricas de distancia custom (extensión de DistanceMetric).
///
/// Permite añadir métricas de distancia custom para vector search
/// sin recompilar el motor.
pub trait DistanceFnExt: Send + Sync {
    fn name(&self) -> &str;
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
}

impl fmt::Debug for dyn DistanceFnExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DistanceFn({})", self.name())
    }
}

// ── Inference Provider Extension ────────────────────────────────────────────

/// Trait para proveedores de inferencia custom (ONNX, TF, PyTorch, etc.).
///
/// Permite registrar backends de inferencia alternativos al MLP built-in.
///
/// # Ejemplo
/// ```
/// use bikodb_core::plugin::InferenceProviderExt;
///
/// struct OnnxProvider { /* model bytes, session, etc. */ }
///
/// impl InferenceProviderExt for OnnxProvider {
///     fn name(&self) -> &str { "onnx-bert" }
///     fn predict(&self, features: &[f32]) -> Vec<f32> {
///         // Run ONNX inference...
///         vec![0.0; 128]
///     }
///     fn input_dim(&self) -> usize { 768 }
///     fn output_dim(&self) -> usize { 128 }
/// }
/// ```
pub trait InferenceProviderExt: Send + Sync {
    fn name(&self) -> &str;
    fn predict(&self, features: &[f32]) -> Vec<f32>;
    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;
    fn description(&self) -> &str { "" }
}

impl fmt::Debug for dyn InferenceProviderExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InferenceProvider({}, {}→{})", self.name(), self.input_dim(), self.output_dim())
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // ── Test Plugin ─────────────────────────────────────────────────────

    struct TestPlugin {
        name: String,
    }

    impl TestPlugin {
        fn new(name: &str) -> Self {
            Self { name: name.to_string() }
        }
    }

    impl Plugin for TestPlugin {
        fn name(&self) -> &str { &self.name }
        fn version(&self) -> &str { "0.1.0" }
        fn description(&self) -> &str { "A test plugin" }
        fn init(&self, _ctx: &dyn PluginContext) -> BikoResult<()> { Ok(()) }
        fn shutdown(&self) -> BikoResult<()> { Ok(()) }
        fn as_any(&self) -> &dyn Any { self }
    }

    struct DummyContext;
    impl PluginContext for DummyContext {
        fn node_count(&self) -> usize { 0 }
        fn edge_count(&self) -> usize { 0 }
        fn registered_types(&self) -> Vec<String> { vec![] }
    }

    #[test]
    fn test_plugin_trait() {
        let p = TestPlugin::new("test");
        assert_eq!(p.name(), "test");
        assert_eq!(p.version(), "0.1.0");
        assert_eq!(p.description(), "A test plugin");
        assert!(p.init(&DummyContext).is_ok());
        assert!(p.shutdown().is_ok());
    }

    // ── Test Hooks ──────────────────────────────────────────────────────

    struct AllowHook;
    impl HookHandler for AllowHook {
        fn name(&self) -> &str { "allow" }
        fn handle(&self, _ctx: &HookContext) -> HookResult { HookResult::Continue }
    }

    struct DenyHook {
        reason: String,
    }
    impl HookHandler for DenyHook {
        fn name(&self) -> &str { "deny" }
        fn handle(&self, _ctx: &HookContext) -> HookResult {
            HookResult::Abort(self.reason.clone())
        }
    }

    #[test]
    fn test_hook_continue() {
        let hook = AllowHook;
        let ctx = HookContext::for_node(HookPoint::PreInsertNode, NodeId(1), TypeId(1));
        assert!(matches!(hook.handle(&ctx), HookResult::Continue));
    }

    #[test]
    fn test_hook_abort() {
        let hook = DenyHook { reason: "read-only mode".into() };
        let ctx = HookContext::for_node(HookPoint::PreInsertNode, NodeId(1), TypeId(1));
        match hook.handle(&ctx) {
            HookResult::Abort(r) => assert_eq!(r, "read-only mode"),
            _ => panic!("Expected abort"),
        }
    }

    #[test]
    fn test_hook_context_for_query() {
        let ctx = HookContext::for_query(HookPoint::PreQuery, "SELECT * FROM Person");
        assert_eq!(ctx.hook_point, HookPoint::PreQuery);
        assert_eq!(ctx.query_text.unwrap(), "SELECT * FROM Person");
        assert!(ctx.node_id.is_none());
    }

    #[test]
    fn test_hook_context_for_edge() {
        let ctx = HookContext::for_edge(HookPoint::PreInsertEdge, EdgeId(5), TypeId(10));
        assert_eq!(ctx.edge_id, Some(EdgeId(5)));
        assert_eq!(ctx.type_id, Some(TypeId(10)));
    }

    // ── Test UDFs ───────────────────────────────────────────────────────

    struct UpperFn;
    impl UserDefinedFn for UpperFn {
        fn name(&self) -> &str { "UPPER" }
        fn call(&self, args: &[Value]) -> UdfReturn {
            match args.first() {
                Some(Value::String(s)) => UdfReturn::Scalar(Value::string(s.to_uppercase())),
                _ => UdfReturn::Scalar(Value::Null),
            }
        }
        fn param_count(&self) -> usize { 1 }
    }

    struct ConcatFn;
    impl UserDefinedFn for ConcatFn {
        fn name(&self) -> &str { "CONCAT" }
        fn call(&self, args: &[Value]) -> UdfReturn {
            let parts: Vec<String> = args.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
            UdfReturn::Scalar(Value::string(parts.join("")))
        }
    }

    #[test]
    fn test_udf_upper() {
        let f = UpperFn;
        match f.call(&[Value::string("hello")]) {
            UdfReturn::Scalar(Value::String(s)) => assert_eq!(s, "HELLO"),
            _ => panic!("Expected string"),
        }
    }

    #[test]
    fn test_udf_concat() {
        let f = ConcatFn;
        match f.call(&[Value::string("a"), Value::string("b"), Value::string("c")]) {
            UdfReturn::Scalar(Value::String(s)) => assert_eq!(s, "abc"),
            _ => panic!("Expected string"),
        }
    }

    #[test]
    fn test_udf_null_on_wrong_type() {
        let f = UpperFn;
        match f.call(&[Value::Int(42)]) {
            UdfReturn::Scalar(Value::Null) => {} // ok
            _ => panic!("Expected Null"),
        }
    }

    // ── Test Graph Algorithm Extension ──────────────────────────────────

    struct DegreeAlg;
    impl GraphAlgorithmExt for DegreeAlg {
        fn name(&self) -> &str { "degree" }
        fn execute(&self, input: &AlgorithmInput) -> BikoResult<AlgorithmResult> {
            let mut scores = HashMap::new();
            for &(src, dst) in &input.edges {
                *scores.entry(src).or_insert(0.0) += 1.0;
                *scores.entry(dst).or_insert(0.0) += 1.0;
            }
            Ok(AlgorithmResult { node_scores: scores, metadata: HashMap::new() })
        }
    }

    #[test]
    fn test_graph_algorithm_ext() {
        let alg = DegreeAlg;
        let input = AlgorithmInput {
            nodes: vec![(1, TypeId(1)), (2, TypeId(1)), (3, TypeId(1))],
            edges: vec![(1, 2), (1, 3), (2, 3)],
            params: HashMap::new(),
        };
        let result = alg.execute(&input).unwrap();
        assert_eq!(*result.node_scores.get(&1).unwrap(), 2.0);
        assert_eq!(*result.node_scores.get(&2).unwrap(), 2.0);
        assert_eq!(*result.node_scores.get(&3).unwrap(), 2.0);
    }

    // ── Test Distance Function Extension ────────────────────────────────

    struct ManhattanDistance;
    impl DistanceFnExt for ManhattanDistance {
        fn name(&self) -> &str { "manhattan" }
        fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
        }
    }

    #[test]
    fn test_custom_distance_fn() {
        let d = ManhattanDistance;
        let dist = d.distance(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((dist - 9.0).abs() < 1e-5);
    }

    // ── Test Inference Provider Extension ────────────────────────────────

    struct IdentityProvider;
    impl InferenceProviderExt for IdentityProvider {
        fn name(&self) -> &str { "identity" }
        fn predict(&self, features: &[f32]) -> Vec<f32> { features.to_vec() }
        fn input_dim(&self) -> usize { 4 }
        fn output_dim(&self) -> usize { 4 }
    }

    #[test]
    fn test_inference_provider_ext() {
        let p = IdentityProvider;
        let out = p.predict(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(p.input_dim(), 4);
        assert_eq!(p.output_dim(), 4);
    }
}
