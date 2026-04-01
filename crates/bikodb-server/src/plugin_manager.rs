// =============================================================================
// bikodb-server::plugin_manager — Gestor de plugins
// =============================================================================
// Gestiona el ciclo de vida de plugins, despacha hooks, y registra UDFs
// y algoritmos custom. Thread-safe con parking_lot::RwLock.
//
// ## Componentes
//   PluginManager        → Registro central de plugins y extensiones
//   PluginManagerContext  → Implementación de PluginContext
// =============================================================================

use bikodb_core::error::{BikoError, BikoResult};
use bikodb_core::plugin::{
    AlgorithmInput, AlgorithmResult, DistanceFnExt, GraphAlgorithmExt, HookContext,
    HookHandler, HookPoint, HookResult, InferenceProviderExt, Plugin, PluginContext,
    PluginInfo, UdfReturn, UserDefinedFn,
};
use bikodb_core::value::Value;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Contexto simple para inicialización de plugins.
pub struct PluginManagerContext {
    pub node_count: usize,
    pub edge_count: usize,
    pub types: Vec<String>,
}

impl PluginContext for PluginManagerContext {
    fn node_count(&self) -> usize {
        self.node_count
    }
    fn edge_count(&self) -> usize {
        self.edge_count
    }
    fn registered_types(&self) -> Vec<String> {
        self.types.clone()
    }
}

/// Gestor central de plugins de BikoDB.
///
/// Administra:
/// - Plugins con lifecycle (init/shutdown)
/// - Hooks pre/post operación con veto
/// - User-defined functions (UDFs)
/// - Algoritmos de grafos custom
/// - Métricas de distancia custom
/// - Proveedores de inferencia custom
pub struct PluginManager {
    /// Plugins registrados: name → plugin.
    plugins: RwLock<HashMap<String, Arc<dyn Plugin>>>,
    /// Hooks registrados: punto → lista de handlers.
    hooks: RwLock<HashMap<HookPoint, Vec<Arc<dyn HookHandler>>>>,
    /// UDFs registradas: name → function.
    udfs: RwLock<HashMap<String, Arc<dyn UserDefinedFn>>>,
    /// Algoritmos de grafos custom: name → algorithm.
    algorithms: RwLock<HashMap<String, Arc<dyn GraphAlgorithmExt>>>,
    /// Métricas de distancia custom: name → function.
    distance_fns: RwLock<HashMap<String, Arc<dyn DistanceFnExt>>>,
    /// Proveedores de inferencia custom: name → provider.
    inference_providers: RwLock<HashMap<String, Arc<dyn InferenceProviderExt>>>,
}

impl PluginManager {
    pub fn new() -> Self {
        Self {
            plugins: RwLock::new(HashMap::new()),
            hooks: RwLock::new(HashMap::new()),
            udfs: RwLock::new(HashMap::new()),
            algorithms: RwLock::new(HashMap::new()),
            distance_fns: RwLock::new(HashMap::new()),
            inference_providers: RwLock::new(HashMap::new()),
        }
    }

    // ── Plugin Lifecycle ────────────────────────────────────────────────

    /// Registra e inicializa un plugin.
    pub fn register_plugin(
        &self,
        plugin: Arc<dyn Plugin>,
        ctx: &dyn PluginContext,
    ) -> BikoResult<()> {
        let name = plugin.name().to_string();
        {
            let plugins = self.plugins.read();
            if plugins.contains_key(&name) {
                return Err(BikoError::Generic(format!(
                    "Plugin '{}' already registered",
                    name
                )));
            }
        }
        plugin.init(ctx)?;
        self.plugins.write().insert(name, plugin);
        Ok(())
    }

    /// Desregistra y shutdown de un plugin.
    pub fn unregister_plugin(&self, name: &str) -> BikoResult<()> {
        let plugin = self
            .plugins
            .write()
            .remove(name)
            .ok_or_else(|| BikoError::Generic(format!("Plugin '{}' not found", name)))?;
        plugin.shutdown()?;
        Ok(())
    }

    /// Lista info de todos los plugins registrados.
    pub fn list_plugins(&self) -> Vec<PluginInfo> {
        self.plugins
            .read()
            .values()
            .map(|p| PluginInfo {
                name: p.name().to_string(),
                version: p.version().to_string(),
                description: p.description().to_string(),
                active: true,
            })
            .collect()
    }

    /// Obtiene un plugin por nombre.
    pub fn get_plugin(&self, name: &str) -> Option<Arc<dyn Plugin>> {
        self.plugins.read().get(name).cloned()
    }

    /// Número de plugins registrados.
    pub fn plugin_count(&self) -> usize {
        self.plugins.read().len()
    }

    // ── Hook System ─────────────────────────────────────────────────────

    /// Registra un hook handler para un punto de intercepción.
    pub fn register_hook(&self, hook_point: HookPoint, handler: Arc<dyn HookHandler>) {
        self.hooks
            .write()
            .entry(hook_point)
            .or_default()
            .push(handler);
    }

    /// Despacha un hook a todos los handlers registrados.
    ///
    /// Para hooks Pre* retorna `Abort` si **cualquier** handler lo solicita.
    /// Para hooks Post* siempre retorna `Continue`.
    pub fn dispatch_hook(&self, ctx: &HookContext) -> HookResult {
        let hooks = self.hooks.read();
        if let Some(handlers) = hooks.get(&ctx.hook_point) {
            for handler in handlers {
                match handler.handle(ctx) {
                    HookResult::Abort(reason) => return HookResult::Abort(reason),
                    HookResult::Continue => {}
                }
            }
        }
        HookResult::Continue
    }

    /// Número de hooks registrados para un punto.
    pub fn hook_count(&self, hook_point: HookPoint) -> usize {
        self.hooks
            .read()
            .get(&hook_point)
            .map(|h| h.len())
            .unwrap_or(0)
    }

    // ── UDFs ────────────────────────────────────────────────────────────

    /// Registra una UDF.
    pub fn register_udf(&self, udf: Arc<dyn UserDefinedFn>) -> BikoResult<()> {
        let name = udf.name().to_string();
        let mut udfs = self.udfs.write();
        if udfs.contains_key(&name) {
            return Err(BikoError::Generic(format!("UDF '{}' already registered", name)));
        }
        udfs.insert(name, udf);
        Ok(())
    }

    /// Desregistra una UDF.
    pub fn unregister_udf(&self, name: &str) -> BikoResult<()> {
        self.udfs
            .write()
            .remove(name)
            .ok_or_else(|| BikoError::Generic(format!("UDF '{}' not found", name)))?;
        Ok(())
    }

    /// Ejecuta una UDF por nombre.
    pub fn call_udf(&self, name: &str, args: &[Value]) -> BikoResult<UdfReturn> {
        let udfs = self.udfs.read();
        let udf = udfs
            .get(name)
            .ok_or_else(|| BikoError::Generic(format!("UDF '{}' not found", name)))?;
        Ok(udf.call(args))
    }

    /// Lista nombres de UDFs registradas.
    pub fn list_udfs(&self) -> Vec<String> {
        self.udfs.read().keys().cloned().collect()
    }

    // ── Graph Algorithms ────────────────────────────────────────────────

    /// Registra un algoritmo de grafos custom.
    pub fn register_algorithm(&self, alg: Arc<dyn GraphAlgorithmExt>) -> BikoResult<()> {
        let name = alg.name().to_string();
        let mut algs = self.algorithms.write();
        if algs.contains_key(&name) {
            return Err(BikoError::Generic(format!(
                "Algorithm '{}' already registered",
                name
            )));
        }
        algs.insert(name, alg);
        Ok(())
    }

    /// Ejecuta un algoritmo custom por nombre.
    pub fn run_algorithm(
        &self,
        name: &str,
        input: &AlgorithmInput,
    ) -> BikoResult<AlgorithmResult> {
        let algs = self.algorithms.read();
        let alg = algs
            .get(name)
            .ok_or_else(|| BikoError::Generic(format!("Algorithm '{}' not found", name)))?;
        alg.execute(input)
    }

    /// Lista nombres de algoritmos registrados.
    pub fn list_algorithms(&self) -> Vec<String> {
        self.algorithms.read().keys().cloned().collect()
    }

    // ── Custom Distance Functions ───────────────────────────────────────

    /// Registra una métrica de distancia custom.
    pub fn register_distance_fn(&self, dist: Arc<dyn DistanceFnExt>) -> BikoResult<()> {
        let name = dist.name().to_string();
        let mut fns = self.distance_fns.write();
        if fns.contains_key(&name) {
            return Err(BikoError::Generic(format!(
                "Distance function '{}' already registered",
                name
            )));
        }
        fns.insert(name, dist);
        Ok(())
    }

    /// Calcula distancia usando una métrica custom.
    pub fn custom_distance(&self, name: &str, a: &[f32], b: &[f32]) -> BikoResult<f32> {
        let fns = self.distance_fns.read();
        let f = fns
            .get(name)
            .ok_or_else(|| BikoError::Generic(format!("Distance function '{}' not found", name)))?;
        Ok(f.distance(a, b))
    }

    /// Lista métricas de distancia custom registradas.
    pub fn list_distance_fns(&self) -> Vec<String> {
        self.distance_fns.read().keys().cloned().collect()
    }

    // ── Custom Inference Providers ──────────────────────────────────────

    /// Registra un proveedor de inferencia custom.
    pub fn register_inference_provider(
        &self,
        provider: Arc<dyn InferenceProviderExt>,
    ) -> BikoResult<()> {
        let name = provider.name().to_string();
        let mut providers = self.inference_providers.write();
        if providers.contains_key(&name) {
            return Err(BikoError::Generic(format!(
                "Inference provider '{}' already registered",
                name
            )));
        }
        providers.insert(name, provider);
        Ok(())
    }

    /// Ejecuta inferencia con un proveedor custom.
    pub fn custom_predict(&self, name: &str, features: &[f32]) -> BikoResult<Vec<f32>> {
        let providers = self.inference_providers.read();
        let p = providers.get(name).ok_or_else(|| {
            BikoError::Generic(format!("Inference provider '{}' not found", name))
        })?;
        Ok(p.predict(features))
    }

    /// Lista proveedores de inferencia custom registrados.
    pub fn list_inference_providers(&self) -> Vec<String> {
        self.inference_providers.read().keys().cloned().collect()
    }

    /// Shutdown all registered plugins (called during Database drop or explicit shutdown).
    pub fn shutdown_all(&self) {
        let plugins = self.plugins.read();
        for plugin in plugins.values() {
            let _ = plugin.shutdown();
        }
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;

    // ── Test Plugin ─────────────────────────────────────────────────────

    struct TestPlugin {
        name: String,
        version: String,
    }

    impl TestPlugin {
        fn new(name: &str, ver: &str) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                version: ver.to_string(),
            })
        }
    }

    impl Plugin for TestPlugin {
        fn name(&self) -> &str { &self.name }
        fn version(&self) -> &str { &self.version }
        fn init(&self, _ctx: &dyn PluginContext) -> BikoResult<()> { Ok(()) }
        fn shutdown(&self) -> BikoResult<()> { Ok(()) }
        fn as_any(&self) -> &dyn Any { self }
    }

    fn dummy_ctx() -> PluginManagerContext {
        PluginManagerContext {
            node_count: 0,
            edge_count: 0,
            types: vec![],
        }
    }

    // ── Plugin Lifecycle Tests ──────────────────────────────────────────

    #[test]
    fn test_register_and_list_plugins() {
        let pm = PluginManager::new();
        let p1 = TestPlugin::new("alpha", "1.0.0");
        let p2 = TestPlugin::new("beta", "2.0.0");
        pm.register_plugin(p1, &dummy_ctx()).unwrap();
        pm.register_plugin(p2, &dummy_ctx()).unwrap();
        assert_eq!(pm.plugin_count(), 2);

        let infos = pm.list_plugins();
        assert_eq!(infos.len(), 2);
        let names: Vec<_> = infos.iter().map(|i| i.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }

    #[test]
    fn test_duplicate_plugin_rejected() {
        let pm = PluginManager::new();
        let p1 = TestPlugin::new("dup", "1.0.0");
        let p2 = TestPlugin::new("dup", "1.0.1");
        pm.register_plugin(p1, &dummy_ctx()).unwrap();
        assert!(pm.register_plugin(p2, &dummy_ctx()).is_err());
    }

    #[test]
    fn test_unregister_plugin() {
        let pm = PluginManager::new();
        let p = TestPlugin::new("temp", "1.0.0");
        pm.register_plugin(p, &dummy_ctx()).unwrap();
        assert_eq!(pm.plugin_count(), 1);
        pm.unregister_plugin("temp").unwrap();
        assert_eq!(pm.plugin_count(), 0);
    }

    #[test]
    fn test_unregister_unknown_fails() {
        let pm = PluginManager::new();
        assert!(pm.unregister_plugin("ghost").is_err());
    }

    #[test]
    fn test_get_plugin() {
        let pm = PluginManager::new();
        let p = TestPlugin::new("finder", "1.0.0");
        pm.register_plugin(p, &dummy_ctx()).unwrap();
        let found = pm.get_plugin("finder");
        assert!(found.is_some());
        assert_eq!(found.unwrap().version(), "1.0.0");
        assert!(pm.get_plugin("nope").is_none());
    }

    // ── Hook Tests ──────────────────────────────────────────────────────

    struct CountHook {
        name: String,
    }
    impl HookHandler for CountHook {
        fn name(&self) -> &str { &self.name }
        fn handle(&self, _ctx: &HookContext) -> HookResult { HookResult::Continue }
    }

    struct VetoHook;
    impl HookHandler for VetoHook {
        fn name(&self) -> &str { "veto" }
        fn handle(&self, _ctx: &HookContext) -> HookResult {
            HookResult::Abort("vetoed!".into())
        }
    }

    #[test]
    fn test_hooks_continue() {
        let pm = PluginManager::new();
        pm.register_hook(
            HookPoint::PreInsertNode,
            Arc::new(CountHook { name: "h1".into() }),
        );
        pm.register_hook(
            HookPoint::PreInsertNode,
            Arc::new(CountHook { name: "h2".into() }),
        );
        assert_eq!(pm.hook_count(HookPoint::PreInsertNode), 2);

        let ctx = HookContext::for_node(
            HookPoint::PreInsertNode,
            bikodb_core::types::NodeId(1),
            bikodb_core::types::TypeId(1),
        );
        assert!(matches!(pm.dispatch_hook(&ctx), HookResult::Continue));
    }

    #[test]
    fn test_hooks_abort() {
        let pm = PluginManager::new();
        pm.register_hook(HookPoint::PreInsertNode, Arc::new(VetoHook));
        pm.register_hook(
            HookPoint::PreInsertNode,
            Arc::new(CountHook { name: "after_veto".into() }),
        );

        let ctx = HookContext::for_node(
            HookPoint::PreInsertNode,
            bikodb_core::types::NodeId(1),
            bikodb_core::types::TypeId(1),
        );
        match pm.dispatch_hook(&ctx) {
            HookResult::Abort(reason) => assert_eq!(reason, "vetoed!"),
            _ => panic!("Expected abort"),
        }
    }

    #[test]
    fn test_no_hooks_continue() {
        let pm = PluginManager::new();
        let ctx = HookContext::for_node(
            HookPoint::PostInsertNode,
            bikodb_core::types::NodeId(1),
            bikodb_core::types::TypeId(1),
        );
        assert!(matches!(pm.dispatch_hook(&ctx), HookResult::Continue));
    }

    // ── UDF Tests ───────────────────────────────────────────────────────

    struct AddFn;
    impl UserDefinedFn for AddFn {
        fn name(&self) -> &str { "ADD" }
        fn call(&self, args: &[Value]) -> UdfReturn {
            let sum: i64 = args.iter().filter_map(|v| match v {
                Value::Int(n) => Some(*n),
                _ => None,
            }).sum();
            UdfReturn::Scalar(Value::Int(sum))
        }
        fn param_count(&self) -> usize { 2 }
    }

    struct LenFn;
    impl UserDefinedFn for LenFn {
        fn name(&self) -> &str { "LEN" }
        fn call(&self, args: &[Value]) -> UdfReturn {
            match args.first() {
                Some(Value::String(s)) => UdfReturn::Scalar(Value::Int(s.len() as i64)),
                Some(Value::List(l)) => UdfReturn::Scalar(Value::Int(l.len() as i64)),
                _ => UdfReturn::Scalar(Value::Int(0)),
            }
        }
        fn param_count(&self) -> usize { 1 }
    }

    #[test]
    fn test_register_and_call_udf() {
        let pm = PluginManager::new();
        pm.register_udf(Arc::new(AddFn)).unwrap();
        let result = pm.call_udf("ADD", &[Value::Int(3), Value::Int(7)]).unwrap();
        match result {
            UdfReturn::Scalar(Value::Int(n)) => assert_eq!(n, 10),
            _ => panic!("Expected int"),
        }
    }

    #[test]
    fn test_duplicate_udf_rejected() {
        let pm = PluginManager::new();
        pm.register_udf(Arc::new(AddFn)).unwrap();
        assert!(pm.register_udf(Arc::new(AddFn)).is_err());
    }

    #[test]
    fn test_unregister_udf() {
        let pm = PluginManager::new();
        pm.register_udf(Arc::new(AddFn)).unwrap();
        pm.unregister_udf("ADD").unwrap();
        assert!(pm.call_udf("ADD", &[Value::Int(1)]).is_err());
    }

    #[test]
    fn test_list_udfs() {
        let pm = PluginManager::new();
        pm.register_udf(Arc::new(AddFn)).unwrap();
        pm.register_udf(Arc::new(LenFn)).unwrap();
        let names = pm.list_udfs();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_call_unknown_udf_fails() {
        let pm = PluginManager::new();
        assert!(pm.call_udf("NOPE", &[]).is_err());
    }

    // ── Algorithm Tests ─────────────────────────────────────────────────

    struct WeightedDegreeAlg;
    impl GraphAlgorithmExt for WeightedDegreeAlg {
        fn name(&self) -> &str { "weighted_degree" }
        fn execute(&self, input: &AlgorithmInput) -> BikoResult<AlgorithmResult> {
            let weight = match input.params.get("weight") {
                Some(Value::Float(w)) => *w,
                _ => 1.0,
            };
            let mut scores = HashMap::new();
            for &(src, dst) in &input.edges {
                *scores.entry(src).or_insert(0.0) += weight;
                *scores.entry(dst).or_insert(0.0) += weight;
            }
            Ok(AlgorithmResult {
                node_scores: scores,
                metadata: HashMap::new(),
            })
        }
    }

    #[test]
    fn test_register_and_run_algorithm() {
        let pm = PluginManager::new();
        pm.register_algorithm(Arc::new(WeightedDegreeAlg)).unwrap();

        let mut params = HashMap::new();
        params.insert("weight".to_string(), Value::Float(2.0));

        let input = AlgorithmInput {
            nodes: vec![
                (1, bikodb_core::types::TypeId(1)),
                (2, bikodb_core::types::TypeId(1)),
            ],
            edges: vec![(1, 2)],
            params,
        };
        let result = pm.run_algorithm("weighted_degree", &input).unwrap();
        assert_eq!(*result.node_scores.get(&1).unwrap(), 2.0);
        assert_eq!(*result.node_scores.get(&2).unwrap(), 2.0);
    }

    #[test]
    fn test_run_unknown_algorithm_fails() {
        let pm = PluginManager::new();
        let input = AlgorithmInput {
            nodes: vec![],
            edges: vec![],
            params: HashMap::new(),
        };
        assert!(pm.run_algorithm("unknown", &input).is_err());
    }

    // ── Distance Function Tests ─────────────────────────────────────────

    struct ChebyshevDistance;
    impl DistanceFnExt for ChebyshevDistance {
        fn name(&self) -> &str { "chebyshev" }
        fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        }
    }

    #[test]
    fn test_custom_distance_fn() {
        let pm = PluginManager::new();
        pm.register_distance_fn(Arc::new(ChebyshevDistance)).unwrap();
        let d = pm.custom_distance("chebyshev", &[1.0, 5.0, 3.0], &[4.0, 2.0, 3.0]).unwrap();
        assert!((d - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_unknown_distance_fn_fails() {
        let pm = PluginManager::new();
        assert!(pm.custom_distance("unknown", &[1.0], &[2.0]).is_err());
    }

    // ── Inference Provider Tests ────────────────────────────────────────

    struct DoubleProvider;
    impl InferenceProviderExt for DoubleProvider {
        fn name(&self) -> &str { "double" }
        fn predict(&self, features: &[f32]) -> Vec<f32> {
            features.iter().map(|x| x * 2.0).collect()
        }
        fn input_dim(&self) -> usize { 3 }
        fn output_dim(&self) -> usize { 3 }
    }

    #[test]
    fn test_custom_inference_provider() {
        let pm = PluginManager::new();
        pm.register_inference_provider(Arc::new(DoubleProvider)).unwrap();
        let out = pm.custom_predict("double", &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(out, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_unknown_inference_provider_fails() {
        let pm = PluginManager::new();
        assert!(pm.custom_predict("unknown", &[1.0]).is_err());
    }

    #[test]
    fn test_list_inference_providers() {
        let pm = PluginManager::new();
        pm.register_inference_provider(Arc::new(DoubleProvider)).unwrap();
        let names = pm.list_inference_providers();
        assert_eq!(names.len(), 1);
        assert!(names.contains(&"double".to_string()));
    }

    // ── Shutdown Tests ──────────────────────────────────────────────────

    #[test]
    fn test_shutdown_all() {
        let pm = PluginManager::new();
        pm.register_plugin(TestPlugin::new("p1", "1.0"), &dummy_ctx()).unwrap();
        pm.register_plugin(TestPlugin::new("p2", "1.0"), &dummy_ctx()).unwrap();
        pm.shutdown_all(); // should not panic
        assert_eq!(pm.plugin_count(), 2); // still registered, just shutdown
    }

    // ── Integration: Multiple extension types ───────────────────────────

    #[test]
    fn test_full_plugin_ecosystem() {
        let pm = PluginManager::new();

        // Register plugin
        pm.register_plugin(TestPlugin::new("ecosystem", "1.0.0"), &dummy_ctx()).unwrap();

        // Register hooks
        pm.register_hook(HookPoint::PreInsertNode, Arc::new(CountHook { name: "h1".into() }));
        pm.register_hook(HookPoint::PostInsertNode, Arc::new(CountHook { name: "h2".into() }));

        // Register UDFs
        pm.register_udf(Arc::new(AddFn)).unwrap();
        pm.register_udf(Arc::new(LenFn)).unwrap();

        // Register algorithm
        pm.register_algorithm(Arc::new(WeightedDegreeAlg)).unwrap();

        // Register distance fn
        pm.register_distance_fn(Arc::new(ChebyshevDistance)).unwrap();

        // Register inference provider
        pm.register_inference_provider(Arc::new(DoubleProvider)).unwrap();

        // Verify counts
        assert_eq!(pm.plugin_count(), 1);
        assert_eq!(pm.hook_count(HookPoint::PreInsertNode), 1);
        assert_eq!(pm.hook_count(HookPoint::PostInsertNode), 1);
        assert_eq!(pm.list_udfs().len(), 2);
        assert_eq!(pm.list_algorithms().len(), 1);
        assert_eq!(pm.list_distance_fns().len(), 1);
        assert_eq!(pm.list_inference_providers().len(), 1);

        // Use everything
        let udf_result = pm.call_udf("ADD", &[Value::Int(5), Value::Int(3)]).unwrap();
        match udf_result {
            UdfReturn::Scalar(Value::Int(n)) => assert_eq!(n, 8),
            _ => panic!("Wrong UDF result"),
        }

        let ctx = HookContext::for_node(
            HookPoint::PreInsertNode,
            bikodb_core::types::NodeId(1),
            bikodb_core::types::TypeId(1),
        );
        assert!(matches!(pm.dispatch_hook(&ctx), HookResult::Continue));
    }
}
