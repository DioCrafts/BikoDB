// =============================================================================
// bikodb-server::database — Fachada principal del motor
// =============================================================================
// Coordina todos los componentes: grafo, storage, query, AI, en una API unificada.
// En modo embedded, esta es la interfaz que usa el desarrollador directamente.
// =============================================================================

use bikodb_ai::embedding::DistanceMetric;
use bikodb_ai::hnsw::{HnswConfig, HnswIndex};
use bikodb_ai::incremental::{EmbeddingPipeline, EmbeddingPipelineConfig, EmbeddingStrategy, EventBus, GraphEvent, IncrementalInference, PropertyFetcher};
use bikodb_ai::inference::{InferenceModel as AiInferenceModel, InferenceRegistry};
use bikodb_core::plugin::{
    AlgorithmInput, AlgorithmResult, DistanceFnExt, GraphAlgorithmExt, HookContext,
    HookHandler, HookPoint, HookResult, InferenceProviderExt, Plugin, UdfReturn,
    UserDefinedFn,
};
use crate::plugin_manager::{PluginManager, PluginManagerContext};
use bikodb_cluster::{ClusterManager, ClusterNodeId, QueryRouter, ShardedGraph, ShardStrategy};
use bikodb_core::error::{BikoError, BikoResult};
use bikodb_core::record::Direction;
use bikodb_core::types::{EdgeId, NodeId, TypeId};
use bikodb_core::value::Value;
use bikodb_execution::operator::Row;
use bikodb_execution::pipeline;
use bikodb_graph::ConcurrentGraph;
use bikodb_query::cypher::{self, LabelMap, RelMap};
use bikodb_query::gremlin;
use bikodb_query::sql::{self, PropMap, TypeMap};
use bikodb_resource::monitor::ResourceMonitor;
use std::collections::HashMap;
use std::sync::Arc;

/// Resultado de búsqueda semántica enriquecido con contexto del grafo.
#[derive(Debug, Clone)]
pub struct SemanticSearchResult {
    /// ID del nodo encontrado.
    pub node_id: NodeId,
    /// Distancia al query vector (menor = más similar).
    pub distance: f32,
    /// Propiedades del nodo (nombre → valor).
    pub properties: Vec<(String, Value)>,
    /// IDs de vecinos directos (salientes).
    pub neighbors: Vec<NodeId>,
}

/// Callback para suscripciones de live queries.
pub type LiveQueryCallback = Arc<dyn Fn(&[Row]) + Send + Sync>;

/// Suscripción a un live query que se re-ejecuta cuando el grafo cambia.
struct LiveQueryEntry {
    /// Lenguaje: "sql", "cypher", o "gremlin".
    language: String,
    /// Query string.
    query: String,
    /// Callback a invocar con resultados actualizados.
    callback: LiveQueryCallback,
    /// Último resultado conocido (para detectar cambios).
    last_result_len: parking_lot::Mutex<usize>,
}

/// Motor de base de datos BikoDB (modo embedded).
///
/// Coordina grafo, storage, query parsing y métricas.
///
/// # Ejemplo
/// ```
/// use bikodb_server::Database;
/// use bikodb_core::types::TypeId;
/// use bikodb_core::value::Value;
///
/// let db = Database::new();
/// db.register_type("Person", TypeId(1));
/// db.register_property("name", 0);
/// db.register_property("age", 1);
///
/// let alice = db.create_node(TypeId(1), vec![
///     ("name", Value::from("Alice")),
///     ("age", Value::Int(30)),
/// ]);
/// let bob = db.create_node(TypeId(1), vec![
///     ("name", Value::from("Bob")),
///     ("age", Value::Int(25)),
/// ]);
///
/// db.create_edge(alice, bob, TypeId(10));
///
/// let results = db.query_sql("SELECT * FROM Person WHERE age > 20").unwrap();
/// assert_eq!(results.len(), 2);
/// ```
pub struct Database {
    graph: Arc<ConcurrentGraph>,
    type_map: parking_lot::RwLock<TypeMap>,
    prop_map: parking_lot::RwLock<PropMap>,
    /// Reverse map: prop name ← prop id
    prop_names: parking_lot::RwLock<HashMap<u16, String>>,
    /// Relationship type name → TypeId mapping (for Cypher/Gremlin)
    rel_map: parking_lot::RwLock<RelMap>,
    monitor: ResourceMonitor,
    /// AI: Event bus for graph mutation events.
    event_bus: Arc<EventBus>,
    /// AI: HNSW vector index for similarity search.
    hnsw_index: Option<Arc<HnswIndex>>,
    /// AI: Embedding pipeline for auto-computing embeddings.
    embedding_pipeline: Option<Arc<EmbeddingPipeline>>,
    /// AI: Inference model registry.
    inference_registry: parking_lot::RwLock<InferenceRegistry>,
    /// AI: Incremental inference engine (auto-re-infer on mutations).
    incremental_inference: Option<Arc<IncrementalInference>>,
    /// Real-time: Live query subscriptions.
    live_queries: Arc<parking_lot::RwLock<Vec<Arc<LiveQueryEntry>>>>,
    /// Real-time: Async event channel sender.
    async_event_tx: Option<std::sync::mpsc::Sender<GraphEvent>>,
    /// Real-time: Async event channel receiver (wrapped for shared access).
    async_event_rx: Option<Arc<parking_lot::Mutex<std::sync::mpsc::Receiver<GraphEvent>>>>,
    /// Cluster: Query router for sharded/distributed graph queries.
    cluster_router: Option<Arc<QueryRouter>>,
    /// Multi-model: Document store (collections of JSON-like documents).
    doc_store: Arc<bikodb_graph::document::DocumentStore>,
    /// Plugin system: manages plugins, hooks, UDFs, algorithms, distance fns.
    plugin_manager: Arc<PluginManager>,
}

impl Database {
    /// Crea una base de datos vacía.
    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self {
            graph: Arc::new(ConcurrentGraph::new()),
            type_map: parking_lot::RwLock::new(TypeMap::new()),
            prop_map: parking_lot::RwLock::new(PropMap::new()),
            prop_names: parking_lot::RwLock::new(HashMap::new()),
            rel_map: parking_lot::RwLock::new(RelMap::new()),
            monitor: ResourceMonitor::new(),
            event_bus: Arc::new(EventBus::new()),
            hnsw_index: None,
            embedding_pipeline: None,
            inference_registry: parking_lot::RwLock::new(InferenceRegistry::new()),
            incremental_inference: None,
            live_queries: Arc::new(parking_lot::RwLock::new(Vec::new())),
            async_event_tx: Some(tx),
            async_event_rx: Some(Arc::new(parking_lot::Mutex::new(rx))),
            cluster_router: None,
            doc_store: Arc::new(bikodb_graph::document::DocumentStore::new()),
            plugin_manager: Arc::new(PluginManager::new()),
        }
    }

    /// Registra un tipo (label) en el schema.
    pub fn register_type(&self, name: &str, type_id: TypeId) {
        self.type_map.write().insert(name.to_string(), type_id);
    }

    /// Registra una propiedad en el schema.
    pub fn register_property(&self, name: &str, prop_id: u16) {
        self.prop_map.write().insert(name.to_string(), prop_id);
        self.prop_names.write().insert(prop_id, name.to_string());
    }

    /// Crea un nodo con propiedades (por nombre).
    ///
    /// Emite automáticamente `GraphEvent::NodeAdded` al event bus.
    pub fn create_node(&self, type_id: TypeId, props: Vec<(&str, Value)>) -> NodeId {
        let prop_map = self.prop_map.read();
        let properties: Vec<(u16, Value)> = props
            .into_iter()
            .filter_map(|(name, val)| {
                prop_map.get(name).map(|&id| (id, val))
            })
            .collect();

        let node_id = self.graph.insert_node_with_props(type_id, properties.clone());

        // Emit event (no-op if no listeners)
        let event = GraphEvent::NodeAdded {
            node_id,
            properties,
        };
        self.event_bus.emit(event.clone());
        self.dispatch_async(event);
        self.notify_live_queries();

        node_id
    }

    /// Crea un edge entre dos nodos.
    ///
    /// Emite automáticamente `GraphEvent::EdgeAdded` al event bus.
    pub fn create_edge(&self, source: NodeId, target: NodeId, type_id: TypeId) -> BikoResult<EdgeId> {
        let edge_id = self.graph.insert_edge(source, target, type_id)?;
        self.event_bus.emit(GraphEvent::EdgeAdded { source, target });
        self.dispatch_async(GraphEvent::EdgeAdded { source, target });
        self.notify_live_queries();
        Ok(edge_id)
    }

    /// Elimina un edge del grafo.
    ///
    /// Emite `GraphEvent::EdgeRemoved` al event bus.
    pub fn remove_edge(&self, edge_id: EdgeId) -> BikoResult<()> {
        let edge = self.graph.get_edge(edge_id)
            .ok_or_else(|| bikodb_core::error::BikoError::Generic(
                format!("Edge {:?} not found", edge_id),
            ))?;
        let source = edge.source;
        let target = edge.target;
        self.graph.remove_edge(edge_id)?;
        self.event_bus.emit(GraphEvent::EdgeRemoved { source, target });
        self.dispatch_async(GraphEvent::EdgeRemoved { source, target });
        self.notify_live_queries();
        Ok(())
    }

    /// Establece una propiedad en un nodo existente.
    ///
    /// Emite `GraphEvent::PropertyChanged` al event bus.
    pub fn set_node_property(&self, node_id: NodeId, prop_name: &str, value: Value) -> BikoResult<()> {
        let prop_map = self.prop_map.read();
        let prop_id = *prop_map.get(prop_name)
            .ok_or_else(|| bikodb_core::error::BikoError::QueryParse(
                format!("Unknown property: {}", prop_name),
            ))?;

        // Get old value for the event
        let old_value = self.graph.get_node(node_id)
            .and_then(|n| n.properties.iter().find(|(k, _)| *k == prop_id).map(|(_, v)| v.clone()));

        self.graph.set_node_property(node_id, prop_id, value.clone())?;

        let event = GraphEvent::PropertyChanged {
            node_id,
            property_id: prop_id,
            old_value,
            new_value: value,
        };
        self.event_bus.emit(event.clone());
        self.dispatch_async(event);
        self.notify_live_queries();

        Ok(())
    }

    /// Elimina un nodo del grafo.
    ///
    /// Emite `GraphEvent::NodeRemoved` al event bus.
    pub fn remove_node(&self, node_id: NodeId) -> BikoResult<()> {
        self.graph.remove_node(node_id)?;
        let event = GraphEvent::NodeRemoved { node_id };
        self.event_bus.emit(event.clone());
        self.dispatch_async(event);
        self.notify_live_queries();
        Ok(())
    }

    /// Ejecuta un query SQL y retorna resultados.
    pub fn query_sql(&self, query: &str) -> BikoResult<Vec<Row>> {
        let type_map = self.type_map.read();
        let prop_map = self.prop_map.read();

        let plan = sql::parse_sql(query, &type_map, &prop_map)
            .map_err(|e| bikodb_core::error::BikoError::QueryParse(e.to_string()))?;

        self.monitor.record_query();
        pipeline::execute(&self.graph, plan)
    }

    /// Obtiene un nodo por ID.
    pub fn get_node(&self, id: NodeId) -> Option<bikodb_graph::graph::NodeData> {
        self.graph.get_node(id)
    }

    /// Vecinos de un nodo.
    pub fn neighbors(&self, id: NodeId, direction: Direction) -> BikoResult<Vec<NodeId>> {
        self.graph.neighbors(id, direction)
    }

    /// Número de nodos.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Número de edges.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Referencia al grafo subyacente (para operaciones avanzadas).
    pub fn graph(&self) -> &ConcurrentGraph {
        &self.graph
    }

    /// Referencia al monitor de recursos.
    pub fn monitor(&self) -> &ResourceMonitor {
        &self.monitor
    }

    /// Snapshot de prop_id → nombre (para serialización HTTP).
    pub fn prop_names_snapshot(&self) -> HashMap<u16, String> {
        self.prop_names.read().clone()
    }

    /// Registra un tipo de relación en el schema.
    pub fn register_relationship(&self, name: &str, type_id: TypeId) {
        self.rel_map.write().insert(name.to_string(), type_id);
    }

    /// Ejecuta un query Cypher y retorna resultados.
    pub fn query_cypher(&self, query: &str) -> BikoResult<Vec<Row>> {
        let label_map: LabelMap = self.type_map.read().clone();
        let prop_map = self.prop_map.read();
        let rel_map = self.rel_map.read();

        let plan = cypher::parse_cypher(query, &label_map, &prop_map, &rel_map)
            .map_err(|e| bikodb_core::error::BikoError::QueryParse(e.to_string()))?;

        self.monitor.record_query();
        pipeline::execute(&self.graph, plan)
    }

    /// Ejecuta un query Gremlin y retorna resultados.
    pub fn query_gremlin(&self, query: &str) -> BikoResult<Vec<Row>> {
        let label_map: gremlin::LabelMap = self.type_map.read().clone();
        let prop_map = self.prop_map.read();
        let rel_map = self.rel_map.read();

        let plan = gremlin::parse_gremlin(query, &label_map, &prop_map, &rel_map)
            .map_err(|e| bikodb_core::error::BikoError::QueryParse(e.to_string()))?;

        self.monitor.record_query();
        pipeline::execute(&self.graph, plan)
    }

    /// Unified query dispatcher. Accepts "sql", "cypher", or "gremlin" as language.
    pub fn query(&self, language: &str, query_str: &str) -> BikoResult<Vec<Row>> {
        match language.to_lowercase().as_str() {
            "sql" => self.query_sql(query_str),
            "cypher" => self.query_cypher(query_str),
            "gremlin" => self.query_gremlin(query_str),
            other => Err(bikodb_core::error::BikoError::QueryParse(
                format!("Unsupported query language: {}", other),
            )),
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // AI / ML Integration
    // ─────────────────────────────────────────────────────────────────────

    /// Referencia al event bus de grafos.
    pub fn event_bus(&self) -> &EventBus {
        &self.event_bus
    }

    /// Configura el índice HNSW para vector search.
    pub fn enable_hnsw(&mut self, dimensions: usize, metric: DistanceMetric) {
        let config = HnswConfig {
            dimensions,
            metric,
            max_connections: 16,
            max_connections_0: 32,
            ef_construction: 200,
        };
        self.hnsw_index = Some(Arc::new(HnswIndex::new(config)));
    }

    /// Referencia al HNSW index (si habilitado).
    pub fn hnsw_index(&self) -> Option<&HnswIndex> {
        self.hnsw_index.as_deref()
    }

    /// Inserta un vector en el HNSW index.
    pub fn insert_vector(&self, node_id: NodeId, vector: Vec<f32>) -> BikoResult<()> {
        match &self.hnsw_index {
            Some(idx) => {
                idx.insert(node_id, vector);
                Ok(())
            }
            None => Err(bikodb_core::error::BikoError::Unsupported(
                "HNSW index not enabled. Call enable_hnsw() first.".into(),
            )),
        }
    }

    /// Busca los k vectores más similares via HNSW.
    pub fn vector_search(&self, query: &[f32], k: usize) -> BikoResult<Vec<(NodeId, f32)>> {
        match &self.hnsw_index {
            Some(idx) => {
                let results = idx.search(query, k, k.max(10) * 2);
                Ok(results.into_iter().map(|r| (r.node_id, r.distance)).collect())
            }
            None => Err(bikodb_core::error::BikoError::Unsupported(
                "HNSW index not enabled.".into(),
            )),
        }
    }

    /// Habilita el pipeline de embeddings automáticos.
    ///
    /// El pipeline reacciona a eventos del grafo y auto-computa embeddings.
    /// Incluye property fetcher para re-computar en PropertyChanged.
    pub fn enable_embedding_pipeline(
        &mut self,
        feature_prop_ids: Vec<u16>,
        embedding_dim: usize,
    ) {
        let config = EmbeddingPipelineConfig {
            feature_prop_ids,
            embedding_dim,
            strategy: EmbeddingStrategy::FeatureBased,
            distance_metric: DistanceMetric::Cosine,
        };

        // Create property fetcher backed by the graph
        let graph_clone = self.graph.clone();
        let fetcher: PropertyFetcher = Arc::new(move |node_id: NodeId| {
            graph_clone.get_node(node_id).map(|n| n.properties)
        });

        let pipeline = Arc::new(
            EmbeddingPipeline::new(config).with_property_fetcher(fetcher)
        );

        // Register pipeline as event listener
        let pipeline_clone = pipeline.clone();
        self.event_bus.on_event(Box::new(move |event| {
            pipeline_clone.handle_event(event);
        }));

        self.embedding_pipeline = Some(pipeline);
    }

    /// Referencia al embedding pipeline (si habilitado).
    pub fn embedding_pipeline(&self) -> Option<&EmbeddingPipeline> {
        self.embedding_pipeline.as_deref()
    }

    /// Crea un nodo con propiedades y emite evento al event bus.
    ///
    /// Alias de `create_node()` — mantenido por compatibilidad.
    /// Desde v0.2, `create_node()` siempre emite eventos.
    pub fn create_node_with_events(&self, type_id: TypeId, props: Vec<(&str, Value)>) -> NodeId {
        self.create_node(type_id, props)
    }

    /// Crea edge y emite evento al event bus.
    ///
    /// Alias de `create_edge()` — mantenido por compatibilidad.
    /// Desde v0.2, `create_edge()` siempre emite eventos.
    pub fn create_edge_with_events(
        &self,
        source: NodeId,
        target: NodeId,
        type_id: TypeId,
    ) -> BikoResult<EdgeId> {
        self.create_edge(source, target, type_id)
    }

    /// Referencia al registro de modelos de inferencia.
    pub fn inference_registry(&self) -> &parking_lot::RwLock<InferenceRegistry> {
        &self.inference_registry
    }

    /// Registra un modelo de inferencia.
    pub fn register_inference_model(&self, model: bikodb_ai::inference::InferenceModel) {
        self.inference_registry.write().register(model);
    }

    /// Ejecuta inferencia en un nodo usando un modelo registrado.
    pub fn infer_node(&self, model_name: &str, node_id: NodeId) -> BikoResult<Vec<f32>> {
        let registry = self.inference_registry.read();
        let node_data = self.graph.get_node(node_id)
            .ok_or(bikodb_core::error::BikoError::NodeNotFound(node_id))?;

        registry.predict(model_name, &extract_features_from_node(&node_data))
            .ok_or_else(|| bikodb_core::error::BikoError::Unsupported(
                format!("Model '{}' not found", model_name),
            ))
    }

    // ─────────────────────────────────────────────────────────────────────
    // Semantic Search — Búsqueda semántica con resultados enriquecidos
    // ─────────────────────────────────────────────────────────────────────

    /// Búsqueda semántica unificada: busca por vector y retorna resultados
    /// enriquecidos con propiedades del nodo y vecinos.
    ///
    /// Usa HNSW si está habilitado, si no usa el embedding pipeline.
    pub fn semantic_search(&self, query: &[f32], k: usize) -> BikoResult<Vec<SemanticSearchResult>> {
        // Get raw results from HNSW or pipeline
        let raw_results: Vec<(NodeId, f32)> = if let Some(idx) = &self.hnsw_index {
            let results = idx.search(query, k, k.max(10) * 2);
            results.into_iter().map(|r| (r.node_id, r.distance)).collect()
        } else if let Some(pipeline) = &self.embedding_pipeline {
            pipeline.search_similar(query, k)
        } else {
            return Err(bikodb_core::error::BikoError::Unsupported(
                "No vector index or embedding pipeline enabled. Call enable_hnsw() or enable_embedding_pipeline() first.".into(),
            ));
        };

        Ok(self.enrich_results(raw_results))
    }

    /// Búsqueda semántica por nodo: genera embedding del nodo fuente y
    /// busca los k nodos más similares (excluye el nodo fuente).
    pub fn semantic_search_by_node(&self, node_id: NodeId, k: usize) -> BikoResult<Vec<SemanticSearchResult>> {
        let node = self.graph.get_node(node_id)
            .ok_or(bikodb_core::error::BikoError::NodeNotFound(node_id))?;

        // Generate embedding for this node
        let embedding = if let Some(pipeline) = &self.embedding_pipeline {
            pipeline.compute_feature_embedding(&node.properties)
        } else {
            extract_features_from_node(&node)
        };

        // Search k+1 because the source node might appear in results
        let mut results = self.semantic_search(&embedding, k + 1)?;
        results.retain(|r| r.node_id != node_id);
        results.truncate(k);
        Ok(results)
    }

    /// Enriquece resultados de búsqueda vectorial con propiedades y vecinos.
    fn enrich_results(&self, raw_results: Vec<(NodeId, f32)>) -> Vec<SemanticSearchResult> {
        let prop_names = self.prop_names.read();
        let mut enriched = Vec::with_capacity(raw_results.len());

        for (node_id, distance) in raw_results {
            let properties = self.graph.get_node(node_id)
                .map(|n| {
                    n.properties.into_iter()
                        .map(|(pid, val)| {
                            let name = prop_names.get(&pid)
                                .cloned()
                                .unwrap_or_else(|| format!("prop_{}", pid));
                            (name, val)
                        })
                        .collect()
                })
                .unwrap_or_default();

            let neighbors = self.graph
                .neighbors(node_id, Direction::Out)
                .unwrap_or_default();

            enriched.push(SemanticSearchResult {
                node_id,
                distance,
                properties,
                neighbors,
            });
        }

        enriched
    }

    // ─────────────────────────────────────────────────────────────────────
    // HNSW Persistence
    // ─────────────────────────────────────────────────────────────────────

    /// Serializa el índice HNSW a bytes para persistencia.
    pub fn save_hnsw(&self) -> BikoResult<Vec<u8>> {
        match &self.hnsw_index {
            Some(idx) => Ok(idx.serialize()),
            None => Err(bikodb_core::error::BikoError::Unsupported(
                "HNSW index not enabled.".into(),
            )),
        }
    }

    /// Carga un índice HNSW desde bytes.
    pub fn load_hnsw(&mut self, data: &[u8]) -> BikoResult<()> {
        let idx = HnswIndex::from_bytes(data)
            .map_err(|e| bikodb_core::error::BikoError::Unsupported(
                format!("Failed to deserialize HNSW index: {}", e),
            ))?;
        self.hnsw_index = Some(Arc::new(idx));
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // Incremental Inference — Re-inferencia automática en mutaciones
    // ─────────────────────────────────────────────────────────────────────

    /// Habilita el motor de inferencia incremental.
    ///
    /// Re-ejecuta automáticamente inferencia en nodos afectados por
    /// mutaciones del grafo, cacheando resultados.
    pub fn enable_incremental_inference(
        &mut self,
        model: AiInferenceModel,
        feature_prop_ids: Vec<u16>,
    ) {
        let inc = Arc::new(IncrementalInference::new(
            Arc::new(model),
            feature_prop_ids,
        ));

        // Register as event listener
        let inc_clone = inc.clone();
        self.event_bus.on_event(Box::new(move |event| {
            inc_clone.handle_event(event);
        }));

        self.incremental_inference = Some(inc);
    }

    /// Referencia al motor de inferencia incremental (si habilitado).
    pub fn incremental_inference(&self) -> Option<&IncrementalInference> {
        self.incremental_inference.as_deref()
    }

    /// Ejecuta inferencia incremental en todos los nodos dirty (afectados por mutaciones).
    pub fn flush_inference(&self) -> Vec<bikodb_ai::incremental::IncrementalResult> {
        match &self.incremental_inference {
            Some(inc) => {
                let graph = self.graph.clone();
                inc.flush_dirty(move |node_id| {
                    graph.get_node(node_id).map(|n| n.properties)
                })
            }
            None => Vec::new(),
        }
    }

    /// Obtiene predicción cacheada para un nodo.
    pub fn cached_prediction(&self, node_id: NodeId) -> Option<Vec<f32>> {
        self.incremental_inference
            .as_ref()
            .and_then(|inc| inc.cached_prediction(node_id))
    }

    // ─────────────────────────────────────────────────────────────────────
    // Live Query Subscriptions — Consultas reactivas en tiempo real
    // ─────────────────────────────────────────────────────────────────────

    /// Suscribe un live query: se re-ejecuta cuando el grafo cambia.
    ///
    /// El callback recibe los resultados actualizados cada vez que una
    /// mutación del grafo podría afectar los resultados.
    ///
    /// Retorna el índice de la suscripción para poder cancelarla.
    pub fn subscribe_query(
        &self,
        language: &str,
        query_str: &str,
        callback: LiveQueryCallback,
    ) -> usize {
        let entry = Arc::new(LiveQueryEntry {
            language: language.to_string(),
            query: query_str.to_string(),
            callback,
            last_result_len: parking_lot::Mutex::new(0),
        });

        // Execute the initial query and deliver results
        if let Ok(results) = self.query(language, query_str) {
            *entry.last_result_len.lock() = results.len();
            (entry.callback)(&results);
        }

        let mut subs = self.live_queries.write();
        let idx = subs.len();
        subs.push(entry);
        idx
    }

    /// Cancela una suscripción de live query.
    pub fn unsubscribe_query(&self, index: usize) {
        let mut subs = self.live_queries.write();
        if index < subs.len() {
            subs.remove(index);
        }
    }

    /// Número de live queries activas.
    pub fn live_query_count(&self) -> usize {
        self.live_queries.read().len()
    }

    /// Notifica a todos los live queries re-ejecutando sus consultas.
    fn notify_live_queries(&self) {
        let subs = self.live_queries.read();
        for entry in subs.iter() {
            if let Ok(results) = self.query(&entry.language, &entry.query) {
                let new_len = results.len();
                let mut last_len = entry.last_result_len.lock();
                if new_len != *last_len {
                    *last_len = new_len;
                    (entry.callback)(&results);
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Async Event Dispatch — Canal asíncrono para eventos
    // ─────────────────────────────────────────────────────────────────────

    /// Envía un evento al canal asíncrono (non-blocking).
    fn dispatch_async(&self, event: GraphEvent) {
        if let Some(tx) = &self.async_event_tx {
            let _ = tx.send(event);
        }
    }

    /// Drena todos los eventos pendientes del canal asíncrono.
    ///
    /// Retorna los eventos acumulados desde la última llamada.
    /// Útil para procesamiento batch o polling.
    pub fn drain_async_events(&self) -> Vec<GraphEvent> {
        let mut events = Vec::new();
        if let Some(rx) = &self.async_event_rx {
            let rx = rx.lock();
            while let Ok(event) = rx.try_recv() {
                events.push(event);
            }
        }
        events
    }

    /// ¿Hay eventos pendientes en el canal asíncrono?
    pub fn has_pending_events(&self) -> bool {
        // Peek by trying a non-blocking receive
        if let Some(rx) = &self.async_event_rx {
            let rx = rx.lock();
            // Use try_recv to check without consuming
            // Unfortunately, mpsc doesn't support peek, so we check indirectly
            // by attempting to receive. Since we can't put it back, we return true
            // as a heuristic. For accurate check, use drain_async_events().
            drop(rx);
        }
        // A lightweight check: if async_event_tx is active, events may be pending
        self.async_event_tx.is_some()
    }

    // ── Cluster / Sharding ──────────────────────────────────────────────

    /// Habilita el modo cluster (sharding + routing + cluster manager).
    ///
    /// Crea un `ShardedGraph` con la estrategia dada, un `ClusterManager`
    /// con `num_nodes` nodos virtuales, y un `QueryRouter` que los conecta.
    pub fn enable_cluster(&mut self, strategy: ShardStrategy, num_nodes: u16) {
        let sharded = Arc::new(ShardedGraph::new(strategy.clone()));
        let cluster = Arc::new(ClusterManager::new());
        for i in 0..num_nodes {
            cluster.add_node(ClusterNodeId(i));
        }
        // Assign shards to cluster nodes.
        let shard_ids: Vec<_> = (0..strategy.num_shards())
            .map(|i| bikodb_cluster::shard::ShardId(i))
            .collect();
        cluster.assign_shards(&shard_ids);
        let router = Arc::new(QueryRouter::new(sharded, cluster));
        self.cluster_router = Some(router);
    }

    /// Devuelve el `QueryRouter` del cluster, si está habilitado.
    pub fn cluster_router(&self) -> Option<&QueryRouter> {
        self.cluster_router.as_deref()
    }

    /// Inserta un nodo en el cluster (ruta al shard correcto).
    pub fn cluster_insert_node(&self, node_id: NodeId, type_id: TypeId, props: Vec<(u16, Value)>) -> BikoResult<()> {
        let router = self.cluster_router.as_ref().ok_or_else(|| {
            BikoError::Cluster("Cluster not enabled".into())
        })?;
        router.insert_node(node_id, type_id, props);
        Ok(())
    }

    /// Obtiene un nodo desde el cluster.
    pub fn cluster_get_node(&self, node_id: NodeId) -> BikoResult<Option<bikodb_graph::graph::NodeData>> {
        let router = self.cluster_router.as_ref().ok_or_else(|| {
            BikoError::Cluster("Cluster not enabled".into())
        })?;
        Ok(router.get_node(node_id))
    }

    /// Busca todos los nodos de un tipo en el cluster (scatter-gather).
    pub fn cluster_find_by_type(&self, type_id: TypeId) -> BikoResult<Vec<bikodb_graph::graph::NodeData>> {
        let router = self.cluster_router.as_ref().ok_or_else(|| {
            BikoError::Cluster("Cluster not enabled".into())
        })?;
        Ok(router.find_nodes_by_type(type_id))
    }

    /// Cuenta nodos de un tipo en el cluster.
    pub fn cluster_count_by_type(&self, type_id: TypeId) -> BikoResult<usize> {
        let router = self.cluster_router.as_ref().ok_or_else(|| {
            BikoError::Cluster("Cluster not enabled".into())
        })?;
        Ok(router.count_nodes_by_type(type_id))
    }

    // ── Document Store (Multi-modelo) ───────────────────────────────────

    /// Accede al document store subyacente.
    pub fn doc_store(&self) -> &bikodb_graph::document::DocumentStore {
        &self.doc_store
    }

    /// Obtiene (o crea) una colección de documentos por nombre.
    pub fn doc_collection(&self, name: &str) -> Arc<bikodb_graph::document::DocumentCollection> {
        self.doc_store.collection(name)
    }

    /// Inserta un documento en una colección.
    pub fn create_document(
        &self,
        collection: &str,
        fields: HashMap<String, Value>,
    ) -> bikodb_graph::document::DocumentId {
        let col = self.doc_store.collection(collection);
        col.insert(fields)
    }

    /// Inserta un documento vinculado a un nodo del grafo (graph↔document link).
    pub fn create_document_linked(
        &self,
        collection: &str,
        fields: HashMap<String, Value>,
        node_id: NodeId,
    ) -> bikodb_graph::document::DocumentId {
        let col = self.doc_store.collection(collection);
        col.insert_linked(fields, node_id)
    }

    /// Obtiene un documento por ID.
    pub fn get_document(
        &self,
        collection: &str,
        doc_id: bikodb_graph::document::DocumentId,
    ) -> Option<bikodb_graph::document::Document> {
        let col = self.doc_store.collection(collection);
        col.get(doc_id)
    }

    /// Actualiza campos de un documento (merge).
    pub fn update_document(
        &self,
        collection: &str,
        doc_id: bikodb_graph::document::DocumentId,
        updates: HashMap<String, Value>,
    ) -> bool {
        let col = self.doc_store.collection(collection);
        col.update(doc_id, updates)
    }

    /// Actualiza un campo anidado con dot-path.
    pub fn set_document_field(
        &self,
        collection: &str,
        doc_id: bikodb_graph::document::DocumentId,
        path: &str,
        value: Value,
    ) -> bool {
        let col = self.doc_store.collection(collection);
        col.set_field(doc_id, path, value)
    }

    /// Elimina un documento.
    pub fn delete_document(
        &self,
        collection: &str,
        doc_id: bikodb_graph::document::DocumentId,
    ) -> Option<bikodb_graph::document::Document> {
        let col = self.doc_store.collection(collection);
        col.remove(doc_id)
    }

    /// Busca documentos con un filtro (soporta campos anidados).
    pub fn query_documents(
        &self,
        collection: &str,
        filter: &bikodb_graph::document::DocFilter,
    ) -> Vec<bikodb_graph::document::Document> {
        let col = self.doc_store.collection(collection);
        col.find(filter)
    }

    /// Proyección parcial de documentos filtrados.
    pub fn query_documents_project(
        &self,
        collection: &str,
        filter: &bikodb_graph::document::DocFilter,
        fields: &[&str],
    ) -> Vec<HashMap<String, Value>> {
        let col = self.doc_store.collection(collection);
        col.find_project(filter, fields)
    }

    // ── Cross-model: Document ↔ Vector ──────────────────────────────────

    /// Inserta un documento con un embedding, indexándolo en HNSW si está activo.
    ///
    /// Crea un nodo en el grafo (para que tenga NodeId), inserta el vector en HNSW,
    /// y vincula el documento al nodo.
    pub fn create_document_with_vector(
        &self,
        collection: &str,
        fields: HashMap<String, Value>,
        embedding: Vec<f32>,
    ) -> BikoResult<(bikodb_graph::document::DocumentId, NodeId)> {
        // 1. Create a graph node to anchor the vector
        let node_id = self.graph.insert_node(TypeId(0));
        // 2. Insert embedding in HNSW if available
        if let Some(ref hnsw) = self.hnsw_index {
            hnsw.insert(node_id, embedding);
        }
        // 3. Create linked document
        let col = self.doc_store.collection(collection);
        let doc_id = col.insert_linked(fields, node_id);
        Ok((doc_id, node_id))
    }

    /// Búsqueda semántica sobre documentos: vector search → enrich con documento.
    ///
    /// Combina los tres modelos: vector (HNSW) → grafo (NodeId) → documento.
    pub fn semantic_search_documents(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> BikoResult<Vec<(bikodb_graph::document::Document, f32)>> {
        let hnsw = self.hnsw_index.as_ref().ok_or_else(|| {
            BikoError::IndexNotFound("HNSW index not enabled".into())
        })?;
        let neighbors = hnsw.search(query, k, k * 2);

        let col = self.doc_store.collection(collection);
        let mut results = Vec::new();
        for r in &neighbors {
            // Find documents linked to this node
            for doc in col.iter_all() {
                if doc.linked_node == Some(r.node_id) {
                    results.push((doc, r.distance));
                    break;
                }
            }
        }
        Ok(results)
    }

    // ── Cross-model: Graph ↔ Document ───────────────────────────────────

    /// Crea un nodo en el grafo y un documento vinculado en una sola operación.
    pub fn create_node_with_document(
        &self,
        type_id: TypeId,
        node_props: Vec<(&str, Value)>,
        collection: &str,
        doc_fields: HashMap<String, Value>,
    ) -> (NodeId, bikodb_graph::document::DocumentId) {
        let node_id = self.create_node(type_id, node_props);
        let col = self.doc_store.collection(collection);
        let doc_id = col.insert_linked(doc_fields, node_id);
        (node_id, doc_id)
    }

    /// Busca documentos cuyos nodos vinculados son vecinos de un nodo dado.
    pub fn neighbor_documents(
        &self,
        node_id: NodeId,
        collection: &str,
        direction: Direction,
    ) -> BikoResult<Vec<bikodb_graph::document::Document>> {
        let neighbor_ids = self.graph.neighbors(node_id, direction)?;
        let col = self.doc_store.collection(collection);
        let all_docs = col.iter_all();
        let results: Vec<_> = all_docs
            .into_iter()
            .filter(|doc| {
                doc.linked_node.map(|n| neighbor_ids.contains(&n)).unwrap_or(false)
            })
            .collect();
        Ok(results)
    }

    // ── Plugin System ───────────────────────────────────────────────────

    /// Referencia al plugin manager.
    pub fn plugin_manager(&self) -> &PluginManager {
        &self.plugin_manager
    }

    /// Registra un plugin con lifecycle management.
    pub fn register_plugin(&self, plugin: Arc<dyn Plugin>) -> BikoResult<()> {
        let types: Vec<String> = self.type_map.read().keys().cloned().collect();
        let ctx = PluginManagerContext {
            node_count: self.node_count(),
            edge_count: self.edge_count(),
            types,
        };
        self.plugin_manager.register_plugin(plugin, &ctx)
    }

    /// Desregistra un plugin.
    pub fn unregister_plugin(&self, name: &str) -> BikoResult<()> {
        self.plugin_manager.unregister_plugin(name)
    }

    /// Registra un hook handler para un punto de intercepción.
    pub fn register_hook(&self, hook_point: HookPoint, handler: Arc<dyn HookHandler>) {
        self.plugin_manager.register_hook(hook_point, handler);
    }

    /// Registra una UDF (User Defined Function).
    pub fn register_udf(&self, udf: Arc<dyn UserDefinedFn>) -> BikoResult<()> {
        self.plugin_manager.register_udf(udf)
    }

    /// Ejecuta una UDF por nombre.
    pub fn call_udf(&self, name: &str, args: &[Value]) -> BikoResult<UdfReturn> {
        self.plugin_manager.call_udf(name, args)
    }

    /// Registra un algoritmo de grafos custom.
    pub fn register_algorithm(&self, alg: Arc<dyn GraphAlgorithmExt>) -> BikoResult<()> {
        self.plugin_manager.register_algorithm(alg)
    }

    /// Ejecuta un algoritmo de grafos custom.
    ///
    /// Construye el AlgorithmInput a partir del estado actual del grafo.
    pub fn run_algorithm(&self, name: &str, params: HashMap<String, Value>) -> BikoResult<AlgorithmResult> {
        let mut nodes: Vec<(u64, TypeId)> = Vec::new();
        self.graph.iter_nodes(|nid, nd| {
            nodes.push((nid.0, nd.type_id));
        });

        let mut edges: Vec<(u64, u64)> = Vec::new();
        self.graph.iter_edges(|_eid, ed| {
            edges.push((ed.source.0, ed.target.0));
        });

        let input = AlgorithmInput { nodes, edges, params };
        self.plugin_manager.run_algorithm(name, &input)
    }

    /// Registra una métrica de distancia custom.
    pub fn register_distance_fn(&self, dist: Arc<dyn DistanceFnExt>) -> BikoResult<()> {
        self.plugin_manager.register_distance_fn(dist)
    }

    /// Calcula distancia usando una métrica custom.
    pub fn custom_distance(&self, name: &str, a: &[f32], b: &[f32]) -> BikoResult<f32> {
        self.plugin_manager.custom_distance(name, a, b)
    }

    /// Registra un proveedor de inferencia custom.
    pub fn register_inference_provider(&self, provider: Arc<dyn InferenceProviderExt>) -> BikoResult<()> {
        self.plugin_manager.register_inference_provider(provider)
    }

    /// Ejecuta inferencia con un proveedor custom.
    pub fn custom_predict(&self, name: &str, features: &[f32]) -> BikoResult<Vec<f32>> {
        self.plugin_manager.custom_predict(name, features)
    }

    /// Crea un nodo con hooks pre/post. Retorna error si un hook lo veta.
    pub fn create_node_hooked(&self, type_id: TypeId, props: Vec<(&str, Value)>) -> BikoResult<NodeId> {
        // Dispatch PreInsertNode hook
        let pre_ctx = HookContext::for_node(HookPoint::PreInsertNode, NodeId(0), type_id);
        if let HookResult::Abort(reason) = self.plugin_manager.dispatch_hook(&pre_ctx) {
            return Err(BikoError::Generic(format!("Hook vetoed create_node: {}", reason)));
        }

        let node_id = self.create_node(type_id, props);

        // Dispatch PostInsertNode hook
        let post_ctx = HookContext::for_node(HookPoint::PostInsertNode, node_id, type_id);
        self.plugin_manager.dispatch_hook(&post_ctx);

        Ok(node_id)
    }

    /// Crea un edge con hooks pre/post. Retorna error si un hook lo veta.
    pub fn create_edge_hooked(&self, source: NodeId, target: NodeId, type_id: TypeId) -> BikoResult<EdgeId> {
        let pre_ctx = HookContext::for_edge(HookPoint::PreInsertEdge, EdgeId(0), type_id);
        if let HookResult::Abort(reason) = self.plugin_manager.dispatch_hook(&pre_ctx) {
            return Err(BikoError::Generic(format!("Hook vetoed create_edge: {}", reason)));
        }

        let edge_id = self.create_edge(source, target, type_id)?;

        let post_ctx = HookContext::for_edge(HookPoint::PostInsertEdge, edge_id, type_id);
        self.plugin_manager.dispatch_hook(&post_ctx);

        Ok(edge_id)
    }

    /// Elimina un nodo con hooks pre/post. Retorna error si un hook lo veta.
    pub fn remove_node_hooked(&self, node_id: NodeId) -> BikoResult<()> {
        let type_id = self.graph.get_node(node_id)
            .map(|n| n.type_id)
            .unwrap_or(TypeId(0));
        let pre_ctx = HookContext::for_node(HookPoint::PreDeleteNode, node_id, type_id);
        if let HookResult::Abort(reason) = self.plugin_manager.dispatch_hook(&pre_ctx) {
            return Err(BikoError::Generic(format!("Hook vetoed remove_node: {}", reason)));
        }

        self.remove_node(node_id)?;

        let post_ctx = HookContext::for_node(HookPoint::PostDeleteNode, node_id, type_id);
        self.plugin_manager.dispatch_hook(&post_ctx);

        Ok(())
    }

    /// Ejecuta un query SQL con hooks pre/post.
    pub fn query_sql_hooked(&self, query: &str) -> BikoResult<Vec<Row>> {
        let pre_ctx = HookContext::for_query(HookPoint::PreQuery, query);
        if let HookResult::Abort(reason) = self.plugin_manager.dispatch_hook(&pre_ctx) {
            return Err(BikoError::Generic(format!("Hook vetoed query: {}", reason)));
        }

        let results = self.query_sql(query)?;

        let post_ctx = HookContext::for_query(HookPoint::PostQuery, query);
        self.plugin_manager.dispatch_hook(&post_ctx);

        Ok(results)
    }
}

/// Extrae features numéricas de un NodeData para inferencia.
fn extract_features_from_node(node: &bikodb_graph::graph::NodeData) -> Vec<f32> {
    let mut features = Vec::new();
    for (_, val) in &node.properties {
        match val {
            Value::Int(v) => features.push(*v as f32),
            Value::Float(v) => features.push(*v as f32),
            Value::Bool(v) => features.push(if *v { 1.0 } else { 0.0 }),
            Value::Embedding(emb) => features.extend_from_slice(emb),
            _ => features.push(0.0),
        }
    }
    features
}

impl Default for Database {
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

    fn setup_db() -> Database {
        let db = Database::new();
        db.register_type("Person", TypeId(1));
        db.register_property("name", 0);
        db.register_property("age", 1);
        db
    }

    #[test]
    fn test_create_and_query() {
        let db = setup_db();

        db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(30)),
        ]);
        db.create_node(TypeId(1), vec![
            ("name", Value::from("Bob")),
            ("age", Value::Int(25)),
        ]);

        let results = db.query_sql("SELECT * FROM Person").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_with_filter() {
        let db = setup_db();

        db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(30)),
        ]);
        db.create_node(TypeId(1), vec![
            ("name", Value::from("Bob")),
            ("age", Value::Int(20)),
        ]);

        let results = db.query_sql("SELECT * FROM Person WHERE age > 25").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_create_edge_and_neighbors() {
        let db = setup_db();
        let a = db.create_node(TypeId(1), vec![("name", Value::from("Alice"))]);
        let b = db.create_node(TypeId(1), vec![("name", Value::from("Bob"))]);

        db.create_edge(a, b, TypeId(10)).unwrap();

        let neighbors = db.neighbors(a, Direction::Out).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], b);
    }

    #[test]
    fn test_node_count() {
        let db = setup_db();
        assert_eq!(db.node_count(), 0);

        db.create_node(TypeId(1), vec![]);
        assert_eq!(db.node_count(), 1);
    }

    #[test]
    fn test_query_cypher() {
        let db = setup_db();
        db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(30)),
        ]);
        db.create_node(TypeId(1), vec![
            ("name", Value::from("Bob")),
            ("age", Value::Int(25)),
        ]);

        let results = db.query_cypher("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_cypher_with_where() {
        let db = setup_db();
        db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(30)),
        ]);
        db.create_node(TypeId(1), vec![
            ("name", Value::from("Bob")),
            ("age", Value::Int(20)),
        ]);

        let results = db.query_cypher("MATCH (n:Person) WHERE n.age > 25 RETURN n").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_gremlin() {
        let db = setup_db();
        db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(30)),
        ]);

        let results = db.query_gremlin("g.V().hasLabel('Person')").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_unified_query_sql() {
        let db = setup_db();
        db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
        ]);

        let results = db.query("sql", "SELECT * FROM Person").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_unified_query_cypher() {
        let db = setup_db();
        db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
        ]);

        let results = db.query("cypher", "MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_unified_query_gremlin() {
        let db = setup_db();
        db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
        ]);

        let results = db.query("gremlin", "g.V().hasLabel('Person')").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_unified_query_unsupported_language() {
        let db = setup_db();
        let result = db.query("sparql", "SELECT * WHERE {}");
        assert!(result.is_err());
    }

    #[test]
    fn test_register_relationship() {
        let db = setup_db();
        db.register_relationship("KNOWS", TypeId(10));
        // Just verify it doesn't panic — relationship registered.
    }

    // ─────────────────────────────────────────────────────────────────
    // AI / ML Integration Tests
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_event_bus_on_create_node() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let db = setup_db();
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        db.event_bus().on_event(Box::new(move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        }));

        db.create_node_with_events(TypeId(1), vec![("name", Value::from("Alice"))]);
        db.create_node_with_events(TypeId(1), vec![("name", Value::from("Bob"))]);

        assert_eq!(counter.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_event_bus_on_create_edge() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let db = setup_db();
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        db.event_bus().on_event(Box::new(move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        }));

        let a = db.create_node_with_events(TypeId(1), vec![]);
        let b = db.create_node_with_events(TypeId(1), vec![]);
        db.create_edge_with_events(a, b, TypeId(10)).unwrap();

        assert_eq!(counter.load(Ordering::Relaxed), 3); // 2 nodes + 1 edge
    }

    #[test]
    fn test_enable_hnsw() {
        let mut db = setup_db();
        db.enable_hnsw(4, DistanceMetric::Euclidean);
        assert!(db.hnsw_index().is_some());
    }

    #[test]
    fn test_vector_search_via_hnsw() {
        let mut db = setup_db();
        db.enable_hnsw(3, DistanceMetric::Euclidean);

        let a = db.create_node(TypeId(1), vec![("name", Value::from("Alice"))]);
        let b = db.create_node(TypeId(1), vec![("name", Value::from("Bob"))]);

        db.insert_vector(a, vec![1.0, 0.0, 0.0]).unwrap();
        db.insert_vector(b, vec![0.0, 1.0, 0.0]).unwrap();

        let results = db.vector_search(&[0.9, 0.1, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, a); // Alice is closer
    }

    #[test]
    fn test_vector_search_without_hnsw_errors() {
        let db = setup_db();
        assert!(db.vector_search(&[1.0], 1).is_err());
        assert!(db.insert_vector(NodeId(1), vec![1.0]).is_err());
    }

    #[test]
    fn test_embedding_pipeline_auto_compute() {
        let mut db = setup_db();
        db.enable_embedding_pipeline(vec![1], 4); // age as feature

        // Create node with events → embedding should auto-compute
        db.create_node_with_events(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(30)),
        ]);

        let pipeline = db.embedding_pipeline().unwrap();
        assert_eq!(pipeline.computed_count(), 1);
        assert_eq!(pipeline.vector_index().len(), 1);
    }

    #[test]
    fn test_embedding_pipeline_similarity_search() {
        let mut db = setup_db();
        db.enable_embedding_pipeline(vec![1], 4); // age-based

        db.create_node_with_events(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(30)),
        ]);
        db.create_node_with_events(TypeId(1), vec![
            ("name", Value::from("Bob")),
            ("age", Value::Int(25)),
        ]);

        let pipeline = db.embedding_pipeline().unwrap();
        let results = pipeline.search_similar(&[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_register_and_use_inference_model() {
        use bikodb_ai::gnn::Activation;
        use bikodb_ai::inference::{InferenceModel, InferenceTask, MlpLayer, MlpModel};

        let db = setup_db();

        // Register a simple model: sum(features)
        let mut layer = MlpLayer::new(2, 1, Activation::None);
        layer.weights = vec![1.0, 1.0]; // sum features
        let mlp = MlpModel::new(vec![layer]);
        let model = InferenceModel::new("sum".into(), InferenceTask::Scoring, mlp, vec![0, 1]);
        db.register_inference_model(model);

        // Create a node and run inference
        let node = db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(30)),
        ]);

        let prediction = db.infer_node("sum", node).unwrap();
        // name → 0.0 (string → 0), age → 30.0 → sum = 30.0
        assert!((prediction[0] - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_infer_nonexistent_model() {
        let db = setup_db();
        let node = db.create_node(TypeId(1), vec![("age", Value::Int(10))]);
        let result = db.infer_node("no_such_model", node);
        assert!(result.is_err());
    }

    #[test]
    fn test_infer_nonexistent_node() {
        let db = setup_db();
        let result = db.infer_node("any", NodeId(999));
        assert!(result.is_err());
    }

    // ─────────────────────────────────────────────────────────────────
    // Transparent Events Tests
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_create_node_emits_event_by_default() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let db = setup_db();
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        db.event_bus().on_event(Box::new(move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        }));

        // Regular create_node should now emit events
        db.create_node(TypeId(1), vec![("name", Value::from("Alice"))]);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_create_edge_emits_event_by_default() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let db = setup_db();
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        db.event_bus().on_event(Box::new(move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        }));

        let a = db.create_node(TypeId(1), vec![]);
        let b = db.create_node(TypeId(1), vec![]);
        db.create_edge(a, b, TypeId(10)).unwrap();

        // 2 node events + 1 edge event = 3
        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_set_node_property() {
        let db = setup_db();
        let node = db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(25)),
        ]);

        db.set_node_property(node, "age", Value::Int(30)).unwrap();

        let data = db.get_node(node).unwrap();
        let age_val = data.properties.iter().find(|(k, _)| *k == 1).map(|(_, v)| v);
        assert_eq!(age_val, Some(&Value::Int(30)));
    }

    #[test]
    fn test_set_node_property_emits_event() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let db = setup_db();
        let property_changed_count = Arc::new(AtomicUsize::new(0));
        let c = property_changed_count.clone();

        db.event_bus().on_event(Box::new(move |event| {
            if matches!(event, GraphEvent::PropertyChanged { .. }) {
                c.fetch_add(1, Ordering::Relaxed);
            }
        }));

        let node = db.create_node(TypeId(1), vec![("age", Value::Int(25))]);
        db.set_node_property(node, "age", Value::Int(30)).unwrap();

        assert_eq!(property_changed_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_remove_node() {
        let db = setup_db();
        let node = db.create_node(TypeId(1), vec![("name", Value::from("Alice"))]);
        assert_eq!(db.node_count(), 1);

        db.remove_node(node).unwrap();
        assert_eq!(db.node_count(), 0);
    }

    #[test]
    fn test_remove_node_emits_event() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let db = setup_db();
        let remove_count = Arc::new(AtomicUsize::new(0));
        let c = remove_count.clone();

        db.event_bus().on_event(Box::new(move |event| {
            if matches!(event, GraphEvent::NodeRemoved { .. }) {
                c.fetch_add(1, Ordering::Relaxed);
            }
        }));

        let node = db.create_node(TypeId(1), vec![]);
        db.remove_node(node).unwrap();

        assert_eq!(remove_count.load(Ordering::Relaxed), 1);
    }

    // ─────────────────────────────────────────────────────────────────
    // PropertyChanged Pipeline Recomputation Tests
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_property_change_recomputes_embedding() {
        let mut db = setup_db();
        db.enable_embedding_pipeline(vec![1], 4); // age as feature

        let node = db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(30)),
        ]);

        let pipeline = db.embedding_pipeline().unwrap();
        let count_after_create = pipeline.computed_count();
        assert!(count_after_create >= 1);

        // Change the age property → should trigger recomputation
        db.set_node_property(node, "age", Value::Int(99)).unwrap();

        // Embedding should have been recomputed
        assert!(pipeline.computed_count() > count_after_create);
    }

    // ─────────────────────────────────────────────────────────────────
    // Semantic Search Tests
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_semantic_search_via_hnsw() {
        let mut db = setup_db();
        db.enable_hnsw(3, DistanceMetric::Euclidean);

        let a = db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(30)),
        ]);
        let b = db.create_node(TypeId(1), vec![
            ("name", Value::from("Bob")),
            ("age", Value::Int(25)),
        ]);

        db.insert_vector(a, vec![1.0, 0.0, 0.0]).unwrap();
        db.insert_vector(b, vec![0.0, 1.0, 0.0]).unwrap();

        let results = db.semantic_search(&[0.9, 0.1, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].node_id, a);

        // Results should have properties
        assert!(!results[0].properties.is_empty());
        let name = results[0].properties.iter().find(|(k, _)| k == "name");
        assert!(name.is_some());
    }

    #[test]
    fn test_semantic_search_via_pipeline() {
        let mut db = setup_db();
        db.enable_embedding_pipeline(vec![1], 4); // age-based

        db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(30)),
        ]);
        db.create_node(TypeId(1), vec![
            ("name", Value::from("Bob")),
            ("age", Value::Int(25)),
        ]);

        let results = db.semantic_search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        // Results should have properties
        assert!(!results[0].properties.is_empty());
    }

    #[test]
    fn test_semantic_search_no_index_errors() {
        let db = setup_db();
        let result = db.semantic_search(&[1.0], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_semantic_search_by_node() {
        let mut db = setup_db();
        db.enable_embedding_pipeline(vec![1], 4);

        let alice = db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(30)),
        ]);
        db.create_node(TypeId(1), vec![
            ("name", Value::from("Bob")),
            ("age", Value::Int(28)),
        ]);
        db.create_node(TypeId(1), vec![
            ("name", Value::from("Charlie")),
            ("age", Value::Int(5)),
        ]);

        // Find nodes similar to Alice (should exclude Alice herself)
        let results = db.semantic_search_by_node(alice, 2).unwrap();
        assert!(results.iter().all(|r| r.node_id != alice));
        assert!(!results.is_empty());
    }

    #[test]
    fn test_semantic_search_enriched_with_neighbors() {
        let mut db = setup_db();
        db.enable_hnsw(3, DistanceMetric::Euclidean);
        db.register_relationship("KNOWS", TypeId(10));

        let a = db.create_node(TypeId(1), vec![("name", Value::from("Alice"))]);
        let b = db.create_node(TypeId(1), vec![("name", Value::from("Bob"))]);
        db.create_edge(a, b, TypeId(10)).unwrap();

        db.insert_vector(a, vec![1.0, 0.0, 0.0]).unwrap();
        db.insert_vector(b, vec![0.0, 1.0, 0.0]).unwrap();

        let results = db.semantic_search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].node_id, a);
        // Alice has Bob as neighbor
        assert_eq!(results[0].neighbors.len(), 1);
        assert_eq!(results[0].neighbors[0], b);
    }

    // ─────────────────────────────────────────────────────────────────
    // HNSW Persistence Tests
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_save_and_load_hnsw() {
        let mut db = setup_db();
        db.enable_hnsw(3, DistanceMetric::Euclidean);

        let a = db.create_node(TypeId(1), vec![("name", Value::from("Alice"))]);
        let b = db.create_node(TypeId(1), vec![("name", Value::from("Bob"))]);
        db.insert_vector(a, vec![1.0, 0.0, 0.0]).unwrap();
        db.insert_vector(b, vec![0.0, 1.0, 0.0]).unwrap();

        // Save
        let data = db.save_hnsw().unwrap();
        assert!(!data.is_empty());

        // Load into new db
        let mut db2 = setup_db();
        db2.load_hnsw(&data).unwrap();

        // Should be searchable
        let results = db2.vector_search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, a);
    }

    #[test]
    fn test_save_hnsw_without_index_errors() {
        let db = setup_db();
        assert!(db.save_hnsw().is_err());
    }

    // ─────────────────────────────────────────────────────────────────
    // remove_edge Tests
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_edge() {
        let db = setup_db();
        let a = db.create_node(TypeId(1), vec![("name", Value::from("Alice"))]);
        let b = db.create_node(TypeId(1), vec![("name", Value::from("Bob"))]);
        let edge_id = db.create_edge(a, b, TypeId(10)).unwrap();

        assert_eq!(db.edge_count(), 1);
        db.remove_edge(edge_id).unwrap();
        assert_eq!(db.edge_count(), 0);
    }

    #[test]
    fn test_remove_edge_emits_event() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let db = setup_db();
        let edge_remove_count = Arc::new(AtomicUsize::new(0));
        let c = edge_remove_count.clone();

        db.event_bus().on_event(Box::new(move |event| {
            if matches!(event, GraphEvent::EdgeRemoved { .. }) {
                c.fetch_add(1, Ordering::Relaxed);
            }
        }));

        let a = db.create_node(TypeId(1), vec![]);
        let b = db.create_node(TypeId(1), vec![]);
        let edge_id = db.create_edge(a, b, TypeId(10)).unwrap();
        db.remove_edge(edge_id).unwrap();

        assert_eq!(edge_remove_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_remove_nonexistent_edge_errors() {
        let db = setup_db();
        let result = db.remove_edge(EdgeId(999));
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_edge_updates_neighbors() {
        let db = setup_db();
        let a = db.create_node(TypeId(1), vec![("name", Value::from("Alice"))]);
        let b = db.create_node(TypeId(1), vec![("name", Value::from("Bob"))]);
        let edge_id = db.create_edge(a, b, TypeId(10)).unwrap();

        let neighbors = db.neighbors(a, Direction::Out).unwrap();
        assert_eq!(neighbors.len(), 1);

        db.remove_edge(edge_id).unwrap();

        let neighbors = db.neighbors(a, Direction::Out).unwrap();
        assert_eq!(neighbors.len(), 0);
    }

    // ─────────────────────────────────────────────────────────────────
    // Incremental Inference Tests
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_enable_incremental_inference() {
        use bikodb_ai::gnn::Activation;
        use bikodb_ai::inference::{InferenceModel, InferenceTask, MlpLayer, MlpModel};

        let mut db = setup_db();

        let mut layer = MlpLayer::new(2, 1, Activation::None);
        layer.weights = vec![1.0, 1.0];
        let mlp = MlpModel::new(vec![layer]);
        let model = InferenceModel::new("inc_sum".into(), InferenceTask::Scoring, mlp, vec![0, 1]);

        db.enable_incremental_inference(model, vec![0, 1]);
        assert!(db.incremental_inference().is_some());
    }

    #[test]
    fn test_incremental_inference_marks_dirty_on_mutation() {
        use bikodb_ai::gnn::Activation;
        use bikodb_ai::inference::{InferenceModel, InferenceTask, MlpLayer, MlpModel};

        let mut db = setup_db();

        let mut layer = MlpLayer::new(2, 1, Activation::None);
        layer.weights = vec![1.0, 1.0];
        let mlp = MlpModel::new(vec![layer]);
        let model = InferenceModel::new("inc_sum".into(), InferenceTask::Scoring, mlp, vec![0, 1]);

        db.enable_incremental_inference(model, vec![0, 1]);

        // Create a node → should mark it dirty
        let node = db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(30)),
        ]);

        let inc = db.incremental_inference().unwrap();
        assert!(inc.has_dirty_nodes());
        assert!(inc.dirty_count() >= 1);

        // Flush → should run inference and cache result
        let results = db.flush_inference();
        assert!(!results.is_empty());
        assert_eq!(results[0].node_id, node);

        // Should be cached now
        assert!(db.cached_prediction(node).is_some());
    }

    #[test]
    fn test_incremental_inference_re_infers_on_property_change() {
        use bikodb_ai::gnn::Activation;
        use bikodb_ai::inference::{InferenceModel, InferenceTask, MlpLayer, MlpModel};

        let mut db = setup_db();

        let mut layer = MlpLayer::new(2, 1, Activation::None);
        layer.weights = vec![1.0, 1.0];
        let mlp = MlpModel::new(vec![layer]);
        let model = InferenceModel::new("inc_sum".into(), InferenceTask::Scoring, mlp, vec![0, 1]);

        db.enable_incremental_inference(model, vec![0, 1]);

        let node = db.create_node(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(30)),
        ]);

        // Flush initial
        db.flush_inference();
        let pred1 = db.cached_prediction(node).unwrap();

        // Change property → should mark dirty again
        db.set_node_property(node, "age", Value::Int(99)).unwrap();
        let inc = db.incremental_inference().unwrap();
        assert!(inc.has_dirty_nodes());

        // Flush again → new prediction
        let results = db.flush_inference();
        assert!(!results.is_empty());
        let pred2 = db.cached_prediction(node).unwrap();

        // Predictions should differ (age changed from 30 to 99)
        assert_ne!(pred1, pred2);
    }

    // ─────────────────────────────────────────────────────────────────
    // Live Query Subscription Tests
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_subscribe_query_initial_delivery() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let db = setup_db();
        db.create_node(TypeId(1), vec![("name", Value::from("Alice")), ("age", Value::Int(30))]);
        db.create_node(TypeId(1), vec![("name", Value::from("Bob")), ("age", Value::Int(25))]);

        let call_count = Arc::new(AtomicUsize::new(0));
        let last_len = Arc::new(AtomicUsize::new(0));
        let cc = call_count.clone();
        let ll = last_len.clone();

        db.subscribe_query("sql", "SELECT * FROM Person", Arc::new(move |results| {
            cc.fetch_add(1, Ordering::Relaxed);
            ll.store(results.len(), Ordering::Relaxed);
        }));

        // Initial delivery should have happened
        assert_eq!(call_count.load(Ordering::Relaxed), 1);
        assert_eq!(last_len.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_subscribe_query_reacts_to_mutations() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let db = setup_db();

        let call_count = Arc::new(AtomicUsize::new(0));
        let last_len = Arc::new(AtomicUsize::new(0));
        let cc = call_count.clone();
        let ll = last_len.clone();

        db.subscribe_query("sql", "SELECT * FROM Person", Arc::new(move |results| {
            cc.fetch_add(1, Ordering::Relaxed);
            ll.store(results.len(), Ordering::Relaxed);
        }));

        // Initial: 0 results
        assert_eq!(call_count.load(Ordering::Relaxed), 1);
        assert_eq!(last_len.load(Ordering::Relaxed), 0);

        // Add a node → live query should re-execute
        db.create_node(TypeId(1), vec![("name", Value::from("Alice")), ("age", Value::Int(30))]);

        // Callback should have fired again with 1 result
        assert!(call_count.load(Ordering::Relaxed) >= 2);
        assert_eq!(last_len.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_unsubscribe_query() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let db = setup_db();

        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = call_count.clone();

        let idx = db.subscribe_query("sql", "SELECT * FROM Person", Arc::new(move |_| {
            cc.fetch_add(1, Ordering::Relaxed);
        }));

        assert_eq!(db.live_query_count(), 1);
        db.unsubscribe_query(idx);
        assert_eq!(db.live_query_count(), 0);

        let before = call_count.load(Ordering::Relaxed);
        db.create_node(TypeId(1), vec![("name", Value::from("Alice"))]);
        assert_eq!(call_count.load(Ordering::Relaxed), before);
    }

    #[test]
    fn test_live_query_with_cypher() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let db = setup_db();
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = call_count.clone();

        db.subscribe_query("cypher", "MATCH (n:Person) RETURN n", Arc::new(move |_| {
            cc.fetch_add(1, Ordering::Relaxed);
        }));

        db.create_node(TypeId(1), vec![("name", Value::from("Alice"))]);
        assert!(call_count.load(Ordering::Relaxed) >= 2); // initial + mutation
    }

    // ─────────────────────────────────────────────────────────────────
    // Async Event Dispatch Tests
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_async_event_channel() {
        let db = setup_db();

        db.create_node(TypeId(1), vec![("name", Value::from("Alice"))]);
        db.create_node(TypeId(1), vec![("name", Value::from("Bob"))]);

        let events = db.drain_async_events();
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], GraphEvent::NodeAdded { .. }));
        assert!(matches!(events[1], GraphEvent::NodeAdded { .. }));
    }

    #[test]
    fn test_async_events_include_all_mutation_types() {
        let db = setup_db();

        let a = db.create_node(TypeId(1), vec![("age", Value::Int(10))]);
        let b = db.create_node(TypeId(1), vec![("age", Value::Int(20))]);
        let edge_id = db.create_edge(a, b, TypeId(10)).unwrap();
        db.set_node_property(a, "age", Value::Int(99)).unwrap();
        db.remove_edge(edge_id).unwrap();
        db.remove_node(b).unwrap();

        let events = db.drain_async_events();
        // NodeAdded, NodeAdded, EdgeAdded, PropertyChanged, EdgeRemoved, NodeRemoved
        assert_eq!(events.len(), 6);
        assert!(matches!(events[0], GraphEvent::NodeAdded { .. }));
        assert!(matches!(events[1], GraphEvent::NodeAdded { .. }));
        assert!(matches!(events[2], GraphEvent::EdgeAdded { .. }));
        assert!(matches!(events[3], GraphEvent::PropertyChanged { .. }));
        assert!(matches!(events[4], GraphEvent::EdgeRemoved { .. }));
        assert!(matches!(events[5], GraphEvent::NodeRemoved { .. }));
    }

    #[test]
    fn test_drain_async_events_empties_channel() {
        let db = setup_db();
        db.create_node(TypeId(1), vec![]);

        let events1 = db.drain_async_events();
        assert_eq!(events1.len(), 1);

        // Second drain should be empty
        let events2 = db.drain_async_events();
        assert_eq!(events2.len(), 0);
    }

    // ── Cluster Tests ───────────────────────────────────────────────────

    #[test]
    fn test_enable_cluster_hash_based() {
        let mut db = setup_db();
        db.enable_cluster(ShardStrategy::HashBased { num_shards: 4 }, 2);
        assert!(db.cluster_router().is_some());
    }

    #[test]
    fn test_cluster_insert_and_get_node() {
        let mut db = setup_db();
        db.enable_cluster(ShardStrategy::HashBased { num_shards: 4 }, 2);
        db.cluster_insert_node(NodeId(100), TypeId(1), vec![(0, Value::string("Alice"))]).unwrap();
        let node = db.cluster_get_node(NodeId(100)).unwrap();
        assert!(node.is_some());
        let n = node.unwrap();
        assert_eq!(n.type_id, TypeId(1));
    }

    #[test]
    fn test_cluster_scatter_gather_by_type() {
        let mut db = setup_db();
        db.enable_cluster(ShardStrategy::HashBased { num_shards: 3 }, 2);
        for i in 0..10 {
            db.cluster_insert_node(NodeId(i), TypeId(1), vec![]).unwrap();
        }
        let results = db.cluster_find_by_type(TypeId(1)).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_cluster_count_by_type() {
        let mut db = setup_db();
        db.enable_cluster(ShardStrategy::HashBased { num_shards: 2 }, 2);
        for i in 0..5 {
            db.cluster_insert_node(NodeId(i), TypeId(1), vec![]).unwrap();
        }
        db.cluster_insert_node(NodeId(99), TypeId(2), vec![]).unwrap();
        assert_eq!(db.cluster_count_by_type(TypeId(1)).unwrap(), 5);
        assert_eq!(db.cluster_count_by_type(TypeId(2)).unwrap(), 1);
    }

    #[test]
    fn test_cluster_not_enabled_error() {
        let db = setup_db();
        let res = db.cluster_insert_node(NodeId(0), TypeId(1), vec![]);
        assert!(res.is_err());
    }

    #[test]
    fn test_cluster_range_strategy() {
        let mut db = setup_db();
        db.enable_cluster(ShardStrategy::RangeBased { boundaries: vec![50, 100] }, 3);
        db.cluster_insert_node(NodeId(10), TypeId(1), vec![]).unwrap();
        db.cluster_insert_node(NodeId(60), TypeId(1), vec![]).unwrap();
        db.cluster_insert_node(NodeId(110), TypeId(1), vec![]).unwrap();
        assert_eq!(db.cluster_find_by_type(TypeId(1)).unwrap().len(), 3);
    }

    // ── Document Store Tests ────────────────────────────────────────────

    use bikodb_graph::document::{DocFilter, DocumentId};

    fn sample_doc_fields() -> HashMap<String, Value> {
        let mut addr = HashMap::new();
        addr.insert("city".into(), Value::string("NYC"));
        addr.insert("zip".into(), Value::Int(10001));

        let mut f = HashMap::new();
        f.insert("name".into(), Value::string("Alice"));
        f.insert("age".into(), Value::Int(30));
        f.insert("address".into(), Value::Map(Box::new(addr)));
        f.insert("tags".into(), Value::List(Box::new(vec![
            Value::string("rust"),
            Value::string("graph"),
        ])));
        f
    }

    #[test]
    fn test_create_and_get_document() {
        let db = setup_db();
        let doc_id = db.create_document("users", sample_doc_fields());
        let doc = db.get_document("users", doc_id).unwrap();
        assert_eq!(doc.get_field("name"), Some(&Value::string("Alice")));
    }

    #[test]
    fn test_update_document() {
        let db = setup_db();
        let doc_id = db.create_document("users", sample_doc_fields());
        let mut updates = HashMap::new();
        updates.insert("age".into(), Value::Int(31));
        assert!(db.update_document("users", doc_id, updates));
        let doc = db.get_document("users", doc_id).unwrap();
        assert_eq!(doc.get_field("age"), Some(&Value::Int(31)));
    }

    #[test]
    fn test_set_document_nested_field() {
        let db = setup_db();
        let doc_id = db.create_document("users", sample_doc_fields());
        assert!(db.set_document_field("users", doc_id, "address.city", Value::string("LA")));
        let doc = db.get_document("users", doc_id).unwrap();
        assert_eq!(doc.get_field("address.city"), Some(&Value::string("LA")));
    }

    #[test]
    fn test_delete_document() {
        let db = setup_db();
        let doc_id = db.create_document("users", sample_doc_fields());
        assert!(db.delete_document("users", doc_id).is_some());
        assert!(db.get_document("users", doc_id).is_none());
    }

    #[test]
    fn test_query_documents_flat() {
        let db = setup_db();
        db.create_document("users", sample_doc_fields());
        let mut f2 = HashMap::new();
        f2.insert("name".into(), Value::string("Bob"));
        f2.insert("age".into(), Value::Int(25));
        db.create_document("users", f2);

        let results = db.query_documents("users", &DocFilter::Eq {
            path: "name".into(),
            value: Value::string("Alice"),
        });
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_documents_nested_filter() {
        let db = setup_db();
        db.create_document("users", sample_doc_fields());

        let results = db.query_documents("users", &DocFilter::Eq {
            path: "address.city".into(),
            value: Value::string("NYC"),
        });
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_documents_projection() {
        let db = setup_db();
        db.create_document("users", sample_doc_fields());

        let results = db.query_documents_project(
            "users",
            &DocFilter::Eq { path: "name".into(), value: Value::string("Alice") },
            &["name", "address.city"],
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("name"), Some(&Value::string("Alice")));
        assert_eq!(results[0].get("address.city"), Some(&Value::string("NYC")));
        assert!(results[0].get("age").is_none());
    }

    #[test]
    fn test_doc_collection_names() {
        let db = setup_db();
        db.create_document("users", sample_doc_fields());
        db.create_document("products", HashMap::new());
        let names = db.doc_store().collection_names();
        assert!(names.contains(&"users".to_string()));
        assert!(names.contains(&"products".to_string()));
    }

    // ── Cross-model: Graph ↔ Document ───────────────────────────────────

    #[test]
    fn test_create_node_with_document() {
        let db = setup_db();
        let (node_id, doc_id) = db.create_node_with_document(
            TypeId(1),
            vec![("name", Value::string("Alice"))],
            "user_profiles",
            sample_doc_fields(),
        );
        // Graph node exists
        let node = db.get_node(node_id).unwrap();
        assert_eq!(node.type_id, TypeId(1));
        // Document exists and is linked
        let doc = db.get_document("user_profiles", doc_id).unwrap();
        assert_eq!(doc.linked_node, Some(node_id));
        assert_eq!(doc.get_field("name"), Some(&Value::string("Alice")));
    }

    #[test]
    fn test_create_document_linked() {
        let db = setup_db();
        let node_id = db.create_node(TypeId(1), vec![("name", Value::string("Bob"))]);
        let doc_id = db.create_document_linked("profiles", sample_doc_fields(), node_id);
        let doc = db.get_document("profiles", doc_id).unwrap();
        assert_eq!(doc.linked_node, Some(node_id));
    }

    #[test]
    fn test_neighbor_documents() {
        let db = setup_db();
        // Create Alice node + doc
        let (alice_nid, _) = db.create_node_with_document(
            TypeId(1),
            vec![("name", Value::string("Alice"))],
            "profiles",
            { let mut f = HashMap::new(); f.insert("bio".into(), Value::string("Alice bio")); f },
        );
        // Create Bob node + doc
        let (bob_nid, _) = db.create_node_with_document(
            TypeId(1),
            vec![("name", Value::string("Bob"))],
            "profiles",
            { let mut f = HashMap::new(); f.insert("bio".into(), Value::string("Bob bio")); f },
        );
        // Edge: Alice → Bob
        db.create_edge(alice_nid, bob_nid, TypeId(10)).unwrap();

        // Find documents of Alice's OUT neighbors
        let docs = db.neighbor_documents(alice_nid, "profiles", Direction::Out).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].get_field("bio"), Some(&Value::string("Bob bio")));
    }

    // ── Cross-model: Document ↔ Vector ──────────────────────────────────

    #[test]
    fn test_create_document_with_vector() {
        let mut db = setup_db();
        db.enable_hnsw(3, bikodb_ai::embedding::DistanceMetric::Cosine);

        let mut fields = HashMap::new();
        fields.insert("title".into(), Value::string("Rust Book"));

        let (doc_id, node_id) = db.create_document_with_vector(
            "articles",
            fields,
            vec![1.0, 0.0, 0.0],
        ).unwrap();

        // Document exists and is linked
        let doc = db.get_document("articles", doc_id).unwrap();
        assert_eq!(doc.linked_node, Some(node_id));
        assert_eq!(doc.get_field("title"), Some(&Value::string("Rust Book")));

        // Vector was indexed
        let results = db.vector_search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, node_id);
    }

    #[test]
    fn test_semantic_search_documents() {
        let mut db = setup_db();
        db.enable_hnsw(3, bikodb_ai::embedding::DistanceMetric::Cosine);

        let mut f1 = HashMap::new();
        f1.insert("title".into(), Value::string("Rust Book"));
        db.create_document_with_vector("articles", f1, vec![1.0, 0.0, 0.0]).unwrap();

        let mut f2 = HashMap::new();
        f2.insert("title".into(), Value::string("Python Book"));
        db.create_document_with_vector("articles", f2, vec![0.0, 1.0, 0.0]).unwrap();

        // Search near the Rust Book vector
        let results = db.semantic_search_documents("articles", &[0.9, 0.1, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.get_field("title"), Some(&Value::string("Rust Book")));
    }

    // ── Cross-model: Graph ↔ Document ↔ Vector ─────────────────────────

    #[test]
    fn test_full_multimodel_workflow() {
        let mut db = setup_db();
        db.enable_hnsw(3, bikodb_ai::embedding::DistanceMetric::Cosine);

        // Create 2 articles with graph nodes, documents, and embeddings
        let mut f1 = HashMap::new();
        f1.insert("title".into(), Value::string("Graph Databases"));
        f1.insert("category".into(), Value::string("tech"));
        let (doc_id1, node1) = db.create_document_with_vector(
            "articles", f1, vec![1.0, 0.5, 0.0],
        ).unwrap();

        let mut f2 = HashMap::new();
        f2.insert("title".into(), Value::string("Machine Learning"));
        f2.insert("category".into(), Value::string("tech"));
        let (doc_id2, node2) = db.create_document_with_vector(
            "articles", f2, vec![0.0, 0.5, 1.0],
        ).unwrap();

        // Edge: article1 → article2 (related)
        db.create_edge(node1, node2, TypeId(10)).unwrap();

        // 1. Graph query: neighbors of article1
        let neighbors = db.neighbors(node1, Direction::Out).unwrap();
        assert_eq!(neighbors, vec![node2]);

        // 2. Document query: filter by category
        let tech_docs = db.query_documents("articles", &DocFilter::Eq {
            path: "category".into(),
            value: Value::string("tech"),
        });
        assert_eq!(tech_docs.len(), 2);

        // 3. Vector search: find similar to article1
        let similar = db.semantic_search_documents("articles", &[0.9, 0.6, 0.1], 1).unwrap();
        assert_eq!(similar.len(), 1);
        assert_eq!(similar[0].0.get_field("title"), Some(&Value::string("Graph Databases")));

        // 4. Cross-model: neighbor documents
        let neighbor_docs = db.neighbor_documents(node1, "articles", Direction::Out).unwrap();
        assert_eq!(neighbor_docs.len(), 1);
        assert_eq!(neighbor_docs[0].get_field("title"), Some(&Value::string("Machine Learning")));
    }

    // ═════════════════════════════════════════════════════════════════════
    // Plugin System Integration Tests
    // ═════════════════════════════════════════════════════════════════════

    use bikodb_core::plugin::{
        AlgorithmInput, AlgorithmResult, DistanceFnExt, GraphAlgorithmExt, HookContext,
        HookHandler, HookPoint, HookResult, InferenceProviderExt, Plugin, PluginContext,
        UdfReturn, UserDefinedFn,
    };

    struct TestPlugin { name: String }
    impl Plugin for TestPlugin {
        fn name(&self) -> &str { &self.name }
        fn version(&self) -> &str { "1.0.0" }
        fn init(&self, _ctx: &dyn PluginContext) -> BikoResult<()> { Ok(()) }
        fn shutdown(&self) -> BikoResult<()> { Ok(()) }
        fn as_any(&self) -> &dyn std::any::Any { self }
    }

    #[test]
    fn test_database_register_plugin() {
        let db = Database::new();
        db.register_type("Person", TypeId(1));
        db.create_node(TypeId(1), vec![]);

        let p: Arc<dyn Plugin> = Arc::new(TestPlugin { name: "test-plugin".into() });
        db.register_plugin(p).unwrap();
        assert_eq!(db.plugin_manager().plugin_count(), 1);

        let infos = db.plugin_manager().list_plugins();
        assert_eq!(infos[0].name, "test-plugin");
    }

    #[test]
    fn test_database_unregister_plugin() {
        let db = Database::new();
        let p: Arc<dyn Plugin> = Arc::new(TestPlugin { name: "temp".into() });
        db.register_plugin(p).unwrap();
        db.unregister_plugin("temp").unwrap();
        assert_eq!(db.plugin_manager().plugin_count(), 0);
    }

    // ── UDF Tests through Database ──────────────────────────────────────

    struct UpperUdf;
    impl UserDefinedFn for UpperUdf {
        fn name(&self) -> &str { "UPPER" }
        fn call(&self, args: &[Value]) -> UdfReturn {
            match args.first() {
                Some(Value::String(s)) => UdfReturn::Scalar(Value::string(s.to_uppercase())),
                _ => UdfReturn::Scalar(Value::Null),
            }
        }
        fn param_count(&self) -> usize { 1 }
    }

    struct SumUdf;
    impl UserDefinedFn for SumUdf {
        fn name(&self) -> &str { "SUM_VALUES" }
        fn call(&self, args: &[Value]) -> UdfReturn {
            let total: i64 = args.iter().filter_map(|v| match v {
                Value::Int(n) => Some(*n),
                _ => None,
            }).sum();
            UdfReturn::Scalar(Value::Int(total))
        }
    }

    #[test]
    fn test_database_udf_register_and_call() {
        let db = Database::new();
        db.register_udf(Arc::new(UpperUdf)).unwrap();
        db.register_udf(Arc::new(SumUdf)).unwrap();

        match db.call_udf("UPPER", &[Value::string("hello")]).unwrap() {
            UdfReturn::Scalar(Value::String(s)) => assert_eq!(s, "HELLO"),
            _ => panic!("Expected HELLO"),
        }

        match db.call_udf("SUM_VALUES", &[Value::Int(10), Value::Int(20), Value::Int(30)]).unwrap() {
            UdfReturn::Scalar(Value::Int(n)) => assert_eq!(n, 60),
            _ => panic!("Expected 60"),
        }
    }

    #[test]
    fn test_database_udf_not_found() {
        let db = Database::new();
        assert!(db.call_udf("NOPE", &[]).is_err());
    }

    // ── Hook Tests through Database ─────────────────────────────────────

    struct AllowHook;
    impl HookHandler for AllowHook {
        fn name(&self) -> &str { "allow" }
        fn handle(&self, _ctx: &HookContext) -> HookResult { HookResult::Continue }
    }

    struct VetoHook;
    impl HookHandler for VetoHook {
        fn name(&self) -> &str { "veto" }
        fn handle(&self, _ctx: &HookContext) -> HookResult {
            HookResult::Abort("operation denied".into())
        }
    }

    #[test]
    fn test_database_hook_allows_create_node() {
        let db = Database::new();
        db.register_type("Person", TypeId(1));
        db.register_property("name", 0);
        db.register_hook(HookPoint::PreInsertNode, Arc::new(AllowHook));
        db.register_hook(HookPoint::PostInsertNode, Arc::new(AllowHook));

        let node_id = db.create_node_hooked(TypeId(1), vec![("name", Value::from("Alice"))]).unwrap();
        assert!(db.get_node(node_id).is_some());
    }

    #[test]
    fn test_database_hook_vetoes_create_node() {
        let db = Database::new();
        db.register_type("Person", TypeId(1));
        db.register_hook(HookPoint::PreInsertNode, Arc::new(VetoHook));

        let result = db.create_node_hooked(TypeId(1), vec![]);
        assert!(result.is_err());
        assert_eq!(db.node_count(), 0); // node was NOT created
    }

    #[test]
    fn test_database_hook_vetoes_create_edge() {
        let db = Database::new();
        db.register_type("Person", TypeId(1));
        let a = db.create_node(TypeId(1), vec![]);
        let b = db.create_node(TypeId(1), vec![]);
        db.register_hook(HookPoint::PreInsertEdge, Arc::new(VetoHook));

        let result = db.create_edge_hooked(a, b, TypeId(10));
        assert!(result.is_err());
        assert_eq!(db.edge_count(), 0);
    }

    #[test]
    fn test_database_hook_vetoes_remove_node() {
        let db = Database::new();
        db.register_type("Person", TypeId(1));
        let a = db.create_node(TypeId(1), vec![]);
        db.register_hook(HookPoint::PreDeleteNode, Arc::new(VetoHook));

        let result = db.remove_node_hooked(a);
        assert!(result.is_err());
        assert_eq!(db.node_count(), 1); // node still exists
    }

    #[test]
    fn test_database_hook_vetoes_query() {
        let db = Database::new();
        db.register_type("Person", TypeId(1));
        db.register_hook(HookPoint::PreQuery, Arc::new(VetoHook));

        let result = db.query_sql_hooked("SELECT * FROM Person");
        assert!(result.is_err());
    }

    // ── Graph Algorithm Plugin Tests ────────────────────────────────────

    struct DegreeAlg;
    impl GraphAlgorithmExt for DegreeAlg {
        fn name(&self) -> &str { "degree" }
        fn execute(&self, input: &AlgorithmInput) -> BikoResult<AlgorithmResult> {
            let mut scores = std::collections::HashMap::new();
            for &(src, dst) in &input.edges {
                *scores.entry(src).or_insert(0.0) += 1.0;
                *scores.entry(dst).or_insert(0.0) += 1.0;
            }
            Ok(AlgorithmResult { node_scores: scores, metadata: std::collections::HashMap::new() })
        }
    }

    #[test]
    fn test_database_custom_algorithm() {
        let db = Database::new();
        db.register_type("Person", TypeId(1));
        db.register_property("name", 0);

        let a = db.create_node(TypeId(1), vec![("name", Value::from("A"))]);
        let b = db.create_node(TypeId(1), vec![("name", Value::from("B"))]);
        let c = db.create_node(TypeId(1), vec![("name", Value::from("C"))]);
        db.create_edge(a, b, TypeId(10)).unwrap();
        db.create_edge(a, c, TypeId(10)).unwrap();
        db.create_edge(b, c, TypeId(10)).unwrap();

        db.register_algorithm(Arc::new(DegreeAlg)).unwrap();
        let result = db.run_algorithm("degree", HashMap::new()).unwrap();

        // A: edges to B and C = degree 2
        assert_eq!(*result.node_scores.get(&a.0).unwrap(), 2.0);
        // B: edge from A and to C = degree 2
        assert_eq!(*result.node_scores.get(&b.0).unwrap(), 2.0);
        // C: edges from A and B = degree 2
        assert_eq!(*result.node_scores.get(&c.0).unwrap(), 2.0);
    }

    // ── Custom Distance Function Tests ──────────────────────────────────

    struct ManhattanDist;
    impl DistanceFnExt for ManhattanDist {
        fn name(&self) -> &str { "manhattan" }
        fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
        }
    }

    #[test]
    fn test_database_custom_distance() {
        let db = Database::new();
        db.register_distance_fn(Arc::new(ManhattanDist)).unwrap();
        let d = db.custom_distance("manhattan", &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        assert!((d - 9.0).abs() < 1e-5);
    }

    // ── Custom Inference Provider Tests ─────────────────────────────────

    struct ScaleProvider;
    impl InferenceProviderExt for ScaleProvider {
        fn name(&self) -> &str { "scale" }
        fn predict(&self, features: &[f32]) -> Vec<f32> {
            features.iter().map(|x| x * 10.0).collect()
        }
        fn input_dim(&self) -> usize { 3 }
        fn output_dim(&self) -> usize { 3 }
    }

    #[test]
    fn test_database_custom_inference() {
        let db = Database::new();
        db.register_inference_provider(Arc::new(ScaleProvider)).unwrap();
        let out = db.custom_predict("scale", &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(out, vec![10.0, 20.0, 30.0]);
    }

    // ── Full Plugin Integration Test ────────────────────────────────────

    #[test]
    fn test_database_full_plugin_ecosystem() {
        let db = Database::new();
        db.register_type("Person", TypeId(1));
        db.register_property("name", 0);
        db.register_property("age", 1);

        // 1. Register a plugin
        let p: Arc<dyn Plugin> = Arc::new(TestPlugin { name: "ecosystem".into() });
        db.register_plugin(p).unwrap();

        // 2. Register hooks
        db.register_hook(HookPoint::PostInsertNode, Arc::new(AllowHook));

        // 3. Register UDFs
        db.register_udf(Arc::new(UpperUdf)).unwrap();
        db.register_udf(Arc::new(SumUdf)).unwrap();

        // 4. Register algorithm
        db.register_algorithm(Arc::new(DegreeAlg)).unwrap();

        // 5. Register distance fn
        db.register_distance_fn(Arc::new(ManhattanDist)).unwrap();

        // 6. Register inference provider
        db.register_inference_provider(Arc::new(ScaleProvider)).unwrap();

        // Create some data
        let a = db.create_node_hooked(TypeId(1), vec![
            ("name", Value::from("Alice")),
            ("age", Value::Int(30)),
        ]).unwrap();
        let b = db.create_node_hooked(TypeId(1), vec![
            ("name", Value::from("Bob")),
            ("age", Value::Int(25)),
        ]).unwrap();
        db.create_edge_hooked(a, b, TypeId(10)).unwrap();

        // Use UDFs
        match db.call_udf("UPPER", &[Value::string("alice")]).unwrap() {
            UdfReturn::Scalar(Value::String(s)) => assert_eq!(s, "ALICE"),
            _ => panic!("expected ALICE"),
        }
        match db.call_udf("SUM_VALUES", &[Value::Int(30), Value::Int(25)]).unwrap() {
            UdfReturn::Scalar(Value::Int(n)) => assert_eq!(n, 55),
            _ => panic!("expected 55"),
        }

        // Use algorithm
        let result = db.run_algorithm("degree", HashMap::new()).unwrap();
        assert_eq!(*result.node_scores.get(&a.0).unwrap(), 1.0);
        assert_eq!(*result.node_scores.get(&b.0).unwrap(), 1.0);

        // Use distance fn
        let d = db.custom_distance("manhattan", &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert!((d - 4.0).abs() < 1e-5);

        // Use inference
        let out = db.custom_predict("scale", &[0.5, 1.5]).unwrap();
        assert_eq!(out, vec![5.0, 15.0]);

        // Plugin still active
        assert_eq!(db.plugin_manager().plugin_count(), 1);
    }
}
