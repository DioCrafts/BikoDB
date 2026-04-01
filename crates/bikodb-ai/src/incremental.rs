// =============================================================================
// bikodb-ai::incremental — Inferencia incremental y embeddings en tiempo real
// =============================================================================
// Sistema de eventos y hooks que dispara re-inferencia y actualización de
// embeddings cuando el grafo muta.
//
// ## Diseño
// - Event bus: cola de GraphEvent (NodeAdded, EdgeAdded, PropertyChanged, etc.)
// - Listeners registrables: callbacks que reaccionan a eventos
// - EmbeddingPipeline: auto-computa embeddings cuando nodos cambian
// - IncrementalInference: re-ejecuta inferencia solo en nodos afectados
//
// ## Inspiración
// - Neo4j Triggers / APOC triggers
// - Qdrant: auto-indexed vectors on insert
// =============================================================================

use crate::embedding::DistanceMetric;
use crate::gnn::{self, GnnModel};
use crate::inference::InferenceModel;
use crate::vector_idx::VectorIndex;
use bikodb_core::types::NodeId;
use bikodb_core::value::Value;
use parking_lot::RwLock;
use std::sync::Arc;

/// Callback para obtener todas las propiedades de un nodo del grafo.
///
/// El pipeline de embeddings lo usa para re-computar embeddings cuando
/// cambian propiedades individuales (necesita todas las props, no solo la nueva).
pub type PropertyFetcher = Arc<dyn Fn(NodeId) -> Option<Vec<(u16, Value)>> + Send + Sync>;

/// Evento del grafo que puede disparar re-inferencia.
#[derive(Debug, Clone)]
pub enum GraphEvent {
    /// Nodo añadido con sus propiedades.
    NodeAdded {
        node_id: NodeId,
        properties: Vec<(u16, Value)>,
    },
    /// Nodo eliminado.
    NodeRemoved {
        node_id: NodeId,
    },
    /// Edge añadido.
    EdgeAdded {
        source: NodeId,
        target: NodeId,
    },
    /// Edge eliminado.
    EdgeRemoved {
        source: NodeId,
        target: NodeId,
    },
    /// Propiedad de nodo cambiada.
    PropertyChanged {
        node_id: NodeId,
        property_id: u16,
        old_value: Option<Value>,
        new_value: Value,
    },
}

impl GraphEvent {
    /// Nodos afectados por este evento.
    pub fn affected_nodes(&self) -> Vec<NodeId> {
        match self {
            GraphEvent::NodeAdded { node_id, .. } => vec![*node_id],
            GraphEvent::NodeRemoved { node_id } => vec![*node_id],
            GraphEvent::EdgeAdded { source, target } => vec![*source, *target],
            GraphEvent::EdgeRemoved { source, target } => vec![*source, *target],
            GraphEvent::PropertyChanged { node_id, .. } => vec![*node_id],
        }
    }
}

/// Callback type para listeners de eventos.
pub type EventCallback = Box<dyn Fn(&GraphEvent) + Send + Sync>;

/// Bus de eventos del grafo.
///
/// Registra listeners y despacha eventos cuando el grafo muta.
///
/// # Ejemplo
/// ```
/// use bikodb_ai::incremental::{EventBus, GraphEvent};
/// use bikodb_core::types::NodeId;
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use std::sync::Arc;
///
/// let bus = EventBus::new();
/// let counter = Arc::new(AtomicUsize::new(0));
/// let c = counter.clone();
/// bus.on_event(Box::new(move |_e| { c.fetch_add(1, Ordering::Relaxed); }));
///
/// bus.emit(GraphEvent::NodeAdded {
///     node_id: NodeId(1),
///     properties: vec![],
/// });
/// assert_eq!(counter.load(Ordering::Relaxed), 1);
/// ```
pub struct EventBus {
    listeners: RwLock<Vec<EventCallback>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            listeners: RwLock::new(Vec::new()),
        }
    }

    /// Registra un listener.
    pub fn on_event(&self, callback: EventCallback) {
        self.listeners.write().push(callback);
    }

    /// Emite un evento a todos los listeners.
    pub fn emit(&self, event: GraphEvent) {
        let listeners = self.listeners.read();
        for cb in listeners.iter() {
            cb(&event);
        }
    }

    /// Número de listeners registrados.
    pub fn listener_count(&self) -> usize {
        self.listeners.read().len()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EmbeddingPipeline — Embeddings actualizables en tiempo real
// ─────────────────────────────────────────────────────────────────────────────

/// Estrategia de generación de embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingStrategy {
    /// Embeddings basados en features de propiedades (flat vector de numeric props).
    FeatureBased,
    /// Embeddings generados por GNN (message passing sobre vecindario).
    GnnBased,
}

/// Configuración del pipeline de embeddings.
pub struct EmbeddingPipelineConfig {
    /// IDs de propiedades a usar como features.
    pub feature_prop_ids: Vec<u16>,
    /// Dimensión de los embeddings resultantes.
    pub embedding_dim: usize,
    /// Estrategia de generación.
    pub strategy: EmbeddingStrategy,
    /// Métrica de distancia para el vector index.
    pub distance_metric: DistanceMetric,
}

/// Pipeline de embeddings que auto-computa y actualiza embeddings.
///
/// Se conecta al EventBus para reaccionar a cambios en el grafo.
pub struct EmbeddingPipeline {
    config: EmbeddingPipelineConfig,
    /// Índice vectorial para búsqueda k-NN.
    vector_index: Arc<VectorIndex>,
    /// Modelo GNN opcional (para estrategia GnnBased).
    gnn_model: Option<Arc<GnnModel>>,
    /// Callback para obtener propiedades completas de un nodo.
    property_fetcher: Option<PropertyFetcher>,
    /// Contador de embeddings computados.
    computed_count: RwLock<usize>,
}

impl EmbeddingPipeline {
    /// Crea un pipeline de embeddings.
    pub fn new(config: EmbeddingPipelineConfig) -> Self {
        let dim = config.embedding_dim;
        let metric = config.distance_metric;
        Self {
            config,
            vector_index: Arc::new(VectorIndex::new(dim, metric)),
            gnn_model: None,
            property_fetcher: None,
            computed_count: RwLock::new(0),
        }
    }

    /// Crea pipeline con modelo GNN.
    pub fn with_gnn(config: EmbeddingPipelineConfig, gnn: Arc<GnnModel>) -> Self {
        let dim = config.embedding_dim;
        let metric = config.distance_metric;
        Self {
            config,
            vector_index: Arc::new(VectorIndex::new(dim, metric)),
            gnn_model: Some(gnn),
            property_fetcher: None,
            computed_count: RwLock::new(0),
        }
    }

    /// Establece el property fetcher para re-computar embeddings en PropertyChanged.
    pub fn with_property_fetcher(mut self, fetcher: PropertyFetcher) -> Self {
        self.property_fetcher = Some(fetcher);
        self
    }

    /// Referencia al vector index.
    pub fn vector_index(&self) -> &VectorIndex {
        &self.vector_index
    }

    /// Número de embeddings computados.
    pub fn computed_count(&self) -> usize {
        *self.computed_count.read()
    }

    /// Computa embedding para un nodo desde sus propiedades (feature-based).
    pub fn compute_feature_embedding(&self, properties: &[(u16, Value)]) -> Vec<f32> {
        let raw = gnn::extract_node_features(properties, &self.config.feature_prop_ids);

        // Pad or truncate to embedding_dim
        let mut embedding = vec![0.0; self.config.embedding_dim];
        let copy_len = raw.len().min(self.config.embedding_dim);
        embedding[..copy_len].copy_from_slice(&raw[..copy_len]);

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for v in embedding.iter_mut() {
                *v /= norm;
            }
        }

        embedding
    }

    /// Computa embedding con GNN (requiere features de vecinos).
    pub fn compute_gnn_embedding(
        &self,
        node_features: &[f32],
        neighbor_features_per_hop: &[Vec<Vec<f32>>],
    ) -> Vec<f32> {
        if let Some(gnn) = &self.gnn_model {
            let mut result = gnn.forward_single(node_features, neighbor_features_per_hop);
            // Pad/truncate to embedding_dim
            result.resize(self.config.embedding_dim, 0.0);
            result
        } else {
            // Fallback to feature-based
            let mut result = node_features.to_vec();
            result.resize(self.config.embedding_dim, 0.0);
            result
        }
    }

    /// Procesa un evento del grafo: computa/actualiza embedding si aplica.
    pub fn handle_event(&self, event: &GraphEvent) {
        match event {
            GraphEvent::NodeAdded { node_id, properties } => {
                let embedding = self.compute_feature_embedding(properties);
                self.vector_index.insert(*node_id, embedding);
                *self.computed_count.write() += 1;
            }
            GraphEvent::NodeRemoved { node_id } => {
                self.vector_index.remove(*node_id);
            }
            GraphEvent::PropertyChanged { node_id, property_id, .. } => {
                // Only recompute if the changed property is one of our features
                if self.config.feature_prop_ids.contains(property_id) {
                    if let Some(ref fetcher) = self.property_fetcher {
                        if let Some(all_props) = fetcher(*node_id) {
                            let embedding = self.compute_feature_embedding(&all_props);
                            self.vector_index.remove(*node_id);
                            self.vector_index.insert(*node_id, embedding);
                            *self.computed_count.write() += 1;
                        }
                    }
                }
            }
            // Edge changes may affect GNN-based embeddings — recompute endpoints
            GraphEvent::EdgeAdded { source, target } |
            GraphEvent::EdgeRemoved { source, target } => {
                if self.config.strategy == EmbeddingStrategy::GnnBased {
                    if let Some(ref fetcher) = self.property_fetcher {
                        for &nid in &[*source, *target] {
                            if let Some(all_props) = fetcher(nid) {
                                let embedding = self.compute_feature_embedding(&all_props);
                                self.vector_index.remove(nid);
                                self.vector_index.insert(nid, embedding);
                                *self.computed_count.write() += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Batch compute: genera embeddings para un lote de nodos.
    pub fn compute_batch(&self, nodes: &[(NodeId, Vec<(u16, Value)>)]) {
        for (node_id, properties) in nodes {
            let embedding = self.compute_feature_embedding(properties);
            self.vector_index.insert(*node_id, embedding);
            *self.computed_count.write() += 1;
        }
    }

    /// Búsqueda k-NN en el vector index.
    pub fn search_similar(&self, query: &[f32], k: usize) -> Vec<(NodeId, f32)> {
        self.vector_index
            .search(query, k)
            .into_iter()
            .map(|r| (r.node_id, r.distance))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IncrementalInference — Re-inferencia automática en nodos afectados
// ─────────────────────────────────────────────────────────────────────────────

/// Resultado de inferencia incremental.
#[derive(Debug, Clone)]
pub struct IncrementalResult {
    pub node_id: NodeId,
    pub prediction: Vec<f32>,
}

/// Motor de inferencia incremental.
///
/// Re-ejecuta inferencia solo en nodos afectados por mutaciones del grafo.
pub struct IncrementalInference {
    /// Modelo de inferencia a usar.
    model: Arc<InferenceModel>,
    /// IDs de propiedades usadas como features.
    feature_prop_ids: Vec<u16>,
    /// Cache de predicciones: node_id → última predicción.
    cache: RwLock<std::collections::HashMap<NodeId, Vec<f32>>>,
    /// Nodos pendientes de re-inferencia.
    dirty_nodes: RwLock<Vec<NodeId>>,
}

impl IncrementalInference {
    /// Crea un motor de inferencia incremental.
    pub fn new(model: Arc<InferenceModel>, feature_prop_ids: Vec<u16>) -> Self {
        Self {
            model,
            feature_prop_ids,
            cache: RwLock::new(std::collections::HashMap::new()),
            dirty_nodes: RwLock::new(Vec::new()),
        }
    }

    /// Procesa un evento: marca nodos afectados como dirty.
    pub fn handle_event(&self, event: &GraphEvent) {
        let affected = event.affected_nodes();
        let mut dirty = self.dirty_nodes.write();
        for node_id in affected {
            if !dirty.contains(&node_id) {
                dirty.push(node_id);
            }
        }
    }

    /// Ejecuta inferencia directa en un nodo (sin GNN).
    pub fn infer_node(&self, node_id: NodeId, properties: &[(u16, Value)]) -> Vec<f32> {
        let features = gnn::extract_node_features(properties, &self.feature_prop_ids);
        let prediction = self.model.predict(&features);

        // Update cache
        self.cache.write().insert(node_id, prediction.clone());

        // Remove from dirty
        self.dirty_nodes.write().retain(|&id| id != node_id);

        prediction
    }

    /// Ejecuta inferencia en todos los nodos dirty.
    ///
    /// `fetch_properties`: función que dado un NodeId retorna sus propiedades.
    pub fn flush_dirty<F>(&self, fetch_properties: F) -> Vec<IncrementalResult>
    where
        F: Fn(NodeId) -> Option<Vec<(u16, Value)>>,
    {
        let dirty: Vec<NodeId> = {
            let mut dirty = self.dirty_nodes.write();
            std::mem::take(&mut *dirty)
        };

        let mut results = Vec::with_capacity(dirty.len());
        for node_id in dirty {
            if let Some(props) = fetch_properties(node_id) {
                let prediction = self.infer_node(node_id, &props);
                results.push(IncrementalResult {
                    node_id,
                    prediction,
                });
            }
        }
        results
    }

    /// Obtiene predicción cacheada.
    pub fn cached_prediction(&self, node_id: NodeId) -> Option<Vec<f32>> {
        self.cache.read().get(&node_id).cloned()
    }

    /// ¿Hay nodos pendientes de re-inferencia?
    pub fn has_dirty_nodes(&self) -> bool {
        !self.dirty_nodes.read().is_empty()
    }

    /// Número de nodos dirty.
    pub fn dirty_count(&self) -> usize {
        self.dirty_nodes.read().len()
    }

    /// Número de predicciones cacheadas.
    pub fn cache_size(&self) -> usize {
        self.cache.read().len()
    }

    /// Limpia el cache completo.
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{InferenceTask, MlpLayer, MlpModel};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_event_bus_emit() {
        let bus = EventBus::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        bus.on_event(Box::new(move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        }));

        bus.emit(GraphEvent::NodeAdded {
            node_id: NodeId(1),
            properties: vec![],
        });
        bus.emit(GraphEvent::NodeAdded {
            node_id: NodeId(2),
            properties: vec![],
        });

        assert_eq!(counter.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_event_bus_multiple_listeners() {
        let bus = EventBus::new();
        let c1 = Arc::new(AtomicUsize::new(0));
        let c2 = Arc::new(AtomicUsize::new(0));

        let c1c = c1.clone();
        bus.on_event(Box::new(move |_| { c1c.fetch_add(1, Ordering::Relaxed); }));

        let c2c = c2.clone();
        bus.on_event(Box::new(move |_| { c2c.fetch_add(10, Ordering::Relaxed); }));

        bus.emit(GraphEvent::NodeRemoved { node_id: NodeId(1) });

        assert_eq!(c1.load(Ordering::Relaxed), 1);
        assert_eq!(c2.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_event_affected_nodes() {
        let event = GraphEvent::EdgeAdded {
            source: NodeId(1),
            target: NodeId(2),
        };
        assert_eq!(event.affected_nodes(), vec![NodeId(1), NodeId(2)]);

        let event2 = GraphEvent::NodeAdded {
            node_id: NodeId(5),
            properties: vec![],
        };
        assert_eq!(event2.affected_nodes(), vec![NodeId(5)]);
    }

    #[test]
    fn test_embedding_pipeline_feature_based() {
        let config = EmbeddingPipelineConfig {
            feature_prop_ids: vec![0, 1],
            embedding_dim: 4,
            strategy: EmbeddingStrategy::FeatureBased,
            distance_metric: DistanceMetric::Cosine,
        };
        let pipeline = EmbeddingPipeline::new(config);

        let props = vec![
            (0u16, Value::Float(3.0)),
            (1, Value::Float(4.0)),
        ];

        let embedding = pipeline.compute_feature_embedding(&props);
        assert_eq!(embedding.len(), 4);
        // Normalized: [3/5, 4/5, 0, 0]
        assert!((embedding[0] - 0.6).abs() < 1e-5);
        assert!((embedding[1] - 0.8).abs() < 1e-5);
        assert!((embedding[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_embedding_pipeline_handle_node_added() {
        let config = EmbeddingPipelineConfig {
            feature_prop_ids: vec![0],
            embedding_dim: 2,
            strategy: EmbeddingStrategy::FeatureBased,
            distance_metric: DistanceMetric::Euclidean,
        };
        let pipeline = EmbeddingPipeline::new(config);

        pipeline.handle_event(&GraphEvent::NodeAdded {
            node_id: NodeId(1),
            properties: vec![(0u16, Value::Float(1.0))],
        });
        pipeline.handle_event(&GraphEvent::NodeAdded {
            node_id: NodeId(2),
            properties: vec![(0u16, Value::Float(5.0))],
        });

        assert_eq!(pipeline.computed_count(), 2);
        assert_eq!(pipeline.vector_index().len(), 2);
    }

    #[test]
    fn test_embedding_pipeline_search() {
        let config = EmbeddingPipelineConfig {
            feature_prop_ids: vec![0, 1],
            embedding_dim: 2,
            strategy: EmbeddingStrategy::FeatureBased,
            distance_metric: DistanceMetric::Euclidean,
        };
        let pipeline = EmbeddingPipeline::new(config);

        // Insert nodes manually
        pipeline.vector_index.insert(NodeId(1), vec![1.0, 0.0]);
        pipeline.vector_index.insert(NodeId(2), vec![0.0, 1.0]);
        pipeline.vector_index.insert(NodeId(3), vec![0.9, 0.1]);

        let results = pipeline.search_similar(&[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, NodeId(1)); // exact match
    }

    #[test]
    fn test_embedding_pipeline_remove_on_node_removed() {
        let config = EmbeddingPipelineConfig {
            feature_prop_ids: vec![0],
            embedding_dim: 2,
            strategy: EmbeddingStrategy::FeatureBased,
            distance_metric: DistanceMetric::Cosine,
        };
        let pipeline = EmbeddingPipeline::new(config);

        pipeline.handle_event(&GraphEvent::NodeAdded {
            node_id: NodeId(1),
            properties: vec![(0u16, Value::Float(1.0))],
        });
        assert_eq!(pipeline.vector_index().len(), 1);

        pipeline.handle_event(&GraphEvent::NodeRemoved { node_id: NodeId(1) });
        assert_eq!(pipeline.vector_index().len(), 0);
    }

    #[test]
    fn test_embedding_pipeline_batch() {
        let config = EmbeddingPipelineConfig {
            feature_prop_ids: vec![0],
            embedding_dim: 2,
            strategy: EmbeddingStrategy::FeatureBased,
            distance_metric: DistanceMetric::Cosine,
        };
        let pipeline = EmbeddingPipeline::new(config);

        let batch = vec![
            (NodeId(1), vec![(0u16, Value::Float(1.0))]),
            (NodeId(2), vec![(0u16, Value::Float(2.0))]),
            (NodeId(3), vec![(0u16, Value::Float(3.0))]),
        ];
        pipeline.compute_batch(&batch);

        assert_eq!(pipeline.computed_count(), 3);
        assert_eq!(pipeline.vector_index().len(), 3);
    }

    #[test]
    fn test_incremental_inference_dirty_tracking() {
        let mut layer = MlpLayer::new(2, 1, crate::gnn::Activation::None);
        layer.weights = vec![1.0, 1.0]; // sum
        let mlp = MlpModel::new(vec![layer]);
        let model = Arc::new(InferenceModel::new(
            "test".into(), InferenceTask::Regression, mlp, vec![0, 1],
        ));

        let engine = IncrementalInference::new(model, vec![0, 1]);

        // Emit events → mark dirty
        engine.handle_event(&GraphEvent::NodeAdded {
            node_id: NodeId(1),
            properties: vec![],
        });
        engine.handle_event(&GraphEvent::EdgeAdded {
            source: NodeId(2),
            target: NodeId(3),
        });

        assert!(engine.has_dirty_nodes());
        assert_eq!(engine.dirty_count(), 3); // 1, 2, 3
    }

    #[test]
    fn test_incremental_inference_infer_node() {
        let mut layer = MlpLayer::new(2, 1, crate::gnn::Activation::None);
        layer.weights = vec![1.0, 2.0];
        let mlp = MlpModel::new(vec![layer]);
        let model = Arc::new(InferenceModel::new(
            "test".into(), InferenceTask::Regression, mlp, vec![0, 1],
        ));

        let engine = IncrementalInference::new(model, vec![0, 1]);

        let props = vec![(0u16, Value::Float(3.0)), (1, Value::Float(4.0))];
        let prediction = engine.infer_node(NodeId(1), &props);
        assert!((prediction[0] - 11.0).abs() < 1e-5); // 3*1 + 4*2

        // Should be cached
        let cached = engine.cached_prediction(NodeId(1)).unwrap();
        assert_eq!(cached, prediction);
    }

    #[test]
    fn test_incremental_inference_flush_dirty() {
        let mut layer = MlpLayer::new(1, 1, crate::gnn::Activation::None);
        layer.weights = vec![2.0];
        let mlp = MlpModel::new(vec![layer]);
        let model = Arc::new(InferenceModel::new(
            "test".into(), InferenceTask::Scoring, mlp, vec![0],
        ));

        let engine = IncrementalInference::new(model, vec![0]);

        engine.handle_event(&GraphEvent::NodeAdded {
            node_id: NodeId(1),
            properties: vec![],
        });
        engine.handle_event(&GraphEvent::NodeAdded {
            node_id: NodeId(2),
            properties: vec![],
        });

        assert_eq!(engine.dirty_count(), 2);

        let results = engine.flush_dirty(|node_id| {
            match node_id.0 {
                1 => Some(vec![(0u16, Value::Float(5.0))]),
                2 => Some(vec![(0u16, Value::Float(10.0))]),
                _ => None,
            }
        });

        assert_eq!(results.len(), 2);
        assert!(!engine.has_dirty_nodes());
        assert_eq!(engine.cache_size(), 2);
    }

    #[test]
    fn test_incremental_inference_clear_cache() {
        let layer = MlpLayer::new(1, 1, crate::gnn::Activation::None);
        let mlp = MlpModel::new(vec![layer]);
        let model = Arc::new(InferenceModel::new(
            "test".into(), InferenceTask::Scoring, mlp, vec![0],
        ));

        let engine = IncrementalInference::new(model, vec![0]);
        engine.infer_node(NodeId(1), &[(0u16, Value::Int(1))]);
        assert_eq!(engine.cache_size(), 1);

        engine.clear_cache();
        assert_eq!(engine.cache_size(), 0);
    }

    #[test]
    fn test_event_bus_with_pipeline_integration() {
        let bus = EventBus::new();

        let config = EmbeddingPipelineConfig {
            feature_prop_ids: vec![0],
            embedding_dim: 2,
            strategy: EmbeddingStrategy::FeatureBased,
            distance_metric: DistanceMetric::Cosine,
        };
        let pipeline = Arc::new(EmbeddingPipeline::new(config));
        let pipeline_clone = pipeline.clone();

        // Register pipeline as event listener
        bus.on_event(Box::new(move |event| {
            pipeline_clone.handle_event(event);
        }));

        // Emit node creation
        bus.emit(GraphEvent::NodeAdded {
            node_id: NodeId(42),
            properties: vec![(0u16, Value::Float(1.0))],
        });

        // Pipeline should have auto-computed the embedding
        assert_eq!(pipeline.computed_count(), 1);
        assert_eq!(pipeline.vector_index().len(), 1);
    }

    #[test]
    fn test_property_changed_recomputation() {
        use std::collections::HashMap;
        use std::sync::RwLock as StdRwLock;

        // Simulate a graph store
        let store: Arc<StdRwLock<HashMap<u64, Vec<(u16, Value)>>>> =
            Arc::new(StdRwLock::new(HashMap::new()));
        store.write().unwrap().insert(1, vec![
            (0u16, Value::Float(1.0)),
            (1, Value::Float(2.0)),
        ]);

        let store_clone = store.clone();
        let fetcher: PropertyFetcher = Arc::new(move |node_id: NodeId| {
            store_clone.read().unwrap().get(&node_id.0).cloned()
        });

        let config = EmbeddingPipelineConfig {
            feature_prop_ids: vec![0, 1],
            embedding_dim: 4,
            strategy: EmbeddingStrategy::FeatureBased,
            distance_metric: DistanceMetric::Euclidean,
        };
        let pipeline = Arc::new(EmbeddingPipeline::new(config).with_property_fetcher(fetcher));

        // First add the node
        pipeline.handle_event(&GraphEvent::NodeAdded {
            node_id: NodeId(1),
            properties: vec![(0u16, Value::Float(1.0)), (1, Value::Float(2.0))],
        });
        assert_eq!(pipeline.vector_index().len(), 1);
        let old_results = pipeline.search_similar(&[1.0, 0.0, 0.0, 0.0], 1);
        let old_distance = old_results[0].1;

        // Update the store (simulate property change — different ratio)
        store.write().unwrap().insert(1, vec![
            (0u16, Value::Float(0.1)),
            (1, Value::Float(20.0)),
        ]);

        // Emit PropertyChanged
        pipeline.handle_event(&GraphEvent::PropertyChanged {
            node_id: NodeId(1),
            property_id: 0,
            old_value: Some(Value::Float(1.0)),
            new_value: Value::Float(0.1),
        });

        // Embedding should have been recomputed (vector index still has 1 entry)
        assert_eq!(pipeline.vector_index().len(), 1);
        let new_results = pipeline.search_similar(&[1.0, 0.0, 0.0, 0.0], 1);
        let new_distance = new_results[0].1;

        // Distance should be different because the embedding changed
        assert!((old_distance - new_distance).abs() > 0.01);
    }

    #[test]
    fn test_property_changed_irrelevant_property_ignored() {
        let config = EmbeddingPipelineConfig {
            feature_prop_ids: vec![0], // only prop 0
            embedding_dim: 2,
            strategy: EmbeddingStrategy::FeatureBased,
            distance_metric: DistanceMetric::Euclidean,
        };
        let pipeline = EmbeddingPipeline::new(config);

        pipeline.handle_event(&GraphEvent::NodeAdded {
            node_id: NodeId(1),
            properties: vec![(0u16, Value::Float(1.0))],
        });
        let count_before = pipeline.computed_count();

        // Change property 5 (not in feature_prop_ids) → should NOT recompute
        pipeline.handle_event(&GraphEvent::PropertyChanged {
            node_id: NodeId(1),
            property_id: 5,
            old_value: None,
            new_value: Value::Float(99.0),
        });

        assert_eq!(pipeline.computed_count(), count_before);
    }
}
