// =============================================================================
// bikodb-ai::inference — Motor de inferencia directa sobre grafos
// =============================================================================
// Ejecuta modelos de ML directamente sobre los datos del grafo sin necesidad
// de exportar a frameworks externos.
//
// Soporta:
// - Clasificación de nodos (node classification)
// - Predicción de enlaces (link prediction)
// - Regresión sobre propiedades de nodos
// - Cálculo de scores/embeddings custom
//
// ## Diseño
// - Modelos como funciones puras (features → prediction)
// - Sin dependencias externas (ONNX, TF, PyTorch)
// - Lightweight MLP para inferencia rápida
// - Integración directa con GNN para graph-aware inference
// =============================================================================

use crate::gnn::{Activation, GnnModel};
use std::collections::HashMap;

/// Tipo de tarea de inferencia.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceTask {
    /// Clasificación de nodos (predice clase/categoría).
    NodeClassification,
    /// Predicción de enlaces (predice si debería existir enlace).
    LinkPrediction,
    /// Regresión (predice valor numérico).
    Regression,
    /// Scoring custom (output raw vector).
    Scoring,
}

/// Un modelo de inferencia registrado en el sistema.
pub struct InferenceModel {
    /// Nombre único del modelo.
    pub name: String,
    /// Tarea que resuelve.
    pub task: InferenceTask,
    /// Red neuronal feed-forward para inferencia.
    mlp: MlpModel,
    /// Modelo GNN opcional (para features graph-aware).
    gnn: Option<GnnModel>,
    /// IDs de propiedades usadas como features de entrada.
    pub feature_prop_ids: Vec<u16>,
}

impl InferenceModel {
    /// Crea un modelo de inferencia con MLP.
    pub fn new(
        name: String,
        task: InferenceTask,
        mlp: MlpModel,
        feature_prop_ids: Vec<u16>,
    ) -> Self {
        Self {
            name,
            task,
            mlp,
            gnn: None,
            feature_prop_ids,
        }
    }

    /// Crea un modelo con GNN + MLP head.
    pub fn with_gnn(
        name: String,
        task: InferenceTask,
        gnn: GnnModel,
        mlp: MlpModel,
        feature_prop_ids: Vec<u16>,
    ) -> Self {
        Self {
            name,
            task,
            mlp,
            gnn: Some(gnn),
            feature_prop_ids,
        }
    }

    /// Inferencia directa sobre features de un nodo.
    ///
    /// Si el modelo tiene GNN, las features deben ser el embedding post-GNN.
    /// Si no tiene GNN, las features son las propiedades directas del nodo.
    pub fn predict(&self, features: &[f32]) -> Vec<f32> {
        self.mlp.forward(features)
    }

    /// Inferencia con GNN: dado features del nodo y sus vecinos por hop,
    /// primero pasa por GNN y luego por MLP head.
    pub fn predict_with_neighborhood(
        &self,
        node_features: &[f32],
        neighbor_features_per_hop: &[Vec<Vec<f32>>],
    ) -> Vec<f32> {
        let gnn_output = if let Some(gnn) = &self.gnn {
            gnn.forward_single(node_features, neighbor_features_per_hop)
        } else {
            node_features.to_vec()
        };
        self.mlp.forward(&gnn_output)
    }

    /// Predicción de enlace: dadas features de dos nodos, predice score.
    pub fn predict_link(&self, features_a: &[f32], features_b: &[f32]) -> f32 {
        // Concatenar features de ambos nodos
        let mut combined = Vec::with_capacity(features_a.len() + features_b.len());
        combined.extend_from_slice(features_a);
        combined.extend_from_slice(features_b);
        let output = self.mlp.forward(&combined);
        output.first().copied().unwrap_or(0.0)
    }

    /// ¿Tiene GNN?
    pub fn has_gnn(&self) -> bool {
        self.gnn.is_some()
    }
}

/// Multi-Layer Perceptron para inferencia.
///
/// Red feed-forward simple con capas densas + activaciones.
///
/// # Ejemplo
/// ```
/// use bikodb_ai::inference::{MlpModel, MlpLayer};
/// use bikodb_ai::gnn::Activation;
///
/// let model = MlpModel::new(vec![
///     MlpLayer::new(4, 8, Activation::ReLU),
///     MlpLayer::new(8, 2, Activation::None),
/// ]);
///
/// let input = vec![1.0, 0.5, 0.3, 0.8];
/// let output = model.forward(&input);
/// assert_eq!(output.len(), 2);
/// ```
pub struct MlpModel {
    layers: Vec<MlpLayer>,
}

impl MlpModel {
    /// Crea un MLP con las capas dadas.
    pub fn new(layers: Vec<MlpLayer>) -> Self {
        Self { layers }
    }

    /// Forward pass.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut current = input.to_vec();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }

    /// Número de capas.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// Capa densa (fully connected).
pub struct MlpLayer {
    pub input_dim: usize,
    pub output_dim: usize,
    /// Pesos: output_dim × input_dim (row-major).
    pub weights: Vec<f32>,
    /// Bias: output_dim.
    pub bias: Vec<f32>,
    pub activation: Activation,
}

impl MlpLayer {
    /// Crea una capa con pesos inicializados a cero.
    pub fn new(input_dim: usize, output_dim: usize, activation: Activation) -> Self {
        Self {
            input_dim,
            output_dim,
            weights: vec![0.0; output_dim * input_dim],
            bias: vec![0.0; output_dim],
            activation,
        }
    }

    /// Forward pass de la capa.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            let mut val = self.bias[i];
            for j in 0..self.input_dim.min(input.len()) {
                val += self.weights[i * self.input_dim + j] * input[j];
            }
            output[i] = val;
        }
        self.activation.apply(&mut output);
        output
    }
}

/// Registro de modelos de inferencia.
///
/// Permite registrar múltiples modelos y ejecutar inferencia por nombre.
pub struct InferenceRegistry {
    models: HashMap<String, InferenceModel>,
}

impl InferenceRegistry {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Registra un modelo.
    pub fn register(&mut self, model: InferenceModel) {
        self.models.insert(model.name.clone(), model);
    }

    /// Desregistra un modelo.
    pub fn unregister(&mut self, name: &str) -> Option<InferenceModel> {
        self.models.remove(name)
    }

    /// Ejecuta inferencia usando un modelo registrado.
    pub fn predict(&self, model_name: &str, features: &[f32]) -> Option<Vec<f32>> {
        self.models.get(model_name).map(|m| m.predict(features))
    }

    /// Ejecuta inferencia con neighborhood (GNN mode).
    pub fn predict_with_neighborhood(
        &self,
        model_name: &str,
        node_features: &[f32],
        neighbor_features_per_hop: &[Vec<Vec<f32>>],
    ) -> Option<Vec<f32>> {
        self.models.get(model_name).map(|m| {
            m.predict_with_neighborhood(node_features, neighbor_features_per_hop)
        })
    }

    /// Lista modelos registrados.
    pub fn list_models(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }

    /// ¿Existe el modelo?
    pub fn has_model(&self, name: &str) -> bool {
        self.models.contains_key(name)
    }

    /// Número de modelos registrados.
    pub fn model_count(&self) -> usize {
        self.models.len()
    }
}

impl Default for InferenceRegistry {
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

    #[test]
    fn test_mlp_layer_zero_weights() {
        let layer = MlpLayer::new(3, 2, Activation::None);
        let output = layer.forward(&[1.0, 2.0, 3.0]);
        assert_eq!(output, vec![0.0, 0.0]); // All zeros
    }

    #[test]
    fn test_mlp_layer_identity() {
        let mut layer = MlpLayer::new(2, 2, Activation::None);
        layer.weights = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let output = layer.forward(&[5.0, 7.0]);
        assert!((output[0] - 5.0).abs() < 1e-5);
        assert!((output[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_mlp_layer_with_bias() {
        let mut layer = MlpLayer::new(2, 1, Activation::None);
        layer.weights = vec![1.0, 1.0]; // sum
        layer.bias = vec![10.0];
        let output = layer.forward(&[3.0, 4.0]);
        assert!((output[0] - 17.0).abs() < 1e-5); // 3+4+10
    }

    #[test]
    fn test_mlp_layer_relu() {
        let mut layer = MlpLayer::new(2, 2, Activation::ReLU);
        layer.weights = vec![1.0, 0.0, -1.0, 0.0]; // pass-through and negate
        let output = layer.forward(&[5.0, 0.0]);
        assert!((output[0] - 5.0).abs() < 1e-5);
        assert!((output[1] - 0.0).abs() < 1e-5); // ReLU(-5) = 0
    }

    #[test]
    fn test_mlp_model_two_layers() {
        let mut l1 = MlpLayer::new(3, 2, Activation::None);
        l1.weights = vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // [a+b, c]

        let mut l2 = MlpLayer::new(2, 1, Activation::None);
        l2.weights = vec![1.0, 1.0]; // sum

        let model = MlpModel::new(vec![l1, l2]);
        let output = model.forward(&[2.0, 3.0, 4.0]);
        assert!((output[0] - 9.0).abs() < 1e-5); // (2+3) + 4
    }

    #[test]
    fn test_inference_model_predict() {
        let mut layer = MlpLayer::new(2, 1, Activation::None);
        layer.weights = vec![1.0, 2.0];
        let mlp = MlpModel::new(vec![layer]);

        let model = InferenceModel::new(
            "test".into(),
            InferenceTask::Regression,
            mlp,
            vec![0, 1],
        );

        let result = model.predict(&[3.0, 4.0]);
        assert!((result[0] - 11.0).abs() < 1e-5); // 3*1 + 4*2
    }

    #[test]
    fn test_inference_model_link_prediction() {
        let mut layer = MlpLayer::new(4, 1, Activation::Sigmoid);
        layer.weights = vec![0.5, 0.5, 0.5, 0.5];
        let mlp = MlpModel::new(vec![layer]);

        let model = InferenceModel::new(
            "link_pred".into(),
            InferenceTask::LinkPrediction,
            mlp,
            vec![],
        );

        let score = model.predict_link(&[1.0, 0.0], &[0.0, 1.0]);
        // sigmoid(0.5*1 + 0.5*0 + 0.5*0 + 0.5*1) = sigmoid(1.0) ≈ 0.731
        assert!(score > 0.7 && score < 0.8);
    }

    #[test]
    fn test_inference_registry() {
        let mut registry = InferenceRegistry::new();

        let mut layer = MlpLayer::new(2, 1, Activation::None);
        layer.weights = vec![1.0, 1.0];
        let mlp = MlpModel::new(vec![layer]);

        let model = InferenceModel::new("sum_model".into(), InferenceTask::Scoring, mlp, vec![0, 1]);
        registry.register(model);

        assert!(registry.has_model("sum_model"));
        assert_eq!(registry.model_count(), 1);

        let result = registry.predict("sum_model", &[5.0, 3.0]).unwrap();
        assert!((result[0] - 8.0).abs() < 1e-5);

        assert!(registry.predict("nonexistent", &[1.0]).is_none());
    }

    #[test]
    fn test_inference_with_gnn() {
        use crate::gnn::{Aggregation, GnnLayer};

        // GNN layer: identity for self, zeros for neighbors
        let mut gnn_layer = GnnLayer::new(2, 2, Aggregation::Mean, Activation::None);
        gnn_layer.weights_self = vec![1.0, 0.0, 0.0, 1.0];
        let gnn = GnnModel::new(vec![gnn_layer]);

        // MLP head: sum both dims
        let mut mlp_layer = MlpLayer::new(2, 1, Activation::None);
        mlp_layer.weights = vec![1.0, 1.0];
        let mlp = MlpModel::new(vec![mlp_layer]);

        let model = InferenceModel::with_gnn(
            "gnn_model".into(),
            InferenceTask::NodeClassification,
            gnn,
            mlp,
            vec![0, 1],
        );

        assert!(model.has_gnn());

        let result = model.predict_with_neighborhood(&[3.0, 4.0], &[]);
        assert!((result[0] - 7.0).abs() < 1e-5); // GNN passthrough → 3+4
    }

    #[test]
    fn test_registry_unregister() {
        let mut registry = InferenceRegistry::new();

        let layer = MlpLayer::new(1, 1, Activation::None);
        let mlp = MlpModel::new(vec![layer]);
        let model = InferenceModel::new("temp".into(), InferenceTask::Scoring, mlp, vec![]);
        registry.register(model);

        assert_eq!(registry.model_count(), 1);
        registry.unregister("temp");
        assert_eq!(registry.model_count(), 0);
    }

    #[test]
    fn test_registry_list_models() {
        let mut registry = InferenceRegistry::new();

        for name in &["model_a", "model_b"] {
            let layer = MlpLayer::new(1, 1, Activation::None);
            let mlp = MlpModel::new(vec![layer]);
            let model = InferenceModel::new(name.to_string(), InferenceTask::Scoring, mlp, vec![]);
            registry.register(model);
        }

        let mut models = registry.list_models();
        models.sort();
        assert_eq!(models, vec!["model_a", "model_b"]);
    }
}
