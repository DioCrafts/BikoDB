// =============================================================================
// bikodb-ai::gnn — Graph Neural Network Engine
// =============================================================================
// Implementación de GNN (Graph Neural Networks) nativo en Rust.
//
// Soporta message passing, agregación de features por vecindario, y capas
// tipo GraphSAGE (Sample and Aggregate) y GCN (Graph Convolutional Network).
//
// ## Diseño
// - Sin dependencia de PyTorch/TensorFlow — puro Rust para inferencia
// - Pesos pre-entrenados se cargan como matrices f32
// - Message passing configurable: mean, sum, max aggregation
// - Forward pass directo sobre el grafo in-memory (sin exportar)
//
// ## Referencia
// - Hamilton et al., 2017 — "Inductive Representation Learning on Large Graphs"
// - Kipf & Welling, 2017 — "Semi-Supervised Classification with GCN"
// =============================================================================

use std::collections::HashMap;

/// Tipo de agregación para message passing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregation {
    /// Promedio de features de vecinos.
    Mean,
    /// Suma de features de vecinos.
    Sum,
    /// Máximo element-wise de features de vecinos.
    Max,
}

/// Función de activación.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    None,
}

impl Activation {
    /// Aplica la activación element-wise.
    pub fn apply(&self, x: &mut [f32]) {
        match self {
            Activation::ReLU => {
                for v in x.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            Activation::Sigmoid => {
                for v in x.iter_mut() {
                    *v = 1.0 / (1.0 + (-*v).exp());
                }
            }
            Activation::Tanh => {
                for v in x.iter_mut() {
                    *v = v.tanh();
                }
            }
            Activation::None => {}
        }
    }
}

/// Una capa de GNN (GraphSAGE-style).
///
/// Transforma features de nodos usando message passing:
///   h_v = σ(W_self · h_v + W_neigh · AGG({h_u : u ∈ N(v)}))
#[derive(Debug, Clone)]
pub struct GnnLayer {
    /// Dimensión de entrada.
    pub input_dim: usize,
    /// Dimensión de salida.
    pub output_dim: usize,
    /// Pesos para self-features: output_dim × input_dim (row-major).
    pub weights_self: Vec<f32>,
    /// Pesos para neighbor-aggregated features: output_dim × input_dim (row-major).
    pub weights_neigh: Vec<f32>,
    /// Bias: output_dim.
    pub bias: Vec<f32>,
    /// Tipo de agregación.
    pub aggregation: Aggregation,
    /// Función de activación.
    pub activation: Activation,
}

impl GnnLayer {
    /// Crea una capa GNN con pesos inicializados a cero.
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        aggregation: Aggregation,
        activation: Activation,
    ) -> Self {
        Self {
            input_dim,
            output_dim,
            weights_self: vec![0.0; output_dim * input_dim],
            weights_neigh: vec![0.0; output_dim * input_dim],
            bias: vec![0.0; output_dim],
            aggregation,
            activation,
        }
    }

    /// Crea una capa con pesos aleatorios (Xavier initialization).
    pub fn new_random(
        input_dim: usize,
        output_dim: usize,
        aggregation: Aggregation,
        activation: Activation,
    ) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt() as f32;

        let n = output_dim * input_dim;
        let weights_self: Vec<f32> = (0..n).map(|_| rng.gen_range(-scale..scale)).collect();
        let weights_neigh: Vec<f32> = (0..n).map(|_| rng.gen_range(-scale..scale)).collect();
        let bias = vec![0.0; output_dim];

        Self {
            input_dim,
            output_dim,
            weights_self,
            weights_neigh,
            bias,
            aggregation,
            activation,
        }
    }

    /// Forward pass para un solo nodo.
    ///
    /// - `self_features`: features del nodo actual (input_dim)
    /// - `neighbor_features`: lista de features de vecinos (cada uno input_dim)
    ///
    /// Retorna el nuevo embedding del nodo (output_dim).
    pub fn forward_node(
        &self,
        self_features: &[f32],
        neighbor_features: &[&[f32]],
    ) -> Vec<f32> {
        debug_assert_eq!(self_features.len(), self.input_dim);

        // 1. Aggregate neighbor features
        let aggregated = self.aggregate(neighbor_features);

        // 2. h = W_self · self_features + W_neigh · aggregated + bias
        let mut output = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            let mut val = self.bias[i];
            for j in 0..self.input_dim {
                val += self.weights_self[i * self.input_dim + j] * self_features[j];
                val += self.weights_neigh[i * self.input_dim + j] * aggregated[j];
            }
            output[i] = val;
        }

        // 3. Apply activation
        self.activation.apply(&mut output);

        output
    }

    /// Agrega features de vecinos según la estrategia configurada.
    fn aggregate(&self, neighbor_features: &[&[f32]]) -> Vec<f32> {
        if neighbor_features.is_empty() {
            return vec![0.0; self.input_dim];
        }

        match self.aggregation {
            Aggregation::Mean => {
                let mut result = vec![0.0; self.input_dim];
                for feat in neighbor_features {
                    for (i, v) in feat.iter().enumerate() {
                        result[i] += v;
                    }
                }
                let n = neighbor_features.len() as f32;
                for v in result.iter_mut() {
                    *v /= n;
                }
                result
            }
            Aggregation::Sum => {
                let mut result = vec![0.0; self.input_dim];
                for feat in neighbor_features {
                    for (i, v) in feat.iter().enumerate() {
                        result[i] += v;
                    }
                }
                result
            }
            Aggregation::Max => {
                let mut result = vec![f32::NEG_INFINITY; self.input_dim];
                for feat in neighbor_features {
                    for (i, v) in feat.iter().enumerate() {
                        if *v > result[i] {
                            result[i] = *v;
                        }
                    }
                }
                // Replace -inf with 0 for isolated nodes
                for v in result.iter_mut() {
                    if v.is_infinite() {
                        *v = 0.0;
                    }
                }
                result
            }
        }
    }
}

/// Modelo GNN completo con múltiples capas.
///
/// Ejecuta forward pass sobre un grafo completo (o subgrafo) dado:
/// - Features iniciales de cada nodo
/// - Adjacency list (vecinos de cada nodo)
///
/// # Ejemplo
/// ```
/// use bikodb_ai::gnn::{GnnModel, GnnLayer, Aggregation, Activation};
///
/// let layer = GnnLayer::new(4, 2, Aggregation::Mean, Activation::ReLU);
/// let model = GnnModel::new(vec![layer]);
///
/// // 2 nodos conectados, 4 features cada uno
/// let features = vec![
///     vec![1.0, 0.0, 0.5, 0.2],
///     vec![0.0, 1.0, 0.3, 0.8],
/// ];
/// let adjacency = vec![vec![1usize], vec![0]]; // bidireccional
///
/// let embeddings = model.forward(&features, &adjacency);
/// assert_eq!(embeddings.len(), 2);
/// assert_eq!(embeddings[0].len(), 2); // output_dim = 2
/// ```
pub struct GnnModel {
    layers: Vec<GnnLayer>,
}

impl GnnModel {
    /// Crea un modelo GNN con las capas especificadas.
    pub fn new(layers: Vec<GnnLayer>) -> Self {
        Self { layers }
    }

    /// Número de capas.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Forward pass completo sobre el grafo.
    ///
    /// - `features`: features iniciales de cada nodo [node_idx → Vec<f32>]
    /// - `adjacency`: adjacency list [node_idx → Vec<neighbor_idx>]
    ///
    /// Retorna embeddings finales para cada nodo.
    pub fn forward(
        &self,
        features: &[Vec<f32>],
        adjacency: &[Vec<usize>],
    ) -> Vec<Vec<f32>> {
        assert_eq!(features.len(), adjacency.len());
        let n = features.len();

        let mut current: Vec<Vec<f32>> = features.to_vec();

        for layer in &self.layers {
            let mut next = Vec::with_capacity(n);
            for node_idx in 0..n {
                let self_feat = &current[node_idx];
                let neighbor_feats: Vec<&[f32]> = adjacency[node_idx]
                    .iter()
                    .map(|&nidx| current[nidx].as_slice())
                    .collect();

                let new_embedding = layer.forward_node(self_feat, &neighbor_feats);
                next.push(new_embedding);
            }
            current = next;
        }

        current
    }

    /// Forward pass para un solo nodo (inferencia puntual).
    ///
    /// Útil para inferencia incremental: solo recalcula el embedding
    /// de un nodo y sus vecinos inmediatos.
    ///
    /// - `node_features`: features del nodo actual + vecinos en cada hop
    /// - Se procesa bottom-up las capas usando neighborhood sampling
    pub fn forward_single(
        &self,
        node_feature: &[f32],
        neighbor_features_per_hop: &[Vec<Vec<f32>>],
    ) -> Vec<f32> {
        if self.layers.is_empty() {
            return node_feature.to_vec();
        }

        // Process from last hop to first (bottom-up)
        // hop 0 = immediate neighbors, hop 1 = 2-hop neighbors, etc.
        let depth = self.layers.len().min(neighbor_features_per_hop.len() + 1);

        // Start: if we have enough hops, aggregate from outermost
        let mut current_node = node_feature.to_vec();

        for (layer_idx, layer) in self.layers.iter().enumerate().take(depth) {
            if layer_idx < neighbor_features_per_hop.len() {
                let neighbor_feats: Vec<&[f32]> = neighbor_features_per_hop[layer_idx]
                    .iter()
                    .map(|v| v.as_slice())
                    .collect();
                current_node = layer.forward_node(&current_node, &neighbor_feats);
            } else {
                // No neighbors at this hop — self-only transform
                current_node = layer.forward_node(&current_node, &[]);
            }
        }

        current_node
    }

    /// Serializa los pesos del modelo a bytes (para persistencia).
    pub fn serialize_weights(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Number of layers
        buf.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());

        for layer in &self.layers {
            buf.extend_from_slice(&(layer.input_dim as u32).to_le_bytes());
            buf.extend_from_slice(&(layer.output_dim as u32).to_le_bytes());
            buf.extend_from_slice(&(layer.aggregation as u8).to_le_bytes());
            buf.extend_from_slice(&(layer.activation as u8).to_le_bytes());

            // Weights self
            for &w in &layer.weights_self {
                buf.extend_from_slice(&w.to_le_bytes());
            }
            // Weights neigh
            for &w in &layer.weights_neigh {
                buf.extend_from_slice(&w.to_le_bytes());
            }
            // Bias
            for &b in &layer.bias {
                buf.extend_from_slice(&b.to_le_bytes());
            }
        }
        buf
    }

    /// Deserializa pesos desde bytes.
    pub fn deserialize_weights(data: &[u8]) -> Option<Self> {
        let mut pos = 0;

        if data.len() < 4 {
            return None;
        }
        let num_layers = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            if pos + 10 > data.len() {
                return None;
            }
            let input_dim = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            let output_dim = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            let agg_byte = data[pos];
            pos += 1;
            let act_byte = data[pos];
            pos += 1;

            let aggregation = match agg_byte {
                0 => Aggregation::Mean,
                1 => Aggregation::Sum,
                2 => Aggregation::Max,
                _ => return None,
            };
            let activation = match act_byte {
                0 => Activation::ReLU,
                1 => Activation::Sigmoid,
                2 => Activation::Tanh,
                3 => Activation::None,
                _ => return None,
            };

            let n_self = output_dim * input_dim;
            let n_neigh = output_dim * input_dim;
            let n_bias = output_dim;
            let total_floats = n_self + n_neigh + n_bias;

            if pos + total_floats * 4 > data.len() {
                return None;
            }

            let mut weights_self = Vec::with_capacity(n_self);
            for _ in 0..n_self {
                weights_self.push(f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?));
                pos += 4;
            }

            let mut weights_neigh = Vec::with_capacity(n_neigh);
            for _ in 0..n_neigh {
                weights_neigh.push(f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?));
                pos += 4;
            }

            let mut bias = Vec::with_capacity(n_bias);
            for _ in 0..n_bias {
                bias.push(f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?));
                pos += 4;
            }

            layers.push(GnnLayer {
                input_dim,
                output_dim,
                weights_self,
                weights_neigh,
                bias,
                aggregation,
                activation,
            });
        }

        Some(Self { layers })
    }
}

/// Extrae features de nodos como f32 desde las propiedades.
///
/// Dado un mapping de property_ids, extrae valores numéricos o embedding
/// de cada nodo. Para valores no numéricos, usa 0.0.
pub fn extract_node_features(
    properties: &[(u16, bikodb_core::value::Value)],
    feature_prop_ids: &[u16],
) -> Vec<f32> {
    let prop_map: HashMap<u16, &bikodb_core::value::Value> =
        properties.iter().map(|(k, v)| (*k, v)).collect();

    let mut features = Vec::new();
    for &pid in feature_prop_ids {
        match prop_map.get(&pid) {
            Some(bikodb_core::value::Value::Int(v)) => features.push(*v as f32),
            Some(bikodb_core::value::Value::Float(v)) => features.push(*v as f32),
            Some(bikodb_core::value::Value::Bool(v)) => features.push(if *v { 1.0 } else { 0.0 }),
            Some(bikodb_core::value::Value::Embedding(emb)) => features.extend_from_slice(emb),
            _ => features.push(0.0),
        }
    }
    features
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_relu() {
        let mut v = vec![-1.0, 0.0, 1.0, -0.5, 2.0];
        Activation::ReLU.apply(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_activation_sigmoid() {
        let mut v = vec![0.0];
        Activation::Sigmoid.apply(&mut v);
        assert!((v[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_aggregation_mean() {
        let layer = GnnLayer::new(3, 1, Aggregation::Mean, Activation::None);
        let n1 = vec![1.0, 2.0, 3.0];
        let n2 = vec![3.0, 4.0, 5.0];
        let agg = layer.aggregate(&[&n1, &n2]);
        assert!((agg[0] - 2.0).abs() < 1e-5);
        assert!((agg[1] - 3.0).abs() < 1e-5);
        assert!((agg[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_aggregation_sum() {
        let layer = GnnLayer::new(2, 1, Aggregation::Sum, Activation::None);
        let n1 = vec![1.0, 2.0];
        let n2 = vec![3.0, 4.0];
        let agg = layer.aggregate(&[&n1, &n2]);
        assert!((agg[0] - 4.0).abs() < 1e-5);
        assert!((agg[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_aggregation_max() {
        let layer = GnnLayer::new(2, 1, Aggregation::Max, Activation::None);
        let n1 = vec![1.0, 4.0];
        let n2 = vec![3.0, 2.0];
        let agg = layer.aggregate(&[&n1, &n2]);
        assert!((agg[0] - 3.0).abs() < 1e-5);
        assert!((agg[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_aggregation_empty() {
        let layer = GnnLayer::new(3, 1, Aggregation::Mean, Activation::None);
        let agg = layer.aggregate(&[]);
        assert_eq!(agg, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_layer_forward_identity() {
        // Identity weights: output = self_features (when no neighbors)
        let mut layer = GnnLayer::new(2, 2, Aggregation::Mean, Activation::None);
        // Set W_self to identity
        layer.weights_self = vec![1.0, 0.0, 0.0, 1.0];

        let result = layer.forward_node(&[3.0, 7.0], &[]);
        assert!((result[0] - 3.0).abs() < 1e-5);
        assert!((result[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_layer_forward_with_neighbors() {
        let mut layer = GnnLayer::new(2, 2, Aggregation::Mean, Activation::None);
        // W_self = identity, W_neigh = identity
        layer.weights_self = vec![1.0, 0.0, 0.0, 1.0];
        layer.weights_neigh = vec![1.0, 0.0, 0.0, 1.0];

        let neighbors = vec![vec![2.0, 4.0], vec![4.0, 6.0]];
        let neigh_refs: Vec<&[f32]> = neighbors.iter().map(|v| v.as_slice()).collect();

        // self=[1,1], neigh_mean=[3,5] → output=[4,6]
        let result = layer.forward_node(&[1.0, 1.0], &neigh_refs);
        assert!((result[0] - 4.0).abs() < 1e-5);
        assert!((result[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_model_forward() {
        let mut layer = GnnLayer::new(2, 2, Aggregation::Mean, Activation::None);
        layer.weights_self = vec![1.0, 0.0, 0.0, 1.0];
        layer.weights_neigh = vec![0.0; 4]; // ignore neighbors

        let model = GnnModel::new(vec![layer]);

        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let adjacency = vec![vec![1usize], vec![0]];

        let embeddings = model.forward(&features, &adjacency);
        assert_eq!(embeddings.len(), 2);
        assert!((embeddings[0][0] - 1.0).abs() < 1e-5);
        assert!((embeddings[0][1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_model_two_layers() {
        let mut l1 = GnnLayer::new(2, 2, Aggregation::Mean, Activation::None);
        l1.weights_self = vec![1.0, 0.0, 0.0, 1.0]; // identity
        l1.weights_neigh = vec![0.0; 4];

        let mut l2 = GnnLayer::new(2, 1, Aggregation::Sum, Activation::None);
        l2.weights_self = vec![1.0, 1.0]; // sum both dims
        l2.weights_neigh = vec![0.0; 2];

        let model = GnnModel::new(vec![l1, l2]);

        let features = vec![vec![3.0, 4.0]];
        let adjacency = vec![vec![]];

        let result = model.forward(&features, &adjacency);
        assert_eq!(result.len(), 1);
        assert!((result[0][0] - 7.0).abs() < 1e-5); // 3 + 4
    }

    #[test]
    fn test_model_serialize_deserialize() {
        let mut layer = GnnLayer::new(2, 3, Aggregation::Sum, Activation::ReLU);
        layer.weights_self = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        layer.weights_neigh = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        layer.bias = vec![0.01, 0.02, 0.03];

        let model = GnnModel::new(vec![layer]);
        let bytes = model.serialize_weights();
        let restored = GnnModel::deserialize_weights(&bytes).unwrap();

        assert_eq!(restored.layers.len(), 1);
        assert_eq!(restored.layers[0].input_dim, 2);
        assert_eq!(restored.layers[0].output_dim, 3);
        assert_eq!(restored.layers[0].aggregation, Aggregation::Sum);
        assert_eq!(restored.layers[0].activation, Activation::ReLU);
        assert_eq!(restored.layers[0].weights_self, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(restored.layers[0].bias, vec![0.01, 0.02, 0.03]);
    }

    #[test]
    fn test_forward_single_node() {
        let mut layer = GnnLayer::new(2, 2, Aggregation::Mean, Activation::None);
        layer.weights_self = vec![1.0, 0.0, 0.0, 1.0];
        layer.weights_neigh = vec![1.0, 0.0, 0.0, 1.0];

        let model = GnnModel::new(vec![layer]);

        let neigh = vec![vec![vec![2.0, 3.0], vec![4.0, 5.0]]]; // 1 hop neighbors
        let result = model.forward_single(&[1.0, 1.0], &neigh);
        // neigh mean=[3,4], self=[1,1], output=[4,5]
        assert!((result[0] - 4.0).abs() < 1e-5);
        assert!((result[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_extract_node_features() {
        use bikodb_core::value::Value;

        let props = vec![
            (0u16, Value::Int(42)),
            (1, Value::Float(3.14)),
            (2, Value::Bool(true)),
            (3, Value::from("text")), // non-numeric → 0.0
        ];

        let features = extract_node_features(&props, &[0, 1, 2, 3]);
        assert!((features[0] - 42.0).abs() < 1e-5);
        assert!((features[1] - 3.14).abs() < 0.01);
        assert!((features[2] - 1.0).abs() < 1e-5);
        assert!((features[3] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_random_init() {
        let layer = GnnLayer::new_random(4, 3, Aggregation::Mean, Activation::ReLU);
        assert_eq!(layer.weights_self.len(), 12);
        assert_eq!(layer.weights_neigh.len(), 12);
        assert_eq!(layer.bias.len(), 3);
        // Weights should not all be zero (extremely unlikely)
        let sum: f32 = layer.weights_self.iter().sum();
        // Not checking exact value, just that initialization happened
        assert!(layer.weights_self.iter().any(|&w| w != 0.0));
        let _ = sum;
    }
}
