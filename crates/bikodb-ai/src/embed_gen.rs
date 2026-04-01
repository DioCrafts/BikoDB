// =============================================================================
// bikodb-ai::embed_gen — Generación de embeddings desde propiedades del grafo
// =============================================================================
// Genera embeddings semánticos computados directamente desde las propiedades
// de los nodos, sin necesidad de modelos externos.
//
// ## Estrategias
// - FeatureVector: concatena propiedades numéricas → normaliza
// - HashEmbedding: feature hashing para propiedades categóricas/string
// - StructuralEmbedding: encodifica estructura del grafo (grado, clustering)
// - CombinedEmbedding: combina múltiples estrategias
//
// ## Diseño
// Puro Rust, sin dependencias ML externas. Útil como baseline o como input
// para modelos GNN downstream.
// =============================================================================

use bikodb_core::value::Value;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ─────────────────────────────────────────────────────────────────────────────
// EmbeddingProvider — Trait pluggable para proveedores de embeddings
// ─────────────────────────────────────────────────────────────────────────────

/// Trait para proveedores de embeddings pluggables.
///
/// Permite integrar proveedores externos (transformers, sentence-BERT, etc.)
/// sin cambiar el core engine. Implementaciones built-in:
/// - `FeatureVectorProvider`: concatena propiedades numéricas
/// - `HashEmbeddingProvider`: feature hashing para propiedades mixtas
///
/// # Ejemplo
/// ```
/// use bikodb_ai::embed_gen::EmbeddingProvider;
/// use bikodb_core::value::Value;
///
/// struct MyProvider;
/// impl EmbeddingProvider for MyProvider {
///     fn embed(&self, _properties: &[(u16, Value)]) -> Vec<f32> {
///         vec![0.0; 128]
///     }
///     fn dimensions(&self) -> usize { 128 }
///     fn name(&self) -> &str { "custom" }
/// }
/// ```
pub trait EmbeddingProvider: Send + Sync {
    /// Genera un embedding desde propiedades de un nodo.
    fn embed(&self, properties: &[(u16, Value)]) -> Vec<f32>;
    /// Dimensión del embedding de salida.
    fn dimensions(&self) -> usize;
    /// Nombre del proveedor.
    fn name(&self) -> &str;
}

/// Proveedor de embeddings basado en vector de features numéricas.
pub struct FeatureVectorProvider {
    generator: EmbeddingGenerator,
}

impl FeatureVectorProvider {
    pub fn new(output_dim: usize, property_ids: Vec<u16>) -> Self {
        Self {
            generator: EmbeddingGenerator::new(output_dim, EmbedStrategy::FeatureVector, property_ids),
        }
    }
}

impl EmbeddingProvider for FeatureVectorProvider {
    fn embed(&self, properties: &[(u16, Value)]) -> Vec<f32> {
        self.generator.generate(properties)
    }
    fn dimensions(&self) -> usize {
        self.generator.output_dim
    }
    fn name(&self) -> &str {
        "feature_vector"
    }
}

/// Proveedor de embeddings basado en feature hashing (soporta strings).
pub struct HashEmbeddingProvider {
    generator: EmbeddingGenerator,
}

impl HashEmbeddingProvider {
    pub fn new(output_dim: usize, property_ids: Vec<u16>) -> Self {
        Self {
            generator: EmbeddingGenerator::new(output_dim, EmbedStrategy::HashEmbedding, property_ids),
        }
    }
}

impl EmbeddingProvider for HashEmbeddingProvider {
    fn embed(&self, properties: &[(u16, Value)]) -> Vec<f32> {
        self.generator.generate(properties)
    }
    fn dimensions(&self) -> usize {
        self.generator.output_dim
    }
    fn name(&self) -> &str {
        "hash_embedding"
    }
}

/// Estrategia de generación de embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedStrategy {
    /// Concatena propiedades numéricas en un vector.
    FeatureVector,
    /// Feature hashing para propiedades mixtas (numéricas + categóricas).
    HashEmbedding,
    /// Embedding estructural (basado en topología del grafo).
    Structural,
}

/// Generador de embeddings desde propiedades de nodos.
///
/// # Ejemplo
/// ```
/// use bikodb_ai::embed_gen::{EmbeddingGenerator, EmbedStrategy};
/// use bikodb_core::value::Value;
///
/// let gen = EmbeddingGenerator::new(8, EmbedStrategy::FeatureVector, vec![0, 1, 2]);
///
/// let props = vec![
///     (0u16, Value::Float(1.0)),
///     (1, Value::Float(2.0)),
///     (2, Value::Float(3.0)),
/// ];
/// let emb = gen.generate(&props);
/// assert_eq!(emb.len(), 8);
/// ```
pub struct EmbeddingGenerator {
    /// Dimensión de salida.
    pub output_dim: usize,
    /// Estrategia de generación.
    pub strategy: EmbedStrategy,
    /// IDs de propiedades a considerar.
    pub property_ids: Vec<u16>,
}

impl EmbeddingGenerator {
    /// Crea un generador de embeddings.
    pub fn new(output_dim: usize, strategy: EmbedStrategy, property_ids: Vec<u16>) -> Self {
        Self {
            output_dim,
            strategy,
            property_ids,
        }
    }

    /// Genera un embedding desde las propiedades de un nodo.
    pub fn generate(&self, properties: &[(u16, Value)]) -> Vec<f32> {
        match self.strategy {
            EmbedStrategy::FeatureVector => self.feature_vector(properties),
            EmbedStrategy::HashEmbedding => self.hash_embedding(properties),
            EmbedStrategy::Structural => self.structural_placeholder(properties),
        }
    }

    /// Genera embeddings para un lote de nodos.
    pub fn generate_batch(&self, nodes: &[Vec<(u16, Value)>]) -> Vec<Vec<f32>> {
        nodes.iter().map(|props| self.generate(props)).collect()
    }

    /// FeatureVector: extrae numéricas, pad/truncate, normaliza.
    fn feature_vector(&self, properties: &[(u16, Value)]) -> Vec<f32> {
        let prop_map: std::collections::HashMap<u16, &Value> =
            properties.iter().map(|(k, v)| (*k, v)).collect();

        let mut raw = Vec::new();
        for &pid in &self.property_ids {
            match prop_map.get(&pid) {
                Some(Value::Int(v)) => raw.push(*v as f32),
                Some(Value::Float(v)) => raw.push(*v as f32),
                Some(Value::Bool(v)) => raw.push(if *v { 1.0 } else { 0.0 }),
                Some(Value::Embedding(emb)) => raw.extend_from_slice(emb),
                _ => raw.push(0.0),
            }
        }

        // Pad or truncate to output_dim
        let mut embedding = vec![0.0; self.output_dim];
        let copy_len = raw.len().min(self.output_dim);
        embedding[..copy_len].copy_from_slice(&raw[..copy_len]);

        // L2 normalize
        l2_normalize(&mut embedding);

        embedding
    }

    /// HashEmbedding: feature hashing para propiedades mixtas.
    ///
    /// Cada propiedad se hashea a un bucket en el embedding vector.
    /// Soporta string, int, float, bool — todo se convierte a contribución numérica.
    fn hash_embedding(&self, properties: &[(u16, Value)]) -> Vec<f32> {
        let prop_map: std::collections::HashMap<u16, &Value> =
            properties.iter().map(|(k, v)| (*k, v)).collect();

        let mut embedding = vec![0.0; self.output_dim];
        let dim = self.output_dim;

        for &pid in &self.property_ids {
            if let Some(val) = prop_map.get(&pid) {
                match val {
                    Value::Int(v) => {
                        let bucket = hash_to_bucket(pid, dim);
                        embedding[bucket] += *v as f32;
                    }
                    Value::Float(v) => {
                        let bucket = hash_to_bucket(pid, dim);
                        embedding[bucket] += *v as f32;
                    }
                    Value::Bool(v) => {
                        let bucket = hash_to_bucket(pid, dim);
                        embedding[bucket] += if *v { 1.0 } else { -1.0 };
                    }
                    Value::String(s) => {
                        // Hash each character trigram
                        let bytes = s.as_bytes();
                        for window in bytes.windows(3.min(bytes.len()).max(1)) {
                            let bucket = hash_bytes_to_bucket(window, dim);
                            embedding[bucket] += 1.0;
                        }
                    }
                    Value::Embedding(emb) => {
                        for (i, &v) in emb.iter().enumerate() {
                            embedding[i % dim] += v;
                        }
                    }
                    _ => {} // Skip null, binary, etc.
                }
            }
        }

        // L2 normalize
        l2_normalize(&mut embedding);

        embedding
    }

    /// Structural embedding placeholder.
    /// In a full implementation, this would use node degree, centrality, etc.
    /// For now, creates a simple positional encoding from property values.
    fn structural_placeholder(&self, properties: &[(u16, Value)]) -> Vec<f32> {
        let mut embedding = vec![0.0; self.output_dim];

        // Use property count and value distribution as structural signal
        let num_props = properties.len() as f32;
        if self.output_dim > 0 {
            embedding[0] = num_props;
        }

        // Sum of numeric values as signal
        let mut sum = 0.0f32;
        for (_, val) in properties {
            match val {
                Value::Int(v) => sum += *v as f32,
                Value::Float(v) => sum += *v as f32,
                _ => {}
            }
        }
        if self.output_dim > 1 {
            embedding[1] = sum;
        }

        // Type diversity signal
        let mut type_bits = 0u32;
        for (_, val) in properties {
            type_bits |= match val {
                Value::Int(_) => 1,
                Value::Float(_) => 2,
                Value::String(_) => 4,
                Value::Bool(_) => 8,
                Value::Embedding(_) => 16,
                _ => 0,
            };
        }
        if self.output_dim > 2 {
            embedding[2] = type_bits as f32;
        }

        l2_normalize(&mut embedding);
        embedding
    }
}

/// Genera embedding estructural usando información de topología del grafo.
///
/// - `degree_in`: grado de entrada del nodo
/// - `degree_out`: grado de salida del nodo
/// - `neighbor_degrees`: grados de los vecinos (para clustering)
/// - `output_dim`: dimensión del embedding
pub fn structural_embedding(
    degree_in: usize,
    degree_out: usize,
    neighbor_degrees: &[usize],
    output_dim: usize,
) -> Vec<f32> {
    let mut embedding = vec![0.0; output_dim];

    if output_dim == 0 {
        return embedding;
    }

    // Feature 0: in-degree (normalized)
    embedding[0] = degree_in as f32;

    // Feature 1: out-degree
    if output_dim > 1 {
        embedding[1] = degree_out as f32;
    }

    // Feature 2: total degree
    if output_dim > 2 {
        embedding[2] = (degree_in + degree_out) as f32;
    }

    // Feature 3: average neighbor degree (local clustering signal)
    if output_dim > 3 && !neighbor_degrees.is_empty() {
        let avg: f32 = neighbor_degrees.iter().sum::<usize>() as f32 / neighbor_degrees.len() as f32;
        embedding[3] = avg;
    }

    // Feature 4: degree variance (heterogeneity of neighborhood)
    if output_dim > 4 && !neighbor_degrees.is_empty() {
        let avg: f32 = neighbor_degrees.iter().sum::<usize>() as f32 / neighbor_degrees.len() as f32;
        let variance: f32 = neighbor_degrees.iter()
            .map(|&d| { let diff = d as f32 - avg; diff * diff })
            .sum::<f32>() / neighbor_degrees.len() as f32;
        embedding[4] = variance.sqrt();
    }

    // Fill remaining dims with positional encoding (sinusoidal)
    let total_degree = (degree_in + degree_out) as f32;
    for i in 5..output_dim {
        let freq = (i as f32) / output_dim as f32;
        embedding[i] = (total_degree * freq * std::f32::consts::PI).sin();
    }

    l2_normalize(&mut embedding);
    embedding
}

/// L2 normalization in-place.
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Hash a property ID to a bucket index.
fn hash_to_bucket(prop_id: u16, dim: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    prop_id.hash(&mut hasher);
    hasher.finish() as usize % dim
}

/// Hash bytes to a bucket index.
fn hash_bytes_to_bucket(bytes: &[u8], dim: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish() as usize % dim
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_vector_basic() {
        let gen = EmbeddingGenerator::new(4, EmbedStrategy::FeatureVector, vec![0, 1]);
        let props = vec![
            (0u16, Value::Float(3.0)),
            (1, Value::Float(4.0)),
        ];
        let emb = gen.generate(&props);
        assert_eq!(emb.len(), 4);
        // Should be normalized: [3/5, 4/5, 0, 0]
        assert!((emb[0] - 0.6).abs() < 1e-5);
        assert!((emb[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_feature_vector_int() {
        let gen = EmbeddingGenerator::new(3, EmbedStrategy::FeatureVector, vec![0]);
        let props = vec![(0u16, Value::Int(5))];
        let emb = gen.generate(&props);
        assert_eq!(emb.len(), 3);
        // [5, 0, 0] normalized → [1, 0, 0]
        assert!((emb[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_feature_vector_bool() {
        let gen = EmbeddingGenerator::new(2, EmbedStrategy::FeatureVector, vec![0, 1]);
        let props = vec![
            (0u16, Value::Bool(true)),
            (1, Value::Bool(false)),
        ];
        let emb = gen.generate(&props);
        // [1.0, 0.0] normalized → [1.0, 0.0]
        assert!((emb[0] - 1.0).abs() < 1e-5);
        assert!((emb[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_feature_vector_missing_props() {
        let gen = EmbeddingGenerator::new(3, EmbedStrategy::FeatureVector, vec![0, 1, 2]);
        // Only provide prop 0
        let props = vec![(0u16, Value::Float(1.0))];
        let emb = gen.generate(&props);
        assert_eq!(emb.len(), 3);
        // [1, 0, 0] normalized → [1, 0, 0]
        assert!((emb[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hash_embedding_numeric() {
        let gen = EmbeddingGenerator::new(8, EmbedStrategy::HashEmbedding, vec![0, 1]);
        let props = vec![
            (0u16, Value::Int(10)),
            (1, Value::Float(20.0)),
        ];
        let emb = gen.generate(&props);
        assert_eq!(emb.len(), 8);
        // Should be normalized
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_hash_embedding_string() {
        let gen = EmbeddingGenerator::new(16, EmbedStrategy::HashEmbedding, vec![0]);
        let props = vec![(0u16, Value::from("hello"))];
        let emb = gen.generate(&props);
        assert_eq!(emb.len(), 16);
        // Non-zero (has string content)
        assert!(emb.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_structural_embedding() {
        let emb = structural_embedding(3, 5, &[2, 4, 6], 8);
        assert_eq!(emb.len(), 8);
        // Should be normalized
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_structural_embedding_isolated() {
        let emb = structural_embedding(0, 0, &[], 4);
        assert_eq!(emb.len(), 4);
        // All zeros remains all zeros (normalized)
        assert!(emb.iter().all(|&v| v.abs() < 1e-5));
    }

    #[test]
    fn test_generate_batch() {
        let gen = EmbeddingGenerator::new(4, EmbedStrategy::FeatureVector, vec![0]);
        let nodes = vec![
            vec![(0u16, Value::Float(1.0))],
            vec![(0u16, Value::Float(2.0))],
            vec![(0u16, Value::Float(3.0))],
        ];
        let batch = gen.generate_batch(&nodes);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].len(), 4);
    }

    #[test]
    fn test_structural_placeholder_strategy() {
        let gen = EmbeddingGenerator::new(6, EmbedStrategy::Structural, vec![0, 1]);
        let props = vec![
            (0u16, Value::Int(10)),
            (1, Value::Float(20.0)),
        ];
        let emb = gen.generate(&props);
        assert_eq!(emb.len(), 6);
    }

    #[test]
    fn test_l2_normalization() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        // Should remain zeros (no division by zero)
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_feature_vector_provider() {
        let provider = FeatureVectorProvider::new(4, vec![0, 1]);
        assert_eq!(provider.dimensions(), 4);
        assert_eq!(provider.name(), "feature_vector");

        let props = vec![(0u16, Value::Float(3.0)), (1, Value::Float(4.0))];
        let emb = provider.embed(&props);
        assert_eq!(emb.len(), 4);
        assert!((emb[0] - 0.6).abs() < 1e-5);
        assert!((emb[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_hash_embedding_provider() {
        let provider = HashEmbeddingProvider::new(8, vec![0]);
        assert_eq!(provider.dimensions(), 8);
        assert_eq!(provider.name(), "hash_embedding");

        let props = vec![(0u16, Value::from("hello"))];
        let emb = provider.embed(&props);
        assert_eq!(emb.len(), 8);
        assert!(emb.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_custom_embedding_provider() {
        struct ConstantProvider;
        impl EmbeddingProvider for ConstantProvider {
            fn embed(&self, _properties: &[(u16, Value)]) -> Vec<f32> {
                vec![0.5; 3]
            }
            fn dimensions(&self) -> usize { 3 }
            fn name(&self) -> &str { "constant" }
        }

        let provider = ConstantProvider;
        let emb = provider.embed(&[]);
        assert_eq!(emb, vec![0.5, 0.5, 0.5]);
        assert_eq!(provider.dimensions(), 3);
        assert_eq!(provider.name(), "constant");
    }
}
