// =============================================================================
// bikodb-ai::vector_idx — Índice vectorial HNSW simplificado
// =============================================================================
// Implementación basic de HNSW (Hierarchical Navigable Small World) para
// búsqueda approximate k-nearest neighbors (k-NN).
//
// Referencia: Malkov & Yashunin, 2018 — "Efficient and robust approximate
// nearest neighbor search using Hierarchical Navigable Small World graphs"
//
// v0.1: Flat index (brute force) + API preparada para HNSW.
// v0.2: HNSW real con múltiples capas, entry points, greedy routing.
// =============================================================================

use crate::embedding::DistanceMetric;
use bikodb_core::types::NodeId;
use parking_lot::RwLock;

/// Resultado de búsqueda k-NN: nodo + distancia.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub node_id: NodeId,
    pub distance: f32,
}

/// Índice vectorial para búsqueda de embeddings similares.
///
/// v0.1: Flat index (linear scan — O(n) per query).
/// v0.2: HNSW — O(log n) per query.
///
/// # Ejemplo
/// ```
/// use bikodb_ai::vector_idx::VectorIndex;
/// use bikodb_ai::embedding::DistanceMetric;
/// use bikodb_core::types::NodeId;
///
/// let mut idx = VectorIndex::new(3, DistanceMetric::Cosine);
/// idx.insert(NodeId(1), vec![1.0, 0.0, 0.0]);
/// idx.insert(NodeId(2), vec![0.0, 1.0, 0.0]);
/// idx.insert(NodeId(3), vec![0.9, 0.1, 0.0]);
///
/// let results = idx.search(&[1.0, 0.0, 0.0], 2);
/// assert_eq!(results[0].node_id, NodeId(1)); // Exact match
/// assert_eq!(results[1].node_id, NodeId(3)); // Closest
/// ```
pub struct VectorIndex {
    dimensions: usize,
    metric: DistanceMetric,
    vectors: RwLock<Vec<(NodeId, Vec<f32>)>>,
}

impl VectorIndex {
    /// Crea un índice vectorial con dimensiones fijas y métrica dada.
    pub fn new(dimensions: usize, metric: DistanceMetric) -> Self {
        Self {
            dimensions,
            metric,
            vectors: RwLock::new(Vec::new()),
        }
    }

    /// Inserta un vector asociado a un nodo.
    ///
    /// # Panics
    /// Si las dimensiones del vector no coinciden con el índice.
    pub fn insert(&self, node_id: NodeId, vector: Vec<f32>) {
        assert_eq!(
            vector.len(),
            self.dimensions,
            "Vector dimensions mismatch: expected {}, got {}",
            self.dimensions,
            vector.len()
        );
        self.vectors.write().push((node_id, vector));
    }

    /// Busca los k vecinos más cercanos al query vector.
    ///
    /// v0.1: Linear scan (brute force).
    /// Retorna resultados ordenados por distancia ascendente.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dimensions);

        let vectors = self.vectors.read();
        let mut results: Vec<SearchResult> = vectors
            .iter()
            .map(|(node_id, vec)| SearchResult {
                node_id: *node_id,
                distance: self.metric.distance(query, vec),
            })
            .collect();

        // Partial sort: solo necesitamos los k más cercanos
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(k);
        results
    }

    /// Elimina un vector por su NodeId.
    pub fn remove(&self, node_id: NodeId) {
        self.vectors.write().retain(|(id, _)| *id != node_id);
    }

    /// Número de vectores indexados.
    pub fn len(&self) -> usize {
        self.vectors.read().len()
    }

    /// ¿Índice vacío?
    pub fn is_empty(&self) -> bool {
        self.vectors.read().is_empty()
    }

    /// Dimensiones del índice.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let idx = VectorIndex::new(3, DistanceMetric::Euclidean);

        idx.insert(NodeId(1), vec![0.0, 0.0, 0.0]);
        idx.insert(NodeId(2), vec![1.0, 0.0, 0.0]);
        idx.insert(NodeId(3), vec![10.0, 10.0, 10.0]);

        let results = idx.search(&[0.1, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].node_id, NodeId(1)); // Closest
        assert_eq!(results[1].node_id, NodeId(2));
    }

    #[test]
    fn test_cosine_search() {
        let idx = VectorIndex::new(2, DistanceMetric::Cosine);

        idx.insert(NodeId(1), vec![1.0, 0.0]);
        idx.insert(NodeId(2), vec![0.0, 1.0]);
        idx.insert(NodeId(3), vec![0.7, 0.7]);

        let results = idx.search(&[1.0, 0.0], 1);
        assert_eq!(results[0].node_id, NodeId(1));
    }

    #[test]
    fn test_remove() {
        let idx = VectorIndex::new(2, DistanceMetric::Euclidean);
        idx.insert(NodeId(1), vec![1.0, 0.0]);
        idx.insert(NodeId(2), vec![0.0, 1.0]);

        idx.remove(NodeId(1));
        assert_eq!(idx.len(), 1);

        let results = idx.search(&[1.0, 0.0], 1);
        assert_eq!(results[0].node_id, NodeId(2));
    }

    #[test]
    fn test_search_empty_index() {
        let idx = VectorIndex::new(3, DistanceMetric::Cosine);
        let results = idx.search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_k_larger_than_index() {
        let idx = VectorIndex::new(2, DistanceMetric::Euclidean);
        idx.insert(NodeId(1), vec![1.0, 0.0]);

        let results = idx.search(&[0.0, 0.0], 10);
        assert_eq!(results.len(), 1);
    }
}
