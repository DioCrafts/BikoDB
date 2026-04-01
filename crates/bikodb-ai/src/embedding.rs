// =============================================================================
// bikodb-ai::embedding — Operaciones sobre embeddings vectoriales
// =============================================================================
// Funciones de distancia y utilidades para trabajar con embeddings.
// Los embeddings se almacenan como Vec<f32> en Value::Embedding.
// =============================================================================

/// Distancia coseno entre dos embeddings.
///
/// Retorna un valor entre 0.0 (idénticos) y 2.0 (opuestos).
/// Formula: 1 - (A·B / (|A| * |B|))
///
/// # Ejemplo
/// ```
/// use bikodb_ai::embedding::cosine_distance;
///
/// let a = vec![1.0, 0.0, 0.0];
/// let b = vec![1.0, 0.0, 0.0];
/// assert!((cosine_distance(&a, &b) - 0.0).abs() < 1e-6);
///
/// let c = vec![0.0, 1.0, 0.0];
/// assert!((cosine_distance(&a, &c) - 1.0).abs() < 1e-6);
/// ```
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Embedding dimensions must match");

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        return 1.0; // Degenerate case
    }

    1.0 - (dot / denom)
}

/// Distancia euclidiana (L2) entre dos embeddings.
///
/// # Ejemplo
/// ```
/// use bikodb_ai::embedding::euclidean_distance;
///
/// let a = vec![0.0, 0.0];
/// let b = vec![3.0, 4.0];
/// assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
/// ```
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Producto punto entre dos embeddings.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Normaliza un embedding a norma L2 = 1.0 (unit vector).
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Métrica de distancia soportada.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

impl DistanceMetric {
    /// Calcula distancia usando la métrica seleccionada.
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::Cosine => cosine_distance(a, b),
            Self::Euclidean => euclidean_distance(a, b),
            Self::DotProduct => -dot_product(a, b), // Negar para que menor = más similar
        }
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let d = cosine_distance(&a, &a);
        assert!(d.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let d = cosine_distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_euclidean() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0];
        let d = euclidean_distance(&a, &b);
        assert!((d - 3.0f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_distance_metric() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];

        let d1 = DistanceMetric::Cosine.distance(&a, &b);
        let d2 = DistanceMetric::Euclidean.distance(&a, &b);

        assert!((d1 - 1.0).abs() < 1e-5);
        assert!((d2 - 2.0f32.sqrt()).abs() < 1e-5);
    }
}
