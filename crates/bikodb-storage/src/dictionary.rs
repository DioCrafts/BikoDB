// =============================================================================
// bikodb-storage::dictionary — Mapeo bidireccional String ↔ u16
// =============================================================================
// Compresión de nombres de propiedades y tipos a IDs numéricos de 16 bits.
//
// Diseño:
// - Lookup bidireccional O(1) con HashMap + Vec
// - Capacidad máxima 65535 nombres (u16::MAX)
// - Thread-safe con parking_lot::RwLock
//
// Inspirado en ArcadeDB Dictionary (string interning for property names).
// =============================================================================

use parking_lot::RwLock;
use std::collections::HashMap;

/// Diccionario bidireccional para internar strings como u16.
///
/// Cada nombre de propiedad/tipo se registra una vez y recibe un ID numérico.
/// Esto reduce el tamaño de registros en disco (2 bytes vs N bytes por nombre).
///
/// # Thread Safety
/// Protegido por RwLock: múltiples readers simultáneos, un writer.
///
/// # Ejemplo
/// ```
/// use bikodb_storage::dictionary::Dictionary;
///
/// let dict = Dictionary::new();
/// let id = dict.get_or_insert("name");
/// assert_eq!(dict.resolve(id), Some("name".to_string()));
/// assert_eq!(dict.get_or_insert("name"), id); // Mismo ID
/// ```
pub struct Dictionary {
    inner: RwLock<DictionaryInner>,
}

struct DictionaryInner {
    /// name → id
    name_to_id: HashMap<String, u16>,
    /// id → name (vec index = id)
    id_to_name: Vec<String>,
}

impl Dictionary {
    /// Crea un diccionario vacío.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(DictionaryInner {
                name_to_id: HashMap::new(),
                id_to_name: Vec::new(),
            }),
        }
    }

    /// Obtiene el ID de un nombre, insertándolo si no existe.
    ///
    /// # Panics
    /// Panics si se exceden 65535 nombres.
    pub fn get_or_insert(&self, name: &str) -> u16 {
        // Fast path: read lock
        {
            let inner = self.inner.read();
            if let Some(&id) = inner.name_to_id.get(name) {
                return id;
            }
        }

        // Slow path: write lock para insertar
        let mut inner = self.inner.write();
        // Double-check (otro thread pudo insertar entre read y write)
        if let Some(&id) = inner.name_to_id.get(name) {
            return id;
        }

        let id = inner.id_to_name.len() as u16;
        assert!(
            (id as usize) < u16::MAX as usize,
            "Dictionary: exceeded max entries (65535)"
        );

        inner.id_to_name.push(name.to_string());
        inner.name_to_id.insert(name.to_string(), id);
        id
    }

    /// Busca el ID de un nombre sin insertarlo.
    pub fn lookup(&self, name: &str) -> Option<u16> {
        self.inner.read().name_to_id.get(name).copied()
    }

    /// Resuelve un ID a su nombre.
    pub fn resolve(&self, id: u16) -> Option<String> {
        let inner = self.inner.read();
        inner.id_to_name.get(id as usize).cloned()
    }

    /// Número de entradas en el diccionario.
    pub fn len(&self) -> usize {
        self.inner.read().id_to_name.len()
    }

    /// ¿Diccionario vacío?
    pub fn is_empty(&self) -> bool {
        self.inner.read().id_to_name.is_empty()
    }

    /// Serializa el diccionario a bytes (para persistencia).
    ///
    /// Formato: [count:u16]([len:u16][utf8 bytes])*
    pub fn to_bytes(&self) -> Vec<u8> {
        let inner = self.inner.read();
        let mut buf = Vec::new();
        let count = inner.id_to_name.len() as u16;
        buf.extend_from_slice(&count.to_le_bytes());

        for name in &inner.id_to_name {
            let bytes = name.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
        buf
    }

    /// Reconstruye un diccionario desde bytes serializados.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 2 {
            return None;
        }

        let count = u16::from_le_bytes([data[0], data[1]]) as usize;
        let mut offset = 2;
        let mut id_to_name = Vec::with_capacity(count);
        let mut name_to_id = HashMap::with_capacity(count);

        for i in 0..count {
            if offset + 2 > data.len() {
                return None;
            }
            let len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;

            if offset + len > data.len() {
                return None;
            }
            let name = std::str::from_utf8(&data[offset..offset + len]).ok()?;
            offset += len;

            id_to_name.push(name.to_string());
            name_to_id.insert(name.to_string(), i as u16);
        }

        Some(Self {
            inner: RwLock::new(DictionaryInner {
                name_to_id,
                id_to_name,
            }),
        })
    }
}

impl Default for Dictionary {
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
    fn test_insert_and_resolve() {
        let dict = Dictionary::new();
        let id = dict.get_or_insert("name");
        assert_eq!(id, 0);
        assert_eq!(dict.resolve(id), Some("name".to_string()));
    }

    #[test]
    fn test_idempotent_insert() {
        let dict = Dictionary::new();
        let id1 = dict.get_or_insert("age");
        let id2 = dict.get_or_insert("age");
        assert_eq!(id1, id2);
        assert_eq!(dict.len(), 1);
    }

    #[test]
    fn test_multiple_entries() {
        let dict = Dictionary::new();
        let a = dict.get_or_insert("a");
        let b = dict.get_or_insert("b");
        let c = dict.get_or_insert("c");

        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_eq!(dict.len(), 3);
        assert_eq!(dict.resolve(a), Some("a".into()));
        assert_eq!(dict.resolve(c), Some("c".into()));
    }

    #[test]
    fn test_lookup() {
        let dict = Dictionary::new();
        assert_eq!(dict.lookup("x"), None);
        dict.get_or_insert("x");
        assert!(dict.lookup("x").is_some());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let dict = Dictionary::new();
        dict.get_or_insert("name");
        dict.get_or_insert("age");
        dict.get_or_insert("email");

        let bytes = dict.to_bytes();
        let dict2 = Dictionary::from_bytes(&bytes).unwrap();

        assert_eq!(dict2.len(), 3);
        assert_eq!(dict2.resolve(0), Some("name".into()));
        assert_eq!(dict2.resolve(1), Some("age".into()));
        assert_eq!(dict2.resolve(2), Some("email".into()));
    }

    #[test]
    fn test_from_bytes_invalid() {
        assert!(Dictionary::from_bytes(&[]).is_none());
        assert!(Dictionary::from_bytes(&[0xFF]).is_none());
    }
}
