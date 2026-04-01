// =============================================================================
// bikodb-storage::bucket — Colección de registros dentro de un archivo
// =============================================================================
// Un Bucket es un contenedor de registros del mismo tipo.
//
// Diseño:
// - Cada tipo (Vertex, Edge, Document) tiene N buckets (para paralelismo)
// - Dentro de un bucket: páginas de tamaño fijo con registros empaquetados
// - Direccionamiento: RID = bucket_id + offset_dentro_del_bucket
// - Auto-grow: nuevas páginas se asignan cuando las existentes están llenas
//
// Inspirado en ArcadeDB Bucket (multi-bucket per type, page-based storage).
// =============================================================================

use bikodb_core::error::{BikoError, BikoResult};
use bikodb_core::types::RID;
use std::collections::HashMap;

/// Un slot dentro de una página del bucket.
///
/// Cada slot almacena un registro serializado con su RID lógico.
#[derive(Debug, Clone)]
struct Slot {
    /// Offset dentro de la data de la página
    offset: u32,
    /// Tamaño del registro serializado
    size: u32,
    /// Indica si el slot fue eliminado (tombstone)
    deleted: bool,
}

/// Una página lógica dentro del bucket (almacena slots + data).
#[derive(Debug, Clone)]
struct BucketPage {
    /// Slots directory (tracked inline)
    slots: Vec<Slot>,
    /// Raw data de registros serializados
    data: Vec<u8>,
    /// Espacio total disponible
    capacity: usize,
}

impl BucketPage {
    fn new(capacity: usize) -> Self {
        Self {
            slots: Vec::new(),
            data: Vec::new(),
            capacity,
        }
    }

    /// Inserta un registro, retorna el índice del slot o None si no cabe.
    fn insert(&mut self, record_data: &[u8]) -> Option<u32> {
        let slot_overhead = 9; // offset(4) + size(4) + deleted(1)
        let needed = record_data.len() + slot_overhead;

        if self.data.len() + needed > self.capacity {
            return None; // No cabe, necesita nueva página
        }

        let offset = self.data.len() as u32;
        let size = record_data.len() as u32;

        self.data.extend_from_slice(record_data);
        let slot_idx = self.slots.len() as u32;
        self.slots.push(Slot {
            offset,
            size,
            deleted: false,
        });

        Some(slot_idx)
    }

    /// Lee un registro por su slot index.
    fn read(&self, slot_idx: u32) -> BikoResult<&[u8]> {
        let slot = self.slots.get(slot_idx as usize).ok_or_else(|| {
            BikoError::Generic(format!("slot {slot_idx}"))
        })?;

        if slot.deleted {
            return Err(BikoError::Generic(format!("slot {slot_idx} (deleted)")));
        }

        let start = slot.offset as usize;
        let end = start + slot.size as usize;
        Ok(&self.data[start..end])
    }

    /// Marca un slot como eliminado (tombstone).
    fn delete(&mut self, slot_idx: u32) -> BikoResult<()> {
        let slot = self.slots.get_mut(slot_idx as usize).ok_or_else(|| {
            BikoError::Generic(format!("slot {slot_idx}"))
        })?;
        slot.deleted = true;
        Ok(())
    }

    /// Actualiza un registro in-place si cabe, o marca error.
    fn update(&mut self, slot_idx: u32, new_data: &[u8]) -> BikoResult<()> {
        let slot = self.slots.get(slot_idx as usize).ok_or_else(|| {
            BikoError::Generic(format!("slot {slot_idx}"))
        })?;

        if slot.deleted {
            return Err(BikoError::Generic(format!("slot {slot_idx} (deleted)")));
        }

        let old_size = slot.size as usize;

        if new_data.len() <= old_size {
            // Cabe in-place
            let start = slot.offset as usize;
            self.data[start..start + new_data.len()].copy_from_slice(new_data);
            // Actualizar size
            self.slots[slot_idx as usize].size = new_data.len() as u32;
            Ok(())
        } else {
            // No cabe in-place: append al final de data y actualizar slot
            let new_offset = self.data.len() as u32;
            self.data.extend_from_slice(new_data);
            let slot = &mut self.slots[slot_idx as usize];
            slot.offset = new_offset;
            slot.size = new_data.len() as u32;
            Ok(())
        }
    }

    /// Número de slots activos (no eliminados).
    fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| !s.deleted).count()
    }
}

/// Bucket: contenedor de registros de un tipo.
///
/// Maneja múltiples páginas internamente. Cuando una página se llena,
/// se asigna una nueva automáticamente.
///
/// # Direccionamiento
/// El offset de un RID dentro de un bucket codifica: page_index * MAX_SLOTS + slot_index
///
/// # Ejemplo
/// ```
/// use bikodb_storage::bucket::Bucket;
///
/// let mut bucket = Bucket::new(0, 4096);
/// let rid = bucket.insert(b"record data").unwrap();
/// let data = bucket.read(&rid).unwrap();
/// assert_eq!(data, b"record data");
/// ```
pub struct Bucket {
    /// ID del bucket (maps to RID.bucket_id)
    id: u16,
    /// Páginas del bucket
    pages: Vec<BucketPage>,
    /// Capacidad de cada página
    page_capacity: usize,
    /// Índice de la próxima página con espacio
    current_page: usize,
}

/// Máximo de slots por página (para codificar page+slot en offset u48)
const MAX_SLOTS_PER_PAGE: u64 = 1 << 24; // 16M slots per page

impl Bucket {
    /// Crea un bucket con el ID dado y capacidad de página.
    pub fn new(id: u16, page_capacity: usize) -> Self {
        Self {
            id,
            pages: vec![BucketPage::new(page_capacity)],
            page_capacity,
            current_page: 0,
        }
    }

    /// Inserta un registro serializado, retorna su RID.
    pub fn insert(&mut self, data: &[u8]) -> BikoResult<RID> {
        // Intentar insertar en la página actual
        if let Some(slot_idx) = self.pages[self.current_page].insert(data) {
            let offset = self.current_page as u64 * MAX_SLOTS_PER_PAGE + slot_idx as u64;
            return Ok(RID {
                bucket_id: self.id,
                offset,
            });
        }

        // Página llena: buscar una con espacio o crear nueva
        let mut found = None;
        for (i, page) in self.pages.iter().enumerate() {
            if page.data.len() + data.len() + 9 <= page.capacity {
                found = Some(i);
                break;
            }
        }

        let page_idx = match found {
            Some(i) => i,
            None => {
                self.pages.push(BucketPage::new(self.page_capacity));
                self.pages.len() - 1
            }
        };

        self.current_page = page_idx;
        let slot_idx = self.pages[page_idx].insert(data).ok_or_else(|| {
            BikoError::Generic("Bucket: record too large for page".into())
        })?;

        let offset = page_idx as u64 * MAX_SLOTS_PER_PAGE + slot_idx as u64;
        Ok(RID {
            bucket_id: self.id,
            offset,
        })
    }

    /// Lee un registro por su RID.
    pub fn read(&self, rid: &RID) -> BikoResult<&[u8]> {
        let (page_idx, slot_idx) = self.decode_offset(rid.offset);
        let page = self.pages.get(page_idx).ok_or_else(|| {
            BikoError::RecordNotFound(*rid)
        })?;
        page.read(slot_idx)
    }

    /// Actualiza un registro existente.
    pub fn update(&mut self, rid: &RID, data: &[u8]) -> BikoResult<()> {
        let (page_idx, slot_idx) = self.decode_offset(rid.offset);
        let page = self.pages.get_mut(page_idx).ok_or_else(|| {
            BikoError::RecordNotFound(*rid)
        })?;
        page.update(slot_idx, data)
    }

    /// Elimina un registro (tombstone).
    pub fn delete(&mut self, rid: &RID) -> BikoResult<()> {
        let (page_idx, slot_idx) = self.decode_offset(rid.offset);
        let page = self.pages.get_mut(page_idx).ok_or_else(|| {
            BikoError::RecordNotFound(*rid)
        })?;
        page.delete(slot_idx)
    }

    /// Número total de registros activos.
    pub fn count(&self) -> usize {
        self.pages.iter().map(|p| p.active_count()).sum()
    }

    /// ID del bucket.
    pub fn id(&self) -> u16 {
        self.id
    }

    /// Decodifica offset → (page_index, slot_index)
    fn decode_offset(&self, offset: u64) -> (usize, u32) {
        let page_idx = (offset / MAX_SLOTS_PER_PAGE) as usize;
        let slot_idx = (offset % MAX_SLOTS_PER_PAGE) as u32;
        (page_idx, slot_idx)
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_insert_read() {
        let mut bucket = Bucket::new(0, 4096);
        let rid = bucket.insert(b"hello world").unwrap();
        let data = bucket.read(&rid).unwrap();
        assert_eq!(data, b"hello world");
    }

    #[test]
    fn test_bucket_multiple_inserts() {
        let mut bucket = Bucket::new(0, 4096);
        let r1 = bucket.insert(b"aaa").unwrap();
        let r2 = bucket.insert(b"bbb").unwrap();
        let r3 = bucket.insert(b"ccc").unwrap();

        assert_eq!(bucket.read(&r1).unwrap(), b"aaa");
        assert_eq!(bucket.read(&r2).unwrap(), b"bbb");
        assert_eq!(bucket.read(&r3).unwrap(), b"ccc");
        assert_eq!(bucket.count(), 3);
    }

    #[test]
    fn test_bucket_delete() {
        let mut bucket = Bucket::new(0, 4096);
        let rid = bucket.insert(b"to delete").unwrap();
        bucket.delete(&rid).unwrap();

        assert!(bucket.read(&rid).is_err());
        assert_eq!(bucket.count(), 0);
    }

    #[test]
    fn test_bucket_update_in_place() {
        let mut bucket = Bucket::new(0, 4096);
        let rid = bucket.insert(b"original data").unwrap();
        bucket.update(&rid, b"new data").unwrap(); // Más corto, cabe in-place

        let data = bucket.read(&rid).unwrap();
        assert_eq!(data, b"new data");
    }

    #[test]
    fn test_bucket_update_grow() {
        let mut bucket = Bucket::new(0, 4096);
        let rid = bucket.insert(b"short").unwrap();
        bucket.update(&rid, b"much longer data than before").unwrap();

        let data = bucket.read(&rid).unwrap();
        assert_eq!(data, b"much longer data than before");
    }

    #[test]
    fn test_bucket_page_overflow() {
        // Página muy pequeña para forzar overflow
        let mut bucket = Bucket::new(0, 64);
        let mut rids = Vec::new();

        for i in 0..10 {
            let data = format!("record_{i}");
            let rid = bucket.insert(data.as_bytes()).unwrap();
            rids.push(rid);
        }

        // Verificar que todos son legibles
        for (i, rid) in rids.iter().enumerate() {
            let data = bucket.read(rid).unwrap();
            assert_eq!(data, format!("record_{i}").as_bytes());
        }
    }

    #[test]
    fn test_bucket_rid_consistency() {
        let mut bucket = Bucket::new(7, 4096);
        let rid = bucket.insert(b"test").unwrap();
        assert_eq!(rid.bucket_id, 7);
    }
}
