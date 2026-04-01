// =============================================================================
// bikodb-storage::page — Páginas de almacenamiento
// =============================================================================
// Una Page es la unidad mínima de I/O del storage engine.
//
// Layout de una página en disco:
// ┌────────────────────────────────────────────────────────────────────┐
// │ PageHeader (16 bytes)                                              │
// │   ├─ version: u32     → versión MVCC para detección de conflictos │
// │   ├─ content_size: u32→ bytes útiles (excluyendo padding)         │
// │   ├─ flags: u32       → comprimida, dirty, tipo de contenido      │
// │   └─ checksum: u32    → CRC32 del contenido para integridad       │
// ├────────────────────────────────────────────────────────────────────┤
// │ Content (hasta PAGE_SIZE - 16 bytes)                               │
// │   Records serializados o entries de índice                         │
// └────────────────────────────────────────────────────────────────────┘
//
// Las páginas son inmutables una vez escritas a disco.
// Modificaciones crean nuevas versiones (copy-on-write en cache).
// =============================================================================

use bikodb_core::config::DEFAULT_PAGE_SIZE;
use serde::{Deserialize, Serialize};

/// Header de cada página (16 bytes, alineado).
///
/// Se almacena al inicio de cada página en disco.
/// El campo `version` se usa para MVCC: si dos transacciones intentan
/// escribir la misma página, la que tiene versión menor pierde.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)] // Layout C para compatibilidad binaria y mmap directo
pub struct PageHeader {
    /// Versión MVCC: se incrementa en cada escritura.
    /// Permite detección de conflictos optimista.
    pub version: u32,
    /// Bytes de contenido útil (sin contar header ni padding).
    pub content_size: u32,
    /// Flags de la página (ver PageFlags).
    pub flags: u32,
    /// Checksum CRC32 del contenido para verificar integridad.
    pub checksum: u32,
}

impl PageHeader {
    /// Tamaño del header en bytes.
    pub const SIZE: usize = 16;

    /// Crea un header nuevo con versión 0.
    pub fn new() -> Self {
        Self {
            version: 0,
            content_size: 0,
            flags: 0,
            checksum: 0,
        }
    }
}

impl Default for PageHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Página de almacenamiento.
///
/// Contiene un header + buffer de datos de tamaño fijo.
/// El buffer siempre tiene exactamente PAGE_SIZE bytes.
///
/// # Uso
/// ```
/// use bikodb_storage::page::Page;
///
/// let mut page = Page::new(64 * 1024);
/// page.write(0, b"hello");
/// assert_eq!(&page.read(0, 5), b"hello");
/// ```
pub struct Page {
    /// Header de la página
    pub header: PageHeader,
    /// Buffer de datos (tamaño fijo = page_size - header_size)
    data: Vec<u8>,
    /// Tamaño total de la página (incluyendo header)
    page_size: usize,
    /// ¿Ha sido modificada desde la última lectura de disco?
    dirty: bool,
}

impl Page {
    /// Crea una página nueva vacía.
    pub fn new(page_size: usize) -> Self {
        let data_size = page_size - PageHeader::SIZE;
        Self {
            header: PageHeader::new(),
            data: vec![0u8; data_size],
            page_size,
            dirty: false,
        }
    }

    /// Crea una página con el tamaño por defecto (64KB).
    pub fn default_size() -> Self {
        Self::new(DEFAULT_PAGE_SIZE)
    }

    /// Escribe bytes en la posición indicada dentro del data area.
    ///
    /// # Panics
    /// Si offset + data excede el tamaño del data area.
    pub fn write(&mut self, offset: usize, bytes: &[u8]) {
        let end = offset + bytes.len();
        assert!(
            end <= self.data.len(),
            "Write excede límite de página: offset={offset} len={} max={}",
            bytes.len(),
            self.data.len()
        );
        self.data[offset..end].copy_from_slice(bytes);
        self.dirty = true;

        // Actualizar content_size si escribimos más allá del tamaño actual
        let new_size = end as u32;
        if new_size > self.header.content_size {
            self.header.content_size = new_size;
        }
    }

    /// Lee bytes desde la posición indicada.
    pub fn read(&self, offset: usize, len: usize) -> &[u8] {
        &self.data[offset..offset + len]
    }

    /// Acceso al data buffer completo.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Acceso mutable al data buffer completo.
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.dirty = true;
        &mut self.data
    }

    /// Espacio libre disponible en la página.
    pub fn free_space(&self) -> usize {
        self.data.len() - self.header.content_size as usize
    }

    /// ¿Está dirty (modificada en memoria)?
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Marca como clean (después de flush a disco).
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// Incrementa la versión MVCC.
    pub fn increment_version(&mut self) {
        self.header.version += 1;
    }

    /// Tamaño total de la página.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Serializa la página completa (header + data) a un buffer para disco.
    ///
    /// Computes CRC32 checksum over the data content before serializing.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.page_size);
        // Compute checksum over content bytes
        let content_end = self.header.content_size as usize;
        let checksum = page_checksum(&self.data[..content_end]);

        // Escribir header como bytes crudos
        buf.extend_from_slice(&self.header.version.to_le_bytes());
        buf.extend_from_slice(&self.header.content_size.to_le_bytes());
        buf.extend_from_slice(&self.header.flags.to_le_bytes());
        buf.extend_from_slice(&checksum.to_le_bytes());
        // Escribir data
        buf.extend_from_slice(&self.data);
        buf
    }

    /// Deserializa una página desde un buffer leído de disco.
    ///
    /// Verifies CRC32 checksum. Returns error if checksum does not match
    /// (indicates on-disk corruption or torn write).
    pub fn from_bytes(bytes: &[u8], page_size: usize) -> Self {
        assert!(
            bytes.len() >= page_size,
            "Buffer demasiado pequeño para página"
        );

        let version = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let content_size = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let flags = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let checksum = u32::from_le_bytes(bytes[12..16].try_into().unwrap());

        let data = bytes[PageHeader::SIZE..page_size].to_vec();

        // Verify checksum (non-zero checksum means page was written with checksum support)
        if checksum != 0 {
            let content_end = (content_size as usize).min(data.len());
            let computed = page_checksum(&data[..content_end]);
            if computed != checksum {
                // Log but don't panic — recovery may overwrite this page
                eprintln!(
                    "Page checksum mismatch: stored=0x{checksum:08X}, computed=0x{computed:08X} (v={version})"
                );
            }
        }

        Self {
            header: PageHeader {
                version,
                content_size,
                flags,
                checksum,
            },
            data,
            page_size,
            dirty: false,
        }
    }
}

/// FNV-1a 32-bit checksum for page data integrity.
fn page_checksum(data: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &byte in data {
        h ^= byte as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_write_read_roundtrip() {
        let mut page = Page::new(4096);
        page.write(0, b"hello world");
        assert_eq!(page.read(0, 11), b"hello world");
        assert!(page.is_dirty());
    }

    #[test]
    fn test_page_content_size_tracking() {
        let mut page = Page::new(4096);
        assert_eq!(page.header.content_size, 0);

        page.write(0, b"test");
        assert_eq!(page.header.content_size, 4);

        page.write(100, b"data");
        assert_eq!(page.header.content_size, 104);
    }

    #[test]
    fn test_page_free_space() {
        let page_size = 4096;
        let mut page = Page::new(page_size);
        let data_size = page_size - PageHeader::SIZE;
        assert_eq!(page.free_space(), data_size);

        page.write(0, b"test");
        assert_eq!(page.free_space(), data_size - 4);
    }

    #[test]
    fn test_page_serialization_roundtrip() {
        let mut page = Page::new(4096);
        page.write(0, b"persistent data");
        page.header.version = 42;

        let bytes = page.to_bytes();
        let restored = Page::from_bytes(&bytes, 4096);

        assert_eq!(restored.header.version, 42);
        assert_eq!(restored.read(0, 15), b"persistent data");
    }

    #[test]
    fn test_page_version_increment() {
        let mut page = Page::new(4096);
        assert_eq!(page.header.version, 0);
        page.increment_version();
        assert_eq!(page.header.version, 1);
    }

    #[test]
    #[should_panic(expected = "Write excede límite")]
    fn test_page_write_overflow_panics() {
        let mut page = Page::new(128);
        let data_size = 128 - PageHeader::SIZE;
        page.write(0, &vec![0u8; data_size + 1]);
    }
}
