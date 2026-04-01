// =============================================================================
// bikodb-storage::mmap — Abstracción sobre archivos mapeados en memoria
// =============================================================================
// Wrapper seguro y ergonómico sobre memmap2.
//
// Diseño:
// - MmapFile: archivo de acceso mmap con grow dinámico
// - read_page / write_page: operaciones alineadas a página
// - Sync a disco explícito (flush)
//
// Inspirado en ArcadeDB MMapManager (direct memory access, aligned pages).
// =============================================================================

use memmap2::{MmapMut, MmapOptions};
use bikodb_core::config::DEFAULT_PAGE_SIZE;
use bikodb_core::error::{BikoError, BikoResult};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

/// Archivo mapeado en memoria con operaciones por página.
///
/// Provee acceso directo a páginas de disco sin pasar por buffers del kernel.
/// El tamaño del archivo crece en incrementos de `page_size` cuando se necesita.
///
/// # Seguridad
/// - No expone punteros raw: todas las ops son slices acotados
/// - Flush explícito antes de descartar
pub struct MmapFile {
    path: PathBuf,
    file: File,
    mmap: Option<MmapMut>,
    page_size: usize,
    page_count: u32,
}

impl MmapFile {
    /// Abre o crea un archivo mmap.
    ///
    /// Si el archivo ya existe, lo abre y calcula el page_count.
    /// Si no existe, lo crea vacío.
    pub fn open(path: impl AsRef<Path>, page_size: usize) -> BikoResult<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let file_len = file.metadata()?.len() as usize;
        let page_count = if file_len == 0 {
            0
        } else {
            (file_len / page_size) as u32
        };

        let mmap = if file_len > 0 {
            let m = unsafe { MmapOptions::new().len(file_len).map_mut(&file)? };
            Some(m)
        } else {
            None
        };

        Ok(Self {
            path,
            file,
            mmap,
            page_size,
            page_count,
        })
    }

    /// Abre un archivo mmap con el page_size por defecto (64KB).
    pub fn open_default(path: impl AsRef<Path>) -> BikoResult<Self> {
        Self::open(path, DEFAULT_PAGE_SIZE)
    }

    /// Lee una página completa como slice de bytes.
    pub fn read_page(&self, page_number: u32) -> BikoResult<&[u8]> {
        if page_number >= self.page_count {
            return Err(BikoError::PageNotFound {
                file_id: 0,
                page_number,
            });
        }
        let mmap = self
            .mmap
            .as_ref()
            .ok_or(BikoError::PageNotFound { file_id: 0, page_number: 0 })?;
        let offset = page_number as usize * self.page_size;
        Ok(&mmap[offset..offset + self.page_size])
    }

    /// Escribe datos en una página. Los datos se truncan/padean al page_size.
    pub fn write_page(&mut self, page_number: u32, data: &[u8]) -> BikoResult<()> {
        // Crecer si es necesario
        while page_number >= self.page_count {
            self.grow_one_page()?;
        }

        let mmap = self
            .mmap
            .as_mut()
            .ok_or(BikoError::PageNotFound { file_id: 0, page_number: 0 })?;
        let offset = page_number as usize * self.page_size;
        let len = data.len().min(self.page_size);
        mmap[offset..offset + len].copy_from_slice(&data[..len]);

        // Zero-fill rest of page if data is shorter
        if len < self.page_size {
            mmap[offset + len..offset + self.page_size].fill(0);
        }

        Ok(())
    }

    /// Agrega una página nueva al final del archivo.
    /// Retorna el número de la nueva página.
    pub fn allocate_page(&mut self) -> BikoResult<u32> {
        let page_num = self.page_count;
        self.grow_one_page()?;
        Ok(page_num)
    }

    /// Crece el archivo en una página.
    fn grow_one_page(&mut self) -> BikoResult<()> {
        let new_len = (self.page_count as usize + 1) * self.page_size;
        self.file.set_len(new_len as u64)?;
        self.page_count += 1;

        // Re-map con el nuevo tamaño
        self.mmap = Some(unsafe { MmapOptions::new().len(new_len).map_mut(&self.file)? });

        Ok(())
    }

    /// Flush del mmap a disco.
    pub fn flush(&self) -> BikoResult<()> {
        if let Some(ref mmap) = self.mmap {
            mmap.flush()?;
        }
        Ok(())
    }

    /// Número de páginas actualmente en el archivo.
    pub fn page_count(&self) -> u32 {
        self.page_count
    }

    /// Tamaño de página configurado.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Ruta del archivo.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for MmapFile {
    fn drop(&mut self) {
        // Best-effort flush on drop
        let _ = self.flush();
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("bikodb_test_mmap");
        fs::create_dir_all(&dir).ok();
        dir.join(name)
    }

    #[test]
    fn test_create_and_write() {
        let path = temp_path("test_create.dat");
        let _ = fs::remove_file(&path);

        let mut mf = MmapFile::open(&path, 4096).unwrap();
        assert_eq!(mf.page_count(), 0);

        mf.write_page(0, &[0xAB; 4096]).unwrap();
        assert_eq!(mf.page_count(), 1);

        let data = mf.read_page(0).unwrap();
        assert_eq!(data[0], 0xAB);
        assert_eq!(data[4095], 0xAB);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_allocate_page() {
        let path = temp_path("test_alloc.dat");
        let _ = fs::remove_file(&path);

        let mut mf = MmapFile::open(&path, 4096).unwrap();
        let p0 = mf.allocate_page().unwrap();
        let p1 = mf.allocate_page().unwrap();

        assert_eq!(p0, 0);
        assert_eq!(p1, 1);
        assert_eq!(mf.page_count(), 2);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_nonexistent_page() {
        let path = temp_path("test_nopage.dat");
        let _ = fs::remove_file(&path);

        let mf = MmapFile::open(&path, 4096).unwrap();
        assert!(mf.read_page(0).is_err());

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_flush() {
        let path = temp_path("test_flush.dat");
        let _ = fs::remove_file(&path);

        let mut mf = MmapFile::open(&path, 4096).unwrap();
        mf.write_page(0, &[1; 4096]).unwrap();
        mf.flush().unwrap();

        // Re-open and verify persistence
        drop(mf);
        let mf2 = MmapFile::open(&path, 4096).unwrap();
        let data = mf2.read_page(0).unwrap();
        assert_eq!(data[0], 1);

        let _ = fs::remove_file(&path);
    }
}
