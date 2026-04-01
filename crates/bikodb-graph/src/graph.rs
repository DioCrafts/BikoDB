// =============================================================================
// bikodb-graph::graph — Grafo concurrente in-memory
// =============================================================================
// Estructura principal del grafo: almacena nodos y edges con adjacency lists.
//
// Diseño inspirado en:
// - ArcadeDB: MutableVertex, MutableEdge, edge buckets
// - Neo4j: doubly-linked relationship chains per node
//
// Diferencias BikoDB:
// - Lock-free reads con DashMap (vs RWLock global en otros engines)
// - SmallVec<4> para edges inline → cache-friendly para powerlaw graphs
// - TypeId en cada nodo/edge para schema enforcement
// =============================================================================

use dashmap::DashMap;
use bikodb_core::error::{BikoError, BikoResult};
use bikodb_core::record::{Direction, Vertex};
use bikodb_core::types::{EdgeId, NodeId, TypeId};
use bikodb_core::value::Value;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicU64, Ordering};

/// Información de un edge almacenado en el grafo.
#[derive(Debug, Clone)]
pub struct EdgeData {
    pub id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub type_id: TypeId,
    pub properties: Vec<(u16, Value)>,
}

impl EdgeData {
    /// Estimated memory usage in bytes (stack + heap).
    pub fn estimated_bytes(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let props_heap = self.properties.capacity()
            * std::mem::size_of::<(u16, Value)>()
            + self.properties.iter().map(|(_, v)| v.estimated_heap_bytes()).sum::<usize>();
        base + props_heap
    }
}

/// Información de un nodo almacenado en el grafo.
///
/// Optimización de memoria:
/// - SmallVec<[EdgeId; 4]> para nodos con ≤4 edges inline (32 bytes en stack)
///   vs SmallVec<[EdgeId; 8]> (64 bytes). La mayoría de nodos en grafos
///   power-law tienen ≤4 edges, así que esto cubre el caso común sin heap.
/// - Properties como Vec<(u16, Value)> con Value compacto (≤32 bytes)
#[derive(Debug, Clone)]
pub struct NodeData {
    pub id: NodeId,
    pub type_id: TypeId,
    pub properties: Vec<(u16, Value)>,
    /// Edges salientes: SmallVec con 4 inline (cache-friendly, 32B)
    pub out_edges: SmallVec<[EdgeId; 4]>,
    /// Edges entrantes
    pub in_edges: SmallVec<[EdgeId; 4]>,
}

impl NodeData {
    pub fn new(id: NodeId, type_id: TypeId) -> Self {
        Self {
            id,
            type_id,
            properties: Vec::new(),
            out_edges: SmallVec::new(),
            in_edges: SmallVec::new(),
        }
    }

    /// Degree (in + out).
    pub fn degree(&self) -> usize {
        self.out_edges.len() + self.in_edges.len()
    }

    /// Estimated memory usage in bytes (stack + heap).
    pub fn estimated_bytes(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let props_heap = self.properties.capacity()
            * std::mem::size_of::<(u16, Value)>()
            + self.properties.iter().map(|(_, v)| v.estimated_heap_bytes()).sum::<usize>();
        // SmallVec spills to heap when > inline capacity
        let out_heap = if self.out_edges.spilled() {
            self.out_edges.capacity() * std::mem::size_of::<EdgeId>()
        } else {
            0
        };
        let in_heap = if self.in_edges.spilled() {
            self.in_edges.capacity() * std::mem::size_of::<EdgeId>()
        } else {
            0
        };
        base + props_heap + out_heap + in_heap
    }
}

/// Grafo concurrente con adjacency lists.
///
/// Los nodos y edges se almacenan en DashMaps para acceso lock-free.
/// Cada nodo mantiene listas de edges salientes/entrantes (SmallVec<4>).
///
/// # Concurrencia
/// - Reads por nodo/edge: lock-free (DashMap sharding)
/// - Writes: lock per shard (~1/64 del grafo)
/// - IDs: generados atómicamente (monotónicos)
///
/// # Ejemplo
/// ```
/// use bikodb_graph::ConcurrentGraph;
/// use bikodb_core::types::TypeId;
///
/// let graph = ConcurrentGraph::new();
/// let n1 = graph.insert_node(TypeId(1));
/// let n2 = graph.insert_node(TypeId(1));
/// let e = graph.insert_edge(n1, n2, TypeId(10)).unwrap();
///
/// assert_eq!(graph.node_count(), 2);
/// assert_eq!(graph.edge_count(), 1);
/// ```
pub struct ConcurrentGraph {
    nodes: DashMap<NodeId, NodeData>,
    edges: DashMap<EdgeId, EdgeData>,
    next_node_id: AtomicU64,
    next_edge_id: AtomicU64,
}

impl ConcurrentGraph {
    /// Crea un grafo vacío.
    pub fn new() -> Self {
        Self {
            nodes: DashMap::new(),
            edges: DashMap::new(),
            next_node_id: AtomicU64::new(1),
            next_edge_id: AtomicU64::new(1),
        }
    }

    /// Crea un grafo con capacidad pre-alocada.
    pub fn with_capacity(node_cap: usize, edge_cap: usize) -> Self {
        Self {
            nodes: DashMap::with_capacity(node_cap),
            edges: DashMap::with_capacity(edge_cap),
            next_node_id: AtomicU64::new(1),
            next_edge_id: AtomicU64::new(1),
        }
    }

    // ── Node operations ─────────────────────────────────────────────────

    /// Inserta un nodo nuevo con el tipo dado. Retorna su NodeId.
    pub fn insert_node(&self, type_id: TypeId) -> NodeId {
        let id = NodeId(self.next_node_id.fetch_add(1, Ordering::Relaxed));
        self.nodes.insert(id, NodeData::new(id, type_id));
        id
    }

    /// Inserta un nodo con propiedades iniciales.
    pub fn insert_node_with_props(
        &self,
        type_id: TypeId,
        properties: Vec<(u16, Value)>,
    ) -> NodeId {
        let id = NodeId(self.next_node_id.fetch_add(1, Ordering::Relaxed));
        let mut data = NodeData::new(id, type_id);
        data.properties = properties;
        self.nodes.insert(id, data);
        id
    }

    /// Inserta un nodo con datos completos y un NodeId específico.
    ///
    /// Usado internamente por el cluster para rebalanceo y migración de nodos.
    pub fn insert_node_data(&self, id: NodeId, data: NodeData) {
        // Advance the counter past this ID to avoid collisions
        let _ = self.next_node_id.fetch_max(id.0 + 1, Ordering::Relaxed);
        self.nodes.insert(id, data);
    }

    /// Obtiene datos de un nodo (clone para evitar data races).
    pub fn get_node(&self, id: NodeId) -> Option<NodeData> {
        self.nodes.get(&id).map(|r| r.clone())
    }

    /// ¿Existe el nodo?
    pub fn contains_node(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    /// Elimina un nodo y todos sus edges asociados.
    pub fn remove_node(&self, id: NodeId) -> BikoResult<()> {
        let node = self.nodes.remove(&id).ok_or(BikoError::NodeNotFound(id))?;

        // Eliminar todos los edges salientes
        for edge_id in &node.1.out_edges {
            if let Some((_, edge)) = self.edges.remove(edge_id) {
                // Quitar este edge de la lista in_edges del target
                if let Some(mut target) = self.nodes.get_mut(&edge.target) {
                    target.in_edges.retain(|e| e != edge_id);
                }
            }
        }

        // Eliminar todos los edges entrantes
        for edge_id in &node.1.in_edges {
            if let Some((_, edge)) = self.edges.remove(edge_id) {
                // Quitar este edge de la lista out_edges del source
                if let Some(mut source) = self.nodes.get_mut(&edge.source) {
                    source.out_edges.retain(|e| e != edge_id);
                }
            }
        }

        Ok(())
    }

    /// Establece una propiedad en un nodo.
    pub fn set_node_property(&self, id: NodeId, prop_id: u16, value: Value) -> BikoResult<()> {
        let mut node = self.nodes.get_mut(&id).ok_or(BikoError::NodeNotFound(id))?;

        if let Some(entry) = node.properties.iter_mut().find(|(k, _)| *k == prop_id) {
            entry.1 = value;
        } else {
            node.properties.push((prop_id, value));
        }
        Ok(())
    }

    // ── Edge operations ─────────────────────────────────────────────────

    /// Inserta un edge entre source y target. Retorna su EdgeId.
    ///
    /// Garantiza atomicidad: si source o target no existen después de la
    /// validación, el edge se limpia para evitar estado inconsistente.
    pub fn insert_edge(
        &self,
        source: NodeId,
        target: NodeId,
        type_id: TypeId,
    ) -> BikoResult<EdgeId> {
        // Verificar que ambos nodos existen
        if !self.nodes.contains_key(&source) {
            return Err(BikoError::NodeNotFound(source));
        }
        if !self.nodes.contains_key(&target) {
            return Err(BikoError::NodeNotFound(target));
        }

        let id = EdgeId(self.next_edge_id.fetch_add(1, Ordering::Relaxed));
        let edge = EdgeData {
            id,
            source,
            target,
            type_id,
            properties: Vec::new(),
        };
        self.edges.insert(id, edge);

        // Actualizar adjacency lists con cleanup en caso de nodo eliminado
        // entre la validación y el update de adjacency.
        let src_ok = if let Some(mut src) = self.nodes.get_mut(&source) {
            src.out_edges.push(id);
            true
        } else {
            false
        };

        let tgt_ok = if let Some(mut tgt) = self.nodes.get_mut(&target) {
            tgt.in_edges.push(id);
            true
        } else {
            false
        };

        // Cleanup: si algún nodo desapareció, revertir estado parcial
        if !src_ok || !tgt_ok {
            self.edges.remove(&id);
            if src_ok {
                if let Some(mut src) = self.nodes.get_mut(&source) {
                    src.out_edges.retain(|e| *e != id);
                }
            }
            if tgt_ok {
                if let Some(mut tgt) = self.nodes.get_mut(&target) {
                    tgt.in_edges.retain(|e| *e != id);
                }
            }
            return Err(BikoError::NodeNotFound(if !src_ok { source } else { target }));
        }

        Ok(id)
    }

    /// Inserta un edge con propiedades.
    pub fn insert_edge_with_props(
        &self,
        source: NodeId,
        target: NodeId,
        type_id: TypeId,
        properties: Vec<(u16, Value)>,
    ) -> BikoResult<EdgeId> {
        let id = self.insert_edge(source, target, type_id)?;
        if let Some(mut edge) = self.edges.get_mut(&id) {
            edge.properties = properties;
        }
        Ok(id)
    }

    /// Obtiene datos de un edge.
    pub fn get_edge(&self, id: EdgeId) -> Option<EdgeData> {
        self.edges.get(&id).map(|r| r.clone())
    }

    /// Elimina un edge.
    pub fn remove_edge(&self, id: EdgeId) -> BikoResult<()> {
        let edge = self.edges.remove(&id).ok_or(BikoError::Generic(
            format!("Edge {:?} not found", id),
        ))?;

        if let Some(mut src) = self.nodes.get_mut(&edge.1.source) {
            src.out_edges.retain(|e| *e != id);
        }
        if let Some(mut tgt) = self.nodes.get_mut(&edge.1.target) {
            tgt.in_edges.retain(|e| *e != id);
        }

        Ok(())
    }

    // ── Adjacency queries ───────────────────────────────────────────────

    /// Vecinos de un nodo en la dirección dada.
    pub fn neighbors(&self, id: NodeId, direction: Direction) -> BikoResult<Vec<NodeId>> {
        let node = self.nodes.get(&id).ok_or(BikoError::NodeNotFound(id))?;

        let mut result = Vec::new();

        if matches!(direction, Direction::Out | Direction::Both) {
            for edge_id in &node.out_edges {
                if let Some(edge) = self.edges.get(edge_id) {
                    result.push(edge.target);
                }
            }
        }

        if matches!(direction, Direction::In | Direction::Both) {
            for edge_id in &node.in_edges {
                if let Some(edge) = self.edges.get(edge_id) {
                    result.push(edge.source);
                }
            }
        }

        Ok(result)
    }

    /// Vecinos filtrados por tipo de edge.
    pub fn neighbors_by_type(
        &self,
        id: NodeId,
        direction: Direction,
        edge_type: TypeId,
    ) -> BikoResult<Vec<NodeId>> {
        let node = self.nodes.get(&id).ok_or(BikoError::NodeNotFound(id))?;

        let mut result = Vec::new();

        if matches!(direction, Direction::Out | Direction::Both) {
            for edge_id in &node.out_edges {
                if let Some(edge) = self.edges.get(edge_id) {
                    if edge.type_id == edge_type {
                        result.push(edge.target);
                    }
                }
            }
        }

        if matches!(direction, Direction::In | Direction::Both) {
            for edge_id in &node.in_edges {
                if let Some(edge) = self.edges.get(edge_id) {
                    if edge.type_id == edge_type {
                        result.push(edge.source);
                    }
                }
            }
        }

        Ok(result)
    }

    /// Edges de un nodo en la dirección dada.
    pub fn edges_of(&self, id: NodeId, direction: Direction) -> BikoResult<Vec<EdgeData>> {
        let node = self.nodes.get(&id).ok_or(BikoError::NodeNotFound(id))?;

        let mut result = Vec::new();

        if matches!(direction, Direction::Out | Direction::Both) {
            for edge_id in &node.out_edges {
                if let Some(edge) = self.edges.get(edge_id) {
                    result.push(edge.clone());
                }
            }
        }

        if matches!(direction, Direction::In | Direction::Both) {
            for edge_id in &node.in_edges {
                if let Some(edge) = self.edges.get(edge_id) {
                    result.push(edge.clone());
                }
            }
        }

        Ok(result)
    }

    // ── Counts ──────────────────────────────────────────────────────────

    /// Número de nodos.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Número de edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Itera sobre todos los nodos, invocando el callback con (NodeId, &NodeData).
    /// Útil para construir representaciones derivadas (CSR, etc.).
    pub fn iter_nodes(&self, mut f: impl FnMut(NodeId, &NodeData)) {
        for entry in self.nodes.iter() {
            f(*entry.key(), entry.value());
        }
    }

    /// Itera sobre todos los edges, invocando el callback con (EdgeId, &EdgeData).
    pub fn iter_edges(&self, mut f: impl FnMut(EdgeId, &EdgeData)) {
        for entry in self.edges.iter() {
            f(*entry.key(), entry.value());
        }
    }

    /// Collects all node IDs matching `type_id` using a parallel scan.
    ///
    /// Uses Rayon to parallelize the DashMap scan across shards.
    /// Returns `(NodeId, properties)` tuples for matching nodes.
    pub fn par_scan_nodes_by_type(
        &self,
        type_id: TypeId,
    ) -> Vec<(NodeId, Vec<(u16, Value)>)> {
        // Collect shard entries into a Vec, then par-filter.
        // DashMap::iter() iterates shard-by-shard; we collect keys first
        // to enable rayon parallelism on the filter + clone step.
        let candidates: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|e| e.value().type_id == type_id)
            .map(|e| *e.key())
            .collect();

        candidates
            .into_par_iter()
            .filter_map(|id| {
                self.nodes
                    .get(&id)
                    .filter(|n| n.type_id == type_id)
                    .map(|n| (n.id, n.properties.clone()))
            })
            .collect()
    }

    // ── Batch insert operations ─────────────────────────────────────────

    /// Inserta un batch de nodos en paralelo. Retorna sus NodeIds.
    ///
    /// Los IDs se pre-asignan atómicamente en bloque (un solo `fetch_add`)
    /// y luego los nodos se insertan en paralelo con rayon.
    ///
    /// # Rendimiento
    /// - 1 atomic op para todo el batch (vs N ops en insert_node individual)
    /// - rayon parallel insert → ~N/P time con P threads
    /// - Pre-allocates DashMap capacity if needed
    pub fn insert_nodes_batch(&self, entries: &[(TypeId, Vec<(u16, Value)>)]) -> Vec<NodeId> {
        let count = entries.len();
        if count == 0 {
            return Vec::new();
        }

        // Reservar todo el rango de IDs en un solo atomic op
        let start_id = self.next_node_id.fetch_add(count as u64, Ordering::Relaxed);
        let ids: Vec<NodeId> = (0..count as u64)
            .map(|i| NodeId(start_id + i))
            .collect();

        // Insertar en paralelo por chunks para reducir contention en DashMap shards
        ids.par_iter()
            .zip(entries.par_iter())
            .for_each(|(id, (type_id, props))| {
                let mut data = NodeData::new(*id, *type_id);
                data.properties = props.clone();
                self.nodes.insert(*id, data);
            });

        ids
    }

    /// Inserta un batch de nodos sin propiedades (máxima velocidad).
    ///
    /// Ideal para bulk load de grafos donde las propiedades se asignan después.
    pub fn insert_nodes_batch_typed(&self, type_id: TypeId, count: usize) -> Vec<NodeId> {
        if count == 0 {
            return Vec::new();
        }

        let start_id = self.next_node_id.fetch_add(count as u64, Ordering::Relaxed);
        let ids: Vec<NodeId> = (0..count as u64)
            .map(|i| NodeId(start_id + i))
            .collect();

        ids.par_iter().for_each(|id| {
            self.nodes.insert(*id, NodeData::new(*id, type_id));
        });

        ids
    }

    /// Inserta un batch de edges en paralelo.
    ///
    /// Los edges se validan y se insertan en paralelo. La actualización de
    /// adjacency lists se hace de forma lock-free (DashMap shard locking).
    ///
    /// Retorna Vec<BikoResult<EdgeId>> — un resultado por edge.
    pub fn insert_edges_batch(
        &self,
        edges: &[(NodeId, NodeId, TypeId)],
    ) -> Vec<BikoResult<EdgeId>> {
        let count = edges.len();
        if count == 0 {
            return Vec::new();
        }

        let start_id = self.next_edge_id.fetch_add(count as u64, Ordering::Relaxed);

        edges
            .par_iter()
            .enumerate()
            .map(|(i, (source, target, type_id))| {
                if !self.nodes.contains_key(source) {
                    return Err(BikoError::NodeNotFound(*source));
                }
                if !self.nodes.contains_key(target) {
                    return Err(BikoError::NodeNotFound(*target));
                }

                let id = EdgeId(start_id + i as u64);
                let edge = EdgeData {
                    id,
                    source: *source,
                    target: *target,
                    type_id: *type_id,
                    properties: Vec::new(),
                };
                self.edges.insert(id, edge);

                if let Some(mut src) = self.nodes.get_mut(source) {
                    src.out_edges.push(id);
                }
                if let Some(mut tgt) = self.nodes.get_mut(target) {
                    tgt.in_edges.push(id);
                }

                Ok(id)
            })
            .collect()
    }

    /// Expone el siguiente NodeId que se asignará (para external ID mapping).
    pub fn next_node_id(&self) -> u64 {
        self.next_node_id.load(Ordering::Relaxed)
    }

    /// Expone el siguiente EdgeId que se asignará.
    pub fn next_edge_id(&self) -> u64 {
        self.next_edge_id.load(Ordering::Relaxed)
    }

    // ── Memory estimation ───────────────────────────────────────────────

    /// Calcula estadísticas detalladas de uso de memoria del grafo.
    ///
    /// Itera sobre todos los nodos y edges para calcular la memoria total.
    /// Útil para monitoreo, backpressure, y decisiones de eviction.
    pub fn memory_stats(&self) -> GraphMemoryStats {
        let mut nodes_bytes: usize = 0;
        let mut edges_bytes: usize = 0;
        let dashmap_overhead_per_entry = 48usize; // hash + metadata + bucket ptr

        for entry in self.nodes.iter() {
            nodes_bytes += entry.value().estimated_bytes() + dashmap_overhead_per_entry;
        }
        for entry in self.edges.iter() {
            edges_bytes += entry.value().estimated_bytes() + dashmap_overhead_per_entry;
        }

        let node_count = self.nodes.len();
        let edge_count = self.edges.len();

        GraphMemoryStats {
            node_count,
            edge_count,
            nodes_bytes,
            edges_bytes,
            total_bytes: nodes_bytes + edges_bytes,
            avg_bytes_per_node: if node_count > 0 { nodes_bytes / node_count } else { 0 },
            avg_bytes_per_edge: if edge_count > 0 { edges_bytes / edge_count } else { 0 },
        }
    }

    /// Estimación rápida de memoria total (sin iterar, basada en contadores).
    ///
    /// Menos precisa que `memory_stats()` pero O(1).
    pub fn estimated_memory_bytes(&self) -> usize {
        let node_avg = std::mem::size_of::<NodeData>() + 48; // stack + DashMap overhead
        let edge_avg = std::mem::size_of::<EdgeData>() + 48;
        self.nodes.len() * node_avg + self.edges.len() * edge_avg
    }
}

/// Estadísticas detalladas de uso de memoria del grafo.
#[derive(Debug, Clone)]
pub struct GraphMemoryStats {
    pub node_count: usize,
    pub edge_count: usize,
    /// Memoria total usada por nodos (incluyendo DashMap overhead)
    pub nodes_bytes: usize,
    /// Memoria total usada por edges (incluyendo DashMap overhead)
    pub edges_bytes: usize,
    /// Memoria total del grafo
    pub total_bytes: usize,
    /// Bytes promedio por nodo
    pub avg_bytes_per_node: usize,
    /// Bytes promedio por edge
    pub avg_bytes_per_edge: usize,
}

impl Default for ConcurrentGraph {
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

    fn person_type() -> TypeId {
        TypeId(1)
    }
    fn knows_type() -> TypeId {
        TypeId(10)
    }
    fn works_at_type() -> TypeId {
        TypeId(11)
    }

    #[test]
    fn test_insert_nodes() {
        let g = ConcurrentGraph::new();
        let n1 = g.insert_node(person_type());
        let n2 = g.insert_node(person_type());
        assert_ne!(n1, n2);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_insert_edge() {
        let g = ConcurrentGraph::new();
        let n1 = g.insert_node(person_type());
        let n2 = g.insert_node(person_type());
        let e = g.insert_edge(n1, n2, knows_type()).unwrap();

        assert_eq!(g.edge_count(), 1);

        let ed = g.get_edge(e).unwrap();
        assert_eq!(ed.source, n1);
        assert_eq!(ed.target, n2);
    }

    #[test]
    fn test_insert_edge_invalid_node() {
        let g = ConcurrentGraph::new();
        let n1 = g.insert_node(person_type());
        let result = g.insert_edge(n1, NodeId(999), knows_type());
        assert!(result.is_err());
    }

    #[test]
    fn test_neighbors() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(person_type());
        let b = g.insert_node(person_type());
        let c = g.insert_node(person_type());

        g.insert_edge(a, b, knows_type()).unwrap();
        g.insert_edge(a, c, knows_type()).unwrap();

        let out = g.neighbors(a, Direction::Out).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.contains(&b));
        assert!(out.contains(&c));

        let inc = g.neighbors(b, Direction::In).unwrap();
        assert_eq!(inc.len(), 1);
        assert!(inc.contains(&a));
    }

    #[test]
    fn test_neighbors_by_type() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(person_type());
        let b = g.insert_node(person_type());
        let c = g.insert_node(TypeId(2)); // Company

        g.insert_edge(a, b, knows_type()).unwrap();
        g.insert_edge(a, c, works_at_type()).unwrap();

        let knows_neighbors = g
            .neighbors_by_type(a, Direction::Out, knows_type())
            .unwrap();
        assert_eq!(knows_neighbors.len(), 1);
        assert_eq!(knows_neighbors[0], b);
    }

    #[test]
    fn test_remove_node() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(person_type());
        let b = g.insert_node(person_type());
        g.insert_edge(a, b, knows_type()).unwrap();

        g.remove_node(a).unwrap();
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);
        // b's in_edges should be cleaned up
        let node_b = g.get_node(b).unwrap();
        assert!(node_b.in_edges.is_empty());
    }

    #[test]
    fn test_remove_edge() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(person_type());
        let b = g.insert_node(person_type());
        let e = g.insert_edge(a, b, knows_type()).unwrap();

        g.remove_edge(e).unwrap();
        assert_eq!(g.edge_count(), 0);

        let node_a = g.get_node(a).unwrap();
        assert!(node_a.out_edges.is_empty());
    }

    #[test]
    fn test_set_node_property() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(person_type());
        g.set_node_property(a, 0, Value::from("Alice")).unwrap();

        let node = g.get_node(a).unwrap();
        assert_eq!(node.properties.len(), 1);
        assert_eq!(node.properties[0].1.as_str(), Some("Alice"));
    }

    #[test]
    fn test_node_with_props() {
        let g = ConcurrentGraph::new();
        let n = g.insert_node_with_props(
            person_type(),
            vec![(0, Value::from("Bob")), (1, Value::Int(30))],
        );
        let data = g.get_node(n).unwrap();
        assert_eq!(data.properties.len(), 2);
    }

    #[test]
    fn test_self_loop() {
        let g = ConcurrentGraph::new();
        let a = g.insert_node(person_type());
        let e = g.insert_edge(a, a, knows_type()).unwrap();

        let out = g.neighbors(a, Direction::Out).unwrap();
        assert_eq!(out, vec![a]);

        let both = g.neighbors(a, Direction::Both).unwrap();
        assert_eq!(both.len(), 2); // a appears as both out and in
    }

    // ── Memory estimation tests ─────────────────────────────────────────

    #[test]
    fn test_memory_stats_empty_graph() {
        let g = ConcurrentGraph::new();
        let stats = g.memory_stats();
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.total_bytes, 0);
    }

    #[test]
    fn test_memory_stats_with_nodes() {
        let g = ConcurrentGraph::new();
        for _ in 0..100 {
            g.insert_node(person_type());
        }
        let stats = g.memory_stats();
        assert_eq!(stats.node_count, 100);
        assert!(stats.nodes_bytes > 0);
        assert!(stats.avg_bytes_per_node > 0);
        // NodeData struct + DashMap overhead should be reasonable
        assert!(
            stats.avg_bytes_per_node < 300,
            "avg_bytes_per_node = {} (expected < 300)",
            stats.avg_bytes_per_node
        );
    }

    #[test]
    fn test_memory_stats_with_edges() {
        let g = ConcurrentGraph::new();
        let mut ids = Vec::new();
        for _ in 0..50 {
            ids.push(g.insert_node(person_type()));
        }
        for i in 0..49 {
            g.insert_edge(ids[i], ids[i + 1], knows_type()).unwrap();
        }
        let stats = g.memory_stats();
        assert_eq!(stats.node_count, 50);
        assert_eq!(stats.edge_count, 49);
        assert!(stats.total_bytes > 0);
        assert!(stats.edges_bytes > 0);
    }

    #[test]
    fn test_estimated_memory_fast() {
        let g = ConcurrentGraph::new();
        for _ in 0..1000 {
            g.insert_node(person_type());
        }
        let fast = g.estimated_memory_bytes();
        let detailed = g.memory_stats().total_bytes;
        // Fast estimate should be in the same ballpark (within 2x)
        assert!(fast > 0);
        assert!(detailed > 0);
        let ratio = fast as f64 / detailed as f64;
        assert!(
            ratio > 0.3 && ratio < 3.0,
            "fast={}, detailed={}, ratio={}",
            fast,
            detailed,
            ratio
        );
    }

    #[test]
    fn test_value_size_reduction() {
        // Value should be ≤ 32 bytes after boxing large variants
        let size = std::mem::size_of::<Value>();
        assert!(
            size <= 32,
            "Value size is {} bytes, expected ≤ 32",
            size
        );
    }

    #[test]
    fn test_nodedata_size_with_smallvec4() {
        // NodeData with SmallVec<4> should be smaller than with SmallVec<8>
        let size = std::mem::size_of::<NodeData>();
        // SmallVec<[EdgeId; 4]> = ~40 bytes each (len + capacity + 4*8),
        // 2 of them = ~80 bytes, + id(8) + type_id(2+pad) + props(24) ≈ ~120
        assert!(
            size <= 160,
            "NodeData size is {} bytes, expected ≤ 160",
            size
        );
    }

    #[test]
    fn test_nodedata_estimated_bytes() {
        let node = NodeData::new(NodeId(1), TypeId(1));
        let est = node.estimated_bytes();
        assert!(est > 0);
        // Empty node (no props, no spilled edges) should be close to struct size
        assert!(est < 200);
    }

    #[test]
    fn test_edgedata_estimated_bytes() {
        let edge = EdgeData {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(2),
            type_id: TypeId(10),
            properties: Vec::new(),
        };
        let est = edge.estimated_bytes();
        assert!(est > 0);
        assert!(est < 120);
    }

    #[test]
    fn test_memory_10k_nodes_reasonable() {
        let g = ConcurrentGraph::with_capacity(10_000, 0);
        let ids = g.insert_nodes_batch_typed(person_type(), 10_000);

        let stats = g.memory_stats();
        let mb = stats.total_bytes as f64 / (1024.0 * 1024.0);

        // 10K nodes should use < 10 MB (vs ~2.2 MB ideal)
        assert!(
            mb < 10.0,
            "10K nodes use {:.2} MB, expected < 10 MB",
            mb
        );
    }

    // ── Concurrent stress tests ─────────────────────────────────────────

    #[test]
    fn test_concurrent_stress_read_write() {
        use std::sync::Arc;
        use std::thread;

        let g = Arc::new(ConcurrentGraph::with_capacity(10_000, 10_000));
        let num_threads = 8;
        let ops_per_thread = 2_000;

        // Pre-populate some nodes so readers find data
        let seed_ids: Vec<NodeId> = (0..100)
            .map(|_| g.insert_node(person_type()))
            .collect();
        for i in 0..99 {
            let _ = g.insert_edge(seed_ids[i], seed_ids[i + 1], knows_type());
        }

        let mut handles = Vec::new();

        // Writer threads: insert nodes + edges concurrently
        for t in 0..(num_threads / 2) {
            let g = Arc::clone(&g);
            handles.push(thread::spawn(move || {
                let mut my_ids = Vec::new();
                for i in 0..ops_per_thread {
                    let n = g.insert_node(TypeId((t + 1) as u16));
                    g.set_node_property(n, 0, Value::Int(i as i64)).unwrap();
                    my_ids.push(n);

                    // Insert edges between consecutive nodes
                    if my_ids.len() >= 2 {
                        let len = my_ids.len();
                        let _ = g.insert_edge(
                            my_ids[len - 2],
                            my_ids[len - 1],
                            knows_type(),
                        );
                    }
                }
                my_ids.len()
            }));
        }

        // Reader threads: query nodes + neighbors concurrently
        for _ in 0..(num_threads / 2) {
            let g = Arc::clone(&g);
            let seed = seed_ids.clone();
            handles.push(thread::spawn(move || {
                let mut reads = 0usize;
                for _ in 0..ops_per_thread {
                    // Read random seed nodes
                    for id in &seed {
                        if g.get_node(*id).is_some() {
                            reads += 1;
                        }
                    }
                    // Neighbor queries
                    let _ = g.neighbors(seed[0], Direction::Out);
                    let _ = g.neighbors(seed[0], Direction::In);
                    // Node count (lock-free)
                    let _ = g.node_count();
                }
                reads
            }));
        }

        let results: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Verify: all writer threads completed successfully
        let writer_nodes: usize = results[..num_threads / 2].iter().sum();
        assert_eq!(writer_nodes, (num_threads / 2) * ops_per_thread);

        // Verify: readers completed without panicking
        let reader_reads: usize = results[num_threads / 2..].iter().sum();
        assert!(reader_reads > 0);

        // Verify: graph is consistent
        let total = g.node_count();
        assert!(
            total >= 100 + writer_nodes,
            "expected >= {} nodes, got {}",
            100 + writer_nodes,
            total,
        );

        // Verify: no edges lost from seed chain
        let seed_neighbors = g.neighbors(seed_ids[0], Direction::Out).unwrap();
        assert!(
            seed_neighbors.contains(&seed_ids[1]),
            "seed edge chain broken"
        );
    }

    #[test]
    fn test_concurrent_batch_insert_and_query() {
        use std::sync::Arc;
        use std::thread;

        let g = Arc::new(ConcurrentGraph::with_capacity(50_000, 0));
        let num_threads = 4;
        let batch_size = 5_000;

        let mut handles = Vec::new();

        // Each thread batch-inserts nodes AND runs parallel scan
        for t in 0..num_threads {
            let g = Arc::clone(&g);
            handles.push(thread::spawn(move || {
                let type_id = TypeId((t + 1) as u16);
                let ids = g.insert_nodes_batch_typed(type_id, batch_size);

                // Immediately query: parallel scan should see my nodes
                let found = g.par_scan_nodes_by_type(type_id);
                assert!(
                    found.len() >= ids.len(),
                    "thread {t}: expected >= {}, found {}",
                    ids.len(),
                    found.len(),
                );
                ids.len()
            }));
        }

        let counts: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let total_inserted: usize = counts.iter().sum();

        assert_eq!(total_inserted, num_threads * batch_size);
        assert_eq!(g.node_count(), total_inserted);
    }

    #[test]
    fn test_concurrent_remove_and_read() {
        use std::sync::Arc;
        use std::thread;

        let g = Arc::new(ConcurrentGraph::new());

        // Insert 1000 nodes
        let ids: Vec<NodeId> = (0..1000)
            .map(|_| g.insert_node(person_type()))
            .collect();
        // Chain edges
        for i in 0..999 {
            let _ = g.insert_edge(ids[i], ids[i + 1], knows_type());
        }

        let ids_arc = Arc::new(ids);

        // Thread 1: remove even-indexed nodes
        let g1 = Arc::clone(&g);
        let ids1 = Arc::clone(&ids_arc);
        let remover = thread::spawn(move || {
            let mut removed = 0;
            for i in (0..1000).step_by(2) {
                if g1.remove_node(ids1[i]).is_ok() {
                    removed += 1;
                }
            }
            removed
        });

        // Thread 2: read all nodes concurrently
        let g2 = Arc::clone(&g);
        let ids2 = Arc::clone(&ids_arc);
        let reader = thread::spawn(move || {
            let mut found = 0;
            for i in 0..1000 {
                if g2.get_node(ids2[i]).is_some() {
                    found += 1;
                }
            }
            found
        });

        let removed = remover.join().unwrap();
        let _found = reader.join().unwrap();

        // 500 even nodes should be removed
        assert_eq!(removed, 500);
        // Only odd nodes remain
        assert_eq!(g.node_count(), 500);
    }
}
