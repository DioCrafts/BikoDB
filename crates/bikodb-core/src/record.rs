// =============================================================================
// bikodb-core::record — Records: Vertex, Edge, Document
// =============================================================================
// Estructuras de datos cache-friendly para nodos y aristas del grafo.
//
// ## Diseño de memoria
//
// Un Vertex en memoria:
// ┌──────────────────────────────────────────────────────────────┐
// │ id: NodeId (8B) │ type_id: TypeId (2B) │ flags: u8 (1B)     │
// │ properties: SmallVec<(u16, Value)>  ← inline hasta 4 props  │
// │ out_edges: SmallVec<EdgeRef>        ← inline hasta 8 edges  │
// │ in_edges:  SmallVec<EdgeRef>        ← inline hasta 8 edges  │
// └──────────────────────────────────────────────────────────────┘
//
// Las propiedades usan u16 (property_id del Dictionary) en vez de String
// para reducir overhead y mejorar cache hits.
//
// ## EdgeRef vs Edge
// - EdgeRef (12 bytes): referencia compacta a un edge (edge_id + target_node)
//   Usada en listas de adjacencia para traversals rápidos.
// - Edge (struct completa): con propiedades, para cuando necesitas los datos.
//
// ## Inspiración
// - ArcadeDB: Vertex/Edge extienden Document, EdgeSegments como linked list
// - Neo4j: Double-linked list de relaciones por nodo
// - CSR: Para OLAP usamos representación CSR separada (en bikodb-graph)
// =============================================================================

use crate::types::{EdgeId, NodeId, TypeId};
use crate::value::Value;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

// ── Record Type ────────────────────────────────────────────────────────────

/// Tipo de registro almacenado (como RECORD_TYPE en ArcadeDB).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RecordType {
    /// Documento genérico (sin edges)
    Document = 0,
    /// Vértice del grafo (con listas de adjacencia)
    Vertex = 1,
    /// Arista del grafo
    Edge = 2,
    /// Segmento de edges (almacenamiento interno)
    EdgeSegment = 3,
}

// ── Direction ──────────────────────────────────────────────────────────────

/// Dirección para traversals y consultas de edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    /// Edges salientes (→)
    Out,
    /// Edges entrantes (←)
    In,
    /// Ambas direcciones (↔)
    Both,
}

// ── PropertyEntry ──────────────────────────────────────────────────────────

/// Una propiedad almacenada como (property_id, value).
///
/// Usamos u16 como ID de propiedad (del Dictionary del schema) en vez de
/// String para ahorrar 24 bytes por propiedad y mejorar cache locality.
///
/// El Dictionary (en bikodb-storage) mapea: "name" → 0, "age" → 1, etc.
pub type PropertyEntry = (u16, Value);

/// Lista de propiedades optimizada: inline hasta 4 propiedades en stack.
/// La mayoría de nodos tienen pocas propiedades (<= 4), así que evitamos
/// heap allocation en el caso común.
pub type PropertyList = SmallVec<[PropertyEntry; 4]>;

// ── EdgeRef ────────────────────────────────────────────────────────────────

/// Referencia compacta a un edge para listas de adjacencia.
///
/// **12 bytes** total — diseñada para caber en cache lines.
/// Para traversals solo necesitamos saber: qué edge es, a dónde va,
/// y de qué tipo es. No cargamos propiedades del edge salvo que se pidan.
///
/// # Layout en memoria
/// ```text
/// ┌─────────────┬─────────────┬──────────┐
/// │ edge_id: u64│ target: u64 │ type: u16│
/// │   8 bytes   │   8 bytes   │  2 bytes │  = 18 bytes (padded to 24)
/// └─────────────┴─────────────┴──────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeRef {
    /// ID del edge (para cargar propiedades si se necesitan)
    pub edge_id: EdgeId,
    /// ID del nodo destino (OUT) u origen (IN) según contexto
    pub target_node: NodeId,
    /// Tipo del edge (para filtrar sin cargar el edge completo)
    pub type_id: TypeId,
}

// ── Vertex ─────────────────────────────────────────────────────────────────

/// Vértice del Knowledge Graph.
///
/// Contiene listas de adjacencia inline (SmallVec) para traversals rápidos.
/// Las primeras 8 EdgeRefs en cada dirección se almacenan en stack.
/// Nodos con alta conectividad hacen spill a heap automáticamente.
///
/// # Ejemplo
/// ```
/// use bikodb_core::record::Vertex;
/// use bikodb_core::types::{NodeId, TypeId};
/// use bikodb_core::value::Value;
///
/// let mut v = Vertex::new(NodeId(1), TypeId(0));
/// v.set_property(0, Value::string("Alice"));    // "name" → property_id 0
/// v.set_property(1, Value::Int(30));             // "age"  → property_id 1
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vertex {
    /// Identificador único del vértice
    pub id: NodeId,
    /// Tipo del vértice (referencia al schema: "Person", "Company", etc.)
    pub type_id: TypeId,
    /// Propiedades del vértice (inline hasta 4)
    pub properties: PropertyList,
    /// Edges salientes (→). Inline hasta 8 edges en stack.
    pub out_edges: SmallVec<[EdgeRef; 8]>,
    /// Edges entrantes (←). Inline hasta 8 edges en stack.
    pub in_edges: SmallVec<[EdgeRef; 8]>,
}

impl Vertex {
    /// Crea un vértice nuevo sin propiedades ni edges.
    pub fn new(id: NodeId, type_id: TypeId) -> Self {
        Self {
            id,
            type_id,
            properties: SmallVec::new(),
            out_edges: SmallVec::new(),
            in_edges: SmallVec::new(),
        }
    }

    /// Establece o actualiza una propiedad por su ID.
    /// Si la propiedad ya existe, la reemplaza.
    pub fn set_property(&mut self, prop_id: u16, value: Value) {
        if let Some(entry) = self.properties.iter_mut().find(|(id, _)| *id == prop_id) {
            entry.1 = value;
        } else {
            self.properties.push((prop_id, value));
        }
    }

    /// Obtiene el valor de una propiedad por su ID.
    pub fn get_property(&self, prop_id: u16) -> Option<&Value> {
        self.properties
            .iter()
            .find(|(id, _)| *id == prop_id)
            .map(|(_, v)| v)
    }

    /// Añade una referencia a un edge saliente.
    pub fn add_out_edge(&mut self, edge_ref: EdgeRef) {
        self.out_edges.push(edge_ref);
    }

    /// Añade una referencia a un edge entrante.
    pub fn add_in_edge(&mut self, edge_ref: EdgeRef) {
        self.in_edges.push(edge_ref);
    }

    /// Número total de edges (OUT + IN).
    pub fn degree(&self) -> usize {
        self.out_edges.len() + self.in_edges.len()
    }

    /// Edges en una dirección específica.
    pub fn edges(&self, direction: Direction) -> &[EdgeRef] {
        match direction {
            Direction::Out => &self.out_edges,
            Direction::In => &self.in_edges,
            Direction::Both => {
                // Para Both, el caller debe iterar ambas listas.
                // Retornamos out_edges por conveniencia; usar edges_iter() para ambos.
                &self.out_edges
            }
        }
    }

    /// Iterador sobre edges en una dirección, opcionalmente filtrado por tipo.
    pub fn edges_filtered(
        &self,
        direction: Direction,
        type_filter: Option<TypeId>,
    ) -> impl Iterator<Item = &EdgeRef> {
        let out_iter: Box<dyn Iterator<Item = &EdgeRef>> = match direction {
            Direction::Out => Box::new(self.out_edges.iter()),
            Direction::In => Box::new(self.in_edges.iter()),
            Direction::Both => Box::new(self.out_edges.iter().chain(self.in_edges.iter())),
        };

        out_iter.filter(move |e| type_filter.is_none() || type_filter == Some(e.type_id))
    }

    /// IDs de nodos vecinos en una dirección.
    pub fn neighbor_ids(&self, direction: Direction) -> Vec<NodeId> {
        self.edges_filtered(direction, None)
            .map(|e| e.target_node)
            .collect()
    }
}

// ── Edge ───────────────────────────────────────────────────────────────────

/// Arista del Knowledge Graph (con propiedades completas).
///
/// Se carga cuando necesitamos acceder a las propiedades del edge.
/// Para traversals puros, solo se usa EdgeRef (más compacto).
///
/// # Ejemplo
/// ```
/// use bikodb_core::record::Edge;
/// use bikodb_core::types::{EdgeId, NodeId, TypeId};
/// use bikodb_core::value::Value;
///
/// let mut e = Edge::new(EdgeId(1), TypeId(0), NodeId(10), NodeId(20));
/// e.set_property(0, Value::Int(2020));  // "since" → property_id 0
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Identificador único del edge
    pub id: EdgeId,
    /// Tipo del edge ("KNOWS", "WORKS_AT", etc.)
    pub type_id: TypeId,
    /// Nodo origen (source)
    pub source: NodeId,
    /// Nodo destino (target)
    pub target: NodeId,
    /// Propiedades del edge
    pub properties: PropertyList,
}

impl Edge {
    /// Crea un edge nuevo sin propiedades.
    pub fn new(id: EdgeId, type_id: TypeId, source: NodeId, target: NodeId) -> Self {
        Self {
            id,
            type_id,
            source,
            target,
            properties: SmallVec::new(),
        }
    }

    /// Establece o actualiza una propiedad.
    pub fn set_property(&mut self, prop_id: u16, value: Value) {
        if let Some(entry) = self.properties.iter_mut().find(|(id, _)| *id == prop_id) {
            entry.1 = value;
        } else {
            self.properties.push((prop_id, value));
        }
    }

    /// Obtiene el valor de una propiedad.
    pub fn get_property(&self, prop_id: u16) -> Option<&Value> {
        self.properties
            .iter()
            .find(|(id, _)| *id == prop_id)
            .map(|(_, v)| v)
    }

    /// Convierte a EdgeRef para almacenar en listas de adjacencia.
    pub fn as_out_ref(&self) -> EdgeRef {
        EdgeRef {
            edge_id: self.id,
            target_node: self.target,
            type_id: self.type_id,
        }
    }

    /// Convierte a EdgeRef desde la perspectiva del target (edge entrante).
    pub fn as_in_ref(&self) -> EdgeRef {
        EdgeRef {
            edge_id: self.id,
            target_node: self.source,
            type_id: self.type_id,
        }
    }
}

// ── Record trait ───────────────────────────────────────────────────────────

/// Trait común para todos los tipos de registro.
///
/// Permite código genérico que opera sobre cualquier record
/// (document, vertex, edge) sin conocer el tipo concreto.
pub trait Record {
    /// Tipo del registro (Document, Vertex, Edge).
    fn record_type(&self) -> RecordType;

    /// Obtiene una propiedad por ID.
    fn get_property(&self, prop_id: u16) -> Option<&Value>;

    /// Establece una propiedad por ID.
    fn set_property(&mut self, prop_id: u16, value: Value);

    /// Lista todas las propiedades como slice.
    fn properties(&self) -> &[(u16, Value)];
}

impl Record for Vertex {
    fn record_type(&self) -> RecordType {
        RecordType::Vertex
    }

    fn get_property(&self, prop_id: u16) -> Option<&Value> {
        self.get_property(prop_id)
    }

    fn set_property(&mut self, prop_id: u16, value: Value) {
        self.set_property(prop_id, value);
    }

    fn properties(&self) -> &[(u16, Value)] {
        &self.properties
    }
}

impl Record for Edge {
    fn record_type(&self) -> RecordType {
        RecordType::Edge
    }

    fn get_property(&self, prop_id: u16) -> Option<&Value> {
        self.get_property(prop_id)
    }

    fn set_property(&mut self, prop_id: u16, value: Value) {
        self.set_property(prop_id, value);
    }

    fn properties(&self) -> &[(u16, Value)] {
        &self.properties
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_creation_and_properties() {
        let mut v = Vertex::new(NodeId(1), TypeId(0));
        v.set_property(0, Value::string("Alice"));
        v.set_property(1, Value::Int(30));

        assert_eq!(v.get_property(0).unwrap().as_str(), Some("Alice"));
        assert_eq!(v.get_property(1).unwrap().as_int(), Some(30));
        assert!(v.get_property(99).is_none());
    }

    #[test]
    fn test_vertex_property_update() {
        let mut v = Vertex::new(NodeId(1), TypeId(0));
        v.set_property(0, Value::string("Alice"));
        v.set_property(0, Value::string("Bob")); // Actualizar

        assert_eq!(v.get_property(0).unwrap().as_str(), Some("Bob"));
        assert_eq!(v.properties.len(), 1); // No duplicata
    }

    #[test]
    fn test_vertex_adjacency() {
        let mut v = Vertex::new(NodeId(1), TypeId(0));

        let er1 = EdgeRef {
            edge_id: EdgeId(100),
            target_node: NodeId(2),
            type_id: TypeId(0),
        };
        let er2 = EdgeRef {
            edge_id: EdgeId(101),
            target_node: NodeId(3),
            type_id: TypeId(1),
        };

        v.add_out_edge(er1);
        v.add_out_edge(er2);

        assert_eq!(v.degree(), 2);
        assert_eq!(v.out_edges.len(), 2);

        // Filtrar por tipo
        let filtered: Vec<_> = v.edges_filtered(Direction::Out, Some(TypeId(0))).collect();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].target_node, NodeId(2));
    }

    #[test]
    fn test_vertex_neighbors() {
        let mut v = Vertex::new(NodeId(1), TypeId(0));
        v.add_out_edge(EdgeRef {
            edge_id: EdgeId(1),
            target_node: NodeId(10),
            type_id: TypeId(0),
        });
        v.add_out_edge(EdgeRef {
            edge_id: EdgeId(2),
            target_node: NodeId(20),
            type_id: TypeId(0),
        });

        let neighbors = v.neighbor_ids(Direction::Out);
        assert_eq!(neighbors, vec![NodeId(10), NodeId(20)]);
    }

    #[test]
    fn test_edge_creation() {
        let mut e = Edge::new(EdgeId(1), TypeId(0), NodeId(10), NodeId(20));
        e.set_property(0, Value::Int(2020));

        assert_eq!(e.source, NodeId(10));
        assert_eq!(e.target, NodeId(20));
        assert_eq!(e.get_property(0).unwrap().as_int(), Some(2020));
    }

    #[test]
    fn test_edge_refs() {
        let e = Edge::new(EdgeId(5), TypeId(1), NodeId(10), NodeId(20));

        let out_ref = e.as_out_ref();
        assert_eq!(out_ref.target_node, NodeId(20));

        let in_ref = e.as_in_ref();
        assert_eq!(in_ref.target_node, NodeId(10));
    }

    #[test]
    fn test_record_trait() {
        let mut v = Vertex::new(NodeId(1), TypeId(0));
        let record: &mut dyn Record = &mut v;
        record.set_property(0, Value::string("test"));
        assert_eq!(record.record_type(), RecordType::Vertex);
        assert!(record.get_property(0).is_some());
    }
}
