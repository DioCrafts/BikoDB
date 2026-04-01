// =============================================================================
// bikodb-server::http_api — HTTP/REST API con axum
// =============================================================================
// API REST completa para integración con apps externas: Python, Node.js, Go,
// Java, microservicios, etc. Cualquier lenguaje con HTTP client puede usar BikoDB.
//
// ## Endpoints
//   POST   /api/v1/schema/types          → Registrar tipo
//   POST   /api/v1/schema/properties     → Registrar propiedad
//   POST   /api/v1/schema/relationships  → Registrar relación
//
//   POST   /api/v1/nodes                 → Crear nodo
//   GET    /api/v1/nodes/:id             → Obtener nodo
//   DELETE /api/v1/nodes/:id             → Eliminar nodo
//   PUT    /api/v1/nodes/:id/properties  → Actualizar propiedad
//   GET    /api/v1/nodes/:id/neighbors   → Vecinos de un nodo
//
//   POST   /api/v1/edges                 → Crear edge
//   DELETE /api/v1/edges/:id             → Eliminar edge
//
//   POST   /api/v1/query                 → Query (SQL/Cypher/Gremlin)
//
//   POST   /api/v1/vectors/insert        → Insertar vector
//   POST   /api/v1/vectors/search        → Vector search (KNN)
//
//   POST   /api/v1/documents             → Crear documento
//   GET    /api/v1/documents/:col/:id    → Obtener documento
//   POST   /api/v1/documents/query       → Query documentos
//
//   POST   /api/v1/udf/call              → Llamar UDF
//   POST   /api/v1/algorithms/run        → Ejecutar algoritmo custom
//
//   GET    /api/v1/status                → Estado del motor
//   GET    /api/v1/health                → Health check
// =============================================================================

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post, put},
    Json, Router,
};
use bikodb_core::error::BikoError;
use bikodb_core::record::Direction;
use bikodb_core::types::{EdgeId, NodeId, TypeId};
use bikodb_core::value::Value;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::Database;

/// Estado compartido del servidor HTTP.
pub type AppState = Arc<parking_lot::RwLock<Database>>;

// ── JSON DTOs ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct RegisterTypeReq {
    pub name: String,
    pub type_id: u16,
}

#[derive(Deserialize)]
pub struct RegisterPropertyReq {
    pub name: String,
    pub prop_id: u16,
}

#[derive(Deserialize)]
pub struct RegisterRelReq {
    pub name: String,
    pub type_id: u16,
}

#[derive(Deserialize)]
pub struct CreateNodeReq {
    pub type_id: u16,
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
pub struct CreateNodeResp {
    pub node_id: u64,
}

#[derive(Serialize, Deserialize)]
pub struct NodeResp {
    pub id: u64,
    pub type_id: u16,
    pub properties: HashMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
pub struct CreateEdgeReq {
    pub source: u64,
    pub target: u64,
    pub type_id: u16,
}

#[derive(Serialize, Deserialize)]
pub struct CreateEdgeResp {
    pub edge_id: u64,
}

#[derive(Deserialize)]
pub struct SetPropertyReq {
    pub name: String,
    pub value: serde_json::Value,
}

#[derive(Deserialize)]
pub struct QueryReq {
    pub language: String,
    pub query: String,
}

#[derive(Serialize, Deserialize)]
pub struct QueryResp {
    pub rows: Vec<RowResp>,
}

#[derive(Serialize, Deserialize)]
pub struct RowResp {
    pub node_id: u64,
    pub properties: HashMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
pub struct VectorInsertReq {
    pub node_id: u64,
    pub vector: Vec<f32>,
}

#[derive(Deserialize)]
pub struct VectorSearchReq {
    pub query: Vec<f32>,
    pub k: usize,
}

#[derive(Serialize, Deserialize)]
pub struct VectorSearchResp {
    pub results: Vec<VectorHit>,
}

#[derive(Serialize, Deserialize)]
pub struct VectorHit {
    pub node_id: u64,
    pub distance: f32,
}

#[derive(Deserialize)]
pub struct CreateDocReq {
    pub collection: String,
    pub fields: HashMap<String, serde_json::Value>,
    pub linked_node: Option<u64>,
}

#[derive(Serialize, Deserialize)]
pub struct CreateDocResp {
    pub doc_id: u64,
}

#[derive(Serialize, Deserialize)]
pub struct DocResp {
    pub id: u64,
    pub fields: HashMap<String, serde_json::Value>,
    pub linked_node: Option<u64>,
}

#[derive(Deserialize)]
pub struct DocQueryReq {
    pub collection: String,
    pub filter: DocFilterReq,
}

#[derive(Deserialize)]
#[serde(tag = "op")]
pub enum DocFilterReq {
    Eq { path: String, value: serde_json::Value },
    Exists { path: String },
    And { filters: Vec<DocFilterReq> },
    Or { filters: Vec<DocFilterReq> },
}

#[derive(Deserialize)]
pub struct UdfCallReq {
    pub name: String,
    pub args: Vec<serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
pub struct UdfCallResp {
    pub result: serde_json::Value,
}

#[derive(Deserialize)]
pub struct AlgorithmRunReq {
    pub name: String,
    #[serde(default)]
    pub params: HashMap<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
pub struct AlgorithmRunResp {
    pub node_scores: HashMap<String, f64>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
pub struct StatusResp {
    pub node_count: usize,
    pub edge_count: usize,
    pub version: String,
    pub plugins: Vec<String>,
    pub udfs: Vec<String>,
    pub algorithms: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct HealthResp {
    pub status: String,
}

#[derive(Serialize, Deserialize)]
pub struct ErrorResp {
    pub error: String,
}

#[derive(Serialize, Deserialize)]
pub struct NeighborsResp {
    pub neighbors: Vec<u64>,
}

#[derive(Serialize, Deserialize)]
pub struct OkResp {
    pub ok: bool,
}

// ── Helper: Convert BikoError to HTTP Response ───────────────────────────────

fn biko_err(e: BikoError) -> (StatusCode, Json<ErrorResp>) {
    let status = match &e {
        BikoError::NodeNotFound(_) => StatusCode::NOT_FOUND,
        BikoError::TypeNotFound(_) => StatusCode::NOT_FOUND,
        BikoError::PropertyNotFound { .. } => StatusCode::NOT_FOUND,
        BikoError::IndexNotFound(_) => StatusCode::NOT_FOUND,
        BikoError::DuplicateKey { .. } => StatusCode::CONFLICT,
        BikoError::TypeAlreadyExists(_) => StatusCode::CONFLICT,
        BikoError::QueryParse(_) => StatusCode::BAD_REQUEST,
        BikoError::Unsupported(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, Json(ErrorResp { error: e.to_string() }))
}

// ── Helper: Convert between Value and serde_json::Value ─────────────────────

fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::json!(*b),
        Value::Int(i) => serde_json::json!(*i),
        Value::Float(f) => serde_json::json!(*f),
        Value::String(s) => serde_json::json!(s),
        Value::List(list) => serde_json::Value::Array(list.iter().map(value_to_json).collect()),
        Value::Map(map) => {
            let obj: serde_json::Map<String, serde_json::Value> = map.iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        Value::Embedding(emb) => serde_json::json!(emb),
        _ => serde_json::json!(format!("{:?}", v)),
    }
}

fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::string(s.clone()),
        serde_json::Value::Array(arr) => Value::List(Box::new(arr.iter().map(json_to_value).collect())),
        serde_json::Value::Object(obj) => {
            let map: std::collections::HashMap<String, Value> = obj.iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect();
            Value::Map(Box::new(map))
        }
    }
}

// ── Helper: Convert Row to RowResp with prop names ──────────────────────────

fn row_to_resp(row: &bikodb_execution::operator::Row, prop_names: &HashMap<u16, String>) -> RowResp {
    let mut properties = HashMap::new();
    for (pid, val) in &row.properties {
        let name = prop_names.get(pid).cloned().unwrap_or_else(|| format!("prop_{}", pid));
        properties.insert(name, value_to_json(val));
    }
    RowResp {
        node_id: row.node_id.0,
        properties,
    }
}

// ── Router Construction ─────────────────────────────────────────────────────

/// Construye el router HTTP/REST completo de BikoDB.
///
/// # Ejemplo
/// ```no_run
/// use bikodb_server::{Database, http_api};
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() {
///     let db = Database::new();
///     let state = Arc::new(parking_lot::RwLock::new(db));
///     let app = http_api::build_router(state);
///     let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
///     axum::serve(listener, app).await.unwrap();
/// }
/// ```
pub fn build_router(state: AppState) -> Router {
    Router::new()
        // Health & Status
        .route("/api/v1/health", get(health))
        .route("/api/v1/status", get(status))
        // Schema
        .route("/api/v1/schema/types", post(register_type))
        .route("/api/v1/schema/properties", post(register_property))
        .route("/api/v1/schema/relationships", post(register_relationship))
        // Nodes
        .route("/api/v1/nodes", post(create_node))
        .route("/api/v1/nodes/{id}", get(get_node))
        .route("/api/v1/nodes/{id}", delete(delete_node))
        .route("/api/v1/nodes/{id}/properties", put(set_node_property))
        .route("/api/v1/nodes/{id}/neighbors", get(get_neighbors))
        // Edges
        .route("/api/v1/edges", post(create_edge))
        .route("/api/v1/edges/{id}", delete(delete_edge))
        // Query
        .route("/api/v1/query", post(query))
        // Vectors
        .route("/api/v1/vectors/insert", post(vector_insert))
        .route("/api/v1/vectors/search", post(vector_search))
        // Documents
        .route("/api/v1/documents", post(create_document))
        .route("/api/v1/documents/{collection}/{id}", get(get_document))
        .route("/api/v1/documents/query", post(query_documents))
        // Plugins / UDFs / Algorithms
        .route("/api/v1/udf/call", post(call_udf))
        .route("/api/v1/algorithms/run", post(run_algorithm))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

// ── Handlers ────────────────────────────────────────────────────────────────

async fn health() -> Json<HealthResp> {
    Json(HealthResp { status: "ok".into() })
}

async fn status(State(state): State<AppState>) -> Json<StatusResp> {
    let db = state.read();
    let pm = db.plugin_manager();
    Json(StatusResp {
        node_count: db.node_count(),
        edge_count: db.edge_count(),
        version: "0.2.0".into(),
        plugins: pm.list_plugins().iter().map(|p| p.name.clone()).collect(),
        udfs: pm.list_udfs(),
        algorithms: pm.list_algorithms(),
    })
}

async fn register_type(
    State(state): State<AppState>,
    Json(req): Json<RegisterTypeReq>,
) -> (StatusCode, Json<OkResp>) {
    let db = state.read();
    db.register_type(&req.name, TypeId(req.type_id));
    (StatusCode::CREATED, Json(OkResp { ok: true }))
}

async fn register_property(
    State(state): State<AppState>,
    Json(req): Json<RegisterPropertyReq>,
) -> (StatusCode, Json<OkResp>) {
    let db = state.read();
    db.register_property(&req.name, req.prop_id);
    (StatusCode::CREATED, Json(OkResp { ok: true }))
}

async fn register_relationship(
    State(state): State<AppState>,
    Json(req): Json<RegisterRelReq>,
) -> (StatusCode, Json<OkResp>) {
    let db = state.read();
    db.register_relationship(&req.name, TypeId(req.type_id));
    (StatusCode::CREATED, Json(OkResp { ok: true }))
}

async fn create_node(
    State(state): State<AppState>,
    Json(req): Json<CreateNodeReq>,
) -> Result<(StatusCode, Json<CreateNodeResp>), (StatusCode, Json<ErrorResp>)> {
    let db = state.read();
    let converted: Vec<(String, Value)> = req.properties.iter()
        .map(|(k, v)| (k.clone(), json_to_value(v)))
        .collect();
    let props: Vec<(&str, Value)> = converted.iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect();
    let node_id = db.create_node(TypeId(req.type_id), props);
    Ok((StatusCode::CREATED, Json(CreateNodeResp { node_id: node_id.0 })))
}

async fn get_node(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<NodeResp>, (StatusCode, Json<ErrorResp>)> {
    let db = state.read();
    let node = db.get_node(NodeId(id))
        .ok_or_else(|| biko_err(BikoError::NodeNotFound(NodeId(id))))?;
    let prop_names: HashMap<u16, String> = db.prop_names_snapshot();
    let mut properties = HashMap::new();
    for (pid, val) in &node.properties {
        let name = prop_names.get(pid).cloned().unwrap_or_else(|| format!("prop_{}", pid));
        properties.insert(name, value_to_json(val));
    }
    Ok(Json(NodeResp {
        id: node.id.0,
        type_id: node.type_id.0,
        properties,
    }))
}

async fn delete_node(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<OkResp>, (StatusCode, Json<ErrorResp>)> {
    let db = state.read();
    db.remove_node(NodeId(id)).map_err(biko_err)?;
    Ok(Json(OkResp { ok: true }))
}

async fn set_node_property(
    State(state): State<AppState>,
    Path(id): Path<u64>,
    Json(req): Json<SetPropertyReq>,
) -> Result<Json<OkResp>, (StatusCode, Json<ErrorResp>)> {
    let db = state.read();
    db.set_node_property(NodeId(id), &req.name, json_to_value(&req.value)).map_err(biko_err)?;
    Ok(Json(OkResp { ok: true }))
}

async fn get_neighbors(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<NeighborsResp>, (StatusCode, Json<ErrorResp>)> {
    let db = state.read();
    let neighbors = db.neighbors(NodeId(id), Direction::Out).map_err(biko_err)?;
    Ok(Json(NeighborsResp {
        neighbors: neighbors.iter().map(|n| n.0).collect(),
    }))
}

async fn create_edge(
    State(state): State<AppState>,
    Json(req): Json<CreateEdgeReq>,
) -> Result<(StatusCode, Json<CreateEdgeResp>), (StatusCode, Json<ErrorResp>)> {
    let db = state.read();
    let edge_id = db.create_edge(NodeId(req.source), NodeId(req.target), TypeId(req.type_id))
        .map_err(biko_err)?;
    Ok((StatusCode::CREATED, Json(CreateEdgeResp { edge_id: edge_id.0 })))
}

async fn delete_edge(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<OkResp>, (StatusCode, Json<ErrorResp>)> {
    let db = state.read();
    db.remove_edge(EdgeId(id)).map_err(biko_err)?;
    Ok(Json(OkResp { ok: true }))
}

async fn query(
    State(state): State<AppState>,
    Json(req): Json<QueryReq>,
) -> Result<Json<QueryResp>, (StatusCode, Json<ErrorResp>)> {
    let db = state.read();
    let rows = db.query(&req.language, &req.query).map_err(biko_err)?;
    let prop_names = db.prop_names_snapshot();
    let resp_rows = rows.iter().map(|r| row_to_resp(r, &prop_names)).collect();
    Ok(Json(QueryResp { rows: resp_rows }))
}

async fn vector_insert(
    State(state): State<AppState>,
    Json(req): Json<VectorInsertReq>,
) -> Result<Json<OkResp>, (StatusCode, Json<ErrorResp>)> {
    let db = state.read();
    db.insert_vector(NodeId(req.node_id), req.vector).map_err(biko_err)?;
    Ok(Json(OkResp { ok: true }))
}

async fn vector_search(
    State(state): State<AppState>,
    Json(req): Json<VectorSearchReq>,
) -> Result<Json<VectorSearchResp>, (StatusCode, Json<ErrorResp>)> {
    let db = state.read();
    let results = db.vector_search(&req.query, req.k).map_err(biko_err)?;
    Ok(Json(VectorSearchResp {
        results: results.iter().map(|(nid, dist)| VectorHit {
            node_id: nid.0,
            distance: *dist,
        }).collect(),
    }))
}

async fn create_document(
    State(state): State<AppState>,
    Json(req): Json<CreateDocReq>,
) -> Result<(StatusCode, Json<CreateDocResp>), (StatusCode, Json<ErrorResp>)> {
    let db = state.read();
    let fields: HashMap<String, Value> = req.fields.iter()
        .map(|(k, v)| (k.clone(), json_to_value(v)))
        .collect();
    let doc_id = if let Some(node_id) = req.linked_node {
        db.create_document_linked(&req.collection, fields, NodeId(node_id))
    } else {
        db.create_document(&req.collection, fields)
    };
    Ok((StatusCode::CREATED, Json(CreateDocResp { doc_id: doc_id.0 })))
}

async fn get_document(
    State(state): State<AppState>,
    Path((collection, id)): Path<(String, u64)>,
) -> Result<Json<DocResp>, (StatusCode, Json<ErrorResp>)> {
    let db = state.read();
    let doc = db.get_document(&collection, bikodb_graph::document::DocumentId(id))
        .ok_or_else(|| (StatusCode::NOT_FOUND, Json(ErrorResp {
            error: format!("Document {} not found in '{}'", id, collection),
        })))?;
    Ok(Json(DocResp {
        id: doc.id.0,
        fields: doc.fields.iter().map(|(k, v)| (k.clone(), value_to_json(v))).collect(),
        linked_node: doc.linked_node.map(|n| n.0),
    }))
}

async fn query_documents(
    State(state): State<AppState>,
    Json(req): Json<DocQueryReq>,
) -> Result<Json<Vec<DocResp>>, (StatusCode, Json<ErrorResp>)> {
    let filter = convert_doc_filter(&req.filter);
    let db = state.read();
    let docs = db.query_documents(&req.collection, &filter);
    Ok(Json(docs.iter().map(|d| DocResp {
        id: d.id.0,
        fields: d.fields.iter().map(|(k, v)| (k.clone(), value_to_json(v))).collect(),
        linked_node: d.linked_node.map(|n| n.0),
    }).collect()))
}

fn convert_doc_filter(req: &DocFilterReq) -> bikodb_graph::document::DocFilter {
    use bikodb_graph::document::DocFilter;
    match req {
        DocFilterReq::Eq { path, value } => DocFilter::Eq {
            path: path.clone(),
            value: json_to_value(value),
        },
        DocFilterReq::Exists { path } => DocFilter::Exists { path: path.clone() },
        DocFilterReq::And { filters } => DocFilter::And(filters.iter().map(convert_doc_filter).collect()),
        DocFilterReq::Or { filters } => DocFilter::Or(filters.iter().map(convert_doc_filter).collect()),
    }
}

async fn call_udf(
    State(state): State<AppState>,
    Json(req): Json<UdfCallReq>,
) -> Result<Json<UdfCallResp>, (StatusCode, Json<ErrorResp>)> {
    let db = state.read();
    let args: Vec<Value> = req.args.iter().map(json_to_value).collect();
    let result = db.call_udf(&req.name, &args).map_err(biko_err)?;
    let value = match result {
        bikodb_core::plugin::UdfReturn::Scalar(v) => value_to_json(&v),
        bikodb_core::plugin::UdfReturn::List(vs) => {
            serde_json::Value::Array(vs.iter().map(value_to_json).collect())
        }
    };
    Ok(Json(UdfCallResp { result: value }))
}

async fn run_algorithm(
    State(state): State<AppState>,
    Json(req): Json<AlgorithmRunReq>,
) -> Result<Json<AlgorithmRunResp>, (StatusCode, Json<ErrorResp>)> {
    let db = state.read();
    let params: HashMap<String, Value> = req.params.iter()
        .map(|(k, v)| (k.clone(), json_to_value(v)))
        .collect();
    let result = db.run_algorithm(&req.name, params).map_err(biko_err)?;
    let node_scores: HashMap<String, f64> = result.node_scores.iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
    Ok(Json(AlgorithmRunResp {
        node_scores,
        metadata: result.metadata.iter().map(|(k, v)| (k.clone(), value_to_json(v))).collect(),
    }))
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, Method};
    use tower::ServiceExt;

    fn test_state() -> AppState {
        let db = Database::new();
        Arc::new(parking_lot::RwLock::new(db))
    }

    fn setup_schema(state: &AppState) {
        let db = state.read();
        db.register_type("Person", TypeId(1));
        db.register_property("name", 0);
        db.register_property("age", 1);
        db.register_relationship("KNOWS", TypeId(10));
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = build_router(test_state());
        let req = Request::builder()
            .uri("/api/v1/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let health: HealthResp = serde_json::from_slice(&body).unwrap();
        assert_eq!(health.status, "ok");
    }

    #[tokio::test]
    async fn test_status_endpoint() {
        let state = test_state();
        setup_schema(&state);
        let app = build_router(state);
        let req = Request::builder()
            .uri("/api/v1/status")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let status: StatusResp = serde_json::from_slice(&body).unwrap();
        assert_eq!(status.version, "0.2.0");
    }

    #[tokio::test]
    async fn test_create_and_get_node() {
        let state = test_state();
        setup_schema(&state);
        let app = build_router(state.clone());

        // Create node
        let create_req = serde_json::json!({
            "type_id": 1,
            "properties": {
                "name": "Alice",
                "age": 30
            }
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/v1/nodes")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&create_req).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let created: CreateNodeResp = serde_json::from_slice(&body).unwrap();
        let node_id = created.node_id;

        // Get node
        let app2 = build_router(state);
        let req = Request::builder()
            .uri(format!("/api/v1/nodes/{}", node_id))
            .body(Body::empty())
            .unwrap();
        let resp = app2.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let node: NodeResp = serde_json::from_slice(&body).unwrap();
        assert_eq!(node.id, node_id);
        assert_eq!(node.type_id, 1);
        assert_eq!(node.properties.get("name"), Some(&serde_json::json!("Alice")));
    }

    #[tokio::test]
    async fn test_create_edge_and_neighbors() {
        let state = test_state();
        setup_schema(&state);

        // Create two nodes
        {
            let db = state.read();
            db.create_node(TypeId(1), vec![("name", Value::from("Alice"))]);
            db.create_node(TypeId(1), vec![("name", Value::from("Bob"))]);
        }

        // Create edge via API
        let app = build_router(state.clone());
        let edge_req = serde_json::json!({
            "source": 1,
            "target": 2,
            "type_id": 10
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/v1/edges")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&edge_req).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        // Get neighbors
        let app2 = build_router(state);
        let req = Request::builder()
            .uri("/api/v1/nodes/1/neighbors")
            .body(Body::empty())
            .unwrap();
        let resp = app2.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let neighbors: NeighborsResp = serde_json::from_slice(&body).unwrap();
        assert!(neighbors.neighbors.contains(&2));
    }

    #[tokio::test]
    async fn test_query_sql_via_api() {
        let state = test_state();
        setup_schema(&state);
        {
            let db = state.read();
            db.create_node(TypeId(1), vec![("name", Value::from("Alice")), ("age", Value::Int(30))]);
            db.create_node(TypeId(1), vec![("name", Value::from("Bob")), ("age", Value::Int(25))]);
        }

        let app = build_router(state);
        let q = serde_json::json!({
            "language": "sql",
            "query": "SELECT * FROM Person WHERE age > 20"
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/v1/query")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&q).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let result: QueryResp = serde_json::from_slice(&body).unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[tokio::test]
    async fn test_query_cypher_via_api() {
        let state = test_state();
        setup_schema(&state);
        {
            let db = state.read();
            db.create_node(TypeId(1), vec![("name", Value::from("Alice"))]);
        }

        let app = build_router(state);
        let q = serde_json::json!({
            "language": "cypher",
            "query": "MATCH (n:Person) RETURN n"
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/v1/query")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&q).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let result: QueryResp = serde_json::from_slice(&body).unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[tokio::test]
    async fn test_delete_node_via_api() {
        let state = test_state();
        setup_schema(&state);
        {
            let db = state.read();
            db.create_node(TypeId(1), vec![("name", Value::from("ToDelete"))]);
        }

        let app = build_router(state.clone());
        let req = Request::builder()
            .method(Method::DELETE)
            .uri("/api/v1/nodes/1")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify deleted
        assert_eq!(state.read().node_count(), 0);
    }

    #[tokio::test]
    async fn test_set_property_via_api() {
        let state = test_state();
        setup_schema(&state);
        {
            let db = state.read();
            db.create_node(TypeId(1), vec![("name", Value::from("Alice"))]);
        }

        let app = build_router(state.clone());
        let prop_req = serde_json::json!({
            "name": "age",
            "value": 30
        });
        let req = Request::builder()
            .method(Method::PUT)
            .uri("/api/v1/nodes/1/properties")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&prop_req).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_nonexistent_node_404() {
        let state = test_state();
        let app = build_router(state);
        let req = Request::builder()
            .uri("/api/v1/nodes/999")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_register_schema_via_api() {
        let state = test_state();
        let app = build_router(state.clone());

        let type_req = serde_json::json!({
            "name": "Movie",
            "type_id": 5
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/v1/schema/types")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&type_req).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn test_create_and_get_document() {
        let state = test_state();
        let app = build_router(state.clone());

        let doc_req = serde_json::json!({
            "collection": "articles",
            "fields": {
                "title": "Graph Databases",
                "year": 2024
            },
            "linked_node": null
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/v1/documents")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&doc_req).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let created: CreateDocResp = serde_json::from_slice(&body).unwrap();

        // Get document
        let app2 = build_router(state);
        let req = Request::builder()
            .uri(format!("/api/v1/documents/articles/{}", created.doc_id))
            .body(Body::empty())
            .unwrap();
        let resp = app2.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let doc: DocResp = serde_json::from_slice(&body).unwrap();
        assert_eq!(doc.fields.get("title"), Some(&serde_json::json!("Graph Databases")));
    }

    #[tokio::test]
    async fn test_invalid_query_language() {
        let state = test_state();
        let app = build_router(state);
        let q = serde_json::json!({
            "language": "sparql",
            "query": "SELECT ?x WHERE { ?x a :Person }"
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/v1/query")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&q).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        // Should return error for unsupported language
        assert_ne!(resp.status(), StatusCode::OK);
    }
}
