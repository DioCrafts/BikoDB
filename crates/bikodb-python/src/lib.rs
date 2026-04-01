// =============================================================================
// bikodb-python — Python bindings para BikoDB via FFI/JSON
// =============================================================================
// Proporciona una API C-compatible (extern "C") que un wrapper Python (ctypes
// o cffi) puede consumir directamente. Todas las funciones reciben/retornan
// JSON strings para máxima interoperabilidad.
//
// ## Arquitectura
// Python (ctypes/cffi) → C FFI (este crate) → Database (Rust)
//
// El patrón es:
//   1. Python envía un JSON string con la operación
//   2. Rust parsea, ejecuta, y retorna un JSON string con el resultado
//   3. Python deserializa el resultado
//
// ## Funciones expuestas
//   bikodb_create()           → Crear una instancia de Database
//   bikodb_destroy()          → Destruir una instancia
//   bikodb_execute()          → Ejecutar operación genérica (JSON in/out)
//   bikodb_free_string()      → Liberar string retornado por Rust
//
// ## Ejemplo Python
// ```python
// import ctypes, json
//
// lib = ctypes.CDLL("./libbikodb_python.dylib")
// lib.bikodb_create.restype = ctypes.c_void_p
// lib.bikodb_execute.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
// lib.bikodb_execute.restype = ctypes.c_char_p
// lib.bikodb_free_string.argtypes = [ctypes.c_char_p]
//
// db = lib.bikodb_create()
//
// # Register type
// cmd = json.dumps({"op": "register_type", "name": "Person", "type_id": 1})
// result = lib.bikodb_execute(db, cmd.encode())
// print(json.loads(result))
//
// # Create node
// cmd = json.dumps({"op": "create_node", "type_id": 1, "properties": {"name": "Alice"}})
// result = lib.bikodb_execute(db, cmd.encode())
// print(json.loads(result))
//
// # Query
// cmd = json.dumps({"op": "query", "language": "sql", "query": "SELECT * FROM Person"})
// result = lib.bikodb_execute(db, cmd.encode())
// print(json.loads(result))
//
// lib.bikodb_destroy(db)
// ```
// =============================================================================

use bikodb_core::record::Direction;
use bikodb_core::types::{NodeId, TypeId, EdgeId};
use bikodb_core::value::Value;
use bikodb_server::Database;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Arc;

/// Opaque handle para Database (thread-safe).
type DbHandle = Arc<parking_lot::RwLock<Database>>;

// ── FFI Functions ───────────────────────────────────────────────────────────

/// Crea una nueva instancia de Database. Retorna puntero opaco.
///
/// # Safety
/// El caller debe llamar `bikodb_destroy()` cuando termine.
#[no_mangle]
pub extern "C" fn bikodb_create() -> *mut DbHandle {
    let db = Arc::new(parking_lot::RwLock::new(Database::new()));
    Box::into_raw(Box::new(db))
}

/// Destruye una instancia de Database.
///
/// # Safety
/// `handle` debe ser un puntero válido retornado por `bikodb_create()`.
#[no_mangle]
pub unsafe extern "C" fn bikodb_destroy(handle: *mut DbHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)); }
    }
}

/// Ejecuta una operación sobre la base de datos.
///
/// Recibe un JSON string con la operación y retorna un JSON string con el resultado.
/// El caller debe liberar el string retornado con `bikodb_free_string()`.
///
/// # Safety
/// `handle` debe ser un puntero válido. `json_cmd` debe ser un C string válido UTF-8.
#[no_mangle]
pub unsafe extern "C" fn bikodb_execute(
    handle: *mut DbHandle,
    json_cmd: *const c_char,
) -> *mut c_char {
    if handle.is_null() || json_cmd.is_null() {
        return error_response("null handle or command");
    }

    let db_handle = unsafe { &*handle };
    let cmd_str = match unsafe { CStr::from_ptr(json_cmd) }.to_str() {
        Ok(s) => s,
        Err(_) => return error_response("invalid UTF-8 in command"),
    };

    let result = execute_command(db_handle, cmd_str);
    match CString::new(result) {
        Ok(cs) => cs.into_raw(),
        Err(_) => error_response("result contains null byte"),
    }
}

/// Libera un string retornado por `bikodb_execute()`.
///
/// # Safety
/// `ptr` debe ser un puntero retornado por `bikodb_execute()`.
#[no_mangle]
pub unsafe extern "C" fn bikodb_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { drop(CString::from_raw(ptr)); }
    }
}

// ── Command Dispatch ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Command {
    op: String,
    #[serde(flatten)]
    params: serde_json::Value,
}

fn error_response(msg: &str) -> *mut c_char {
    let resp = serde_json::json!({"ok": false, "error": msg});
    CString::new(resp.to_string()).unwrap_or_default().into_raw()
}

fn ok_json(data: serde_json::Value) -> String {
    serde_json::json!({"ok": true, "data": data}).to_string()
}

fn err_json(msg: &str) -> String {
    serde_json::json!({"ok": false, "error": msg}).to_string()
}

fn execute_command(db_handle: &DbHandle, json_str: &str) -> String {
    let cmd: Command = match serde_json::from_str(json_str) {
        Ok(c) => c,
        Err(e) => return err_json(&format!("JSON parse error: {}", e)),
    };

    match cmd.op.as_str() {
        "register_type" => cmd_register_type(db_handle, &cmd.params),
        "register_property" => cmd_register_property(db_handle, &cmd.params),
        "register_relationship" => cmd_register_relationship(db_handle, &cmd.params),
        "create_node" => cmd_create_node(db_handle, &cmd.params),
        "get_node" => cmd_get_node(db_handle, &cmd.params),
        "delete_node" => cmd_delete_node(db_handle, &cmd.params),
        "set_property" => cmd_set_property(db_handle, &cmd.params),
        "create_edge" => cmd_create_edge(db_handle, &cmd.params),
        "delete_edge" => cmd_delete_edge(db_handle, &cmd.params),
        "neighbors" => cmd_neighbors(db_handle, &cmd.params),
        "query" => cmd_query(db_handle, &cmd.params),
        "node_count" => { let db = db_handle.read(); ok_json(serde_json::json!(db.node_count())) },
        "edge_count" => { let db = db_handle.read(); ok_json(serde_json::json!(db.edge_count())) },
        "status" => cmd_status(db_handle),
        _ => err_json(&format!("Unknown operation: {}", cmd.op)),
    }
}

fn cmd_register_type(db: &DbHandle, params: &serde_json::Value) -> String {
    let name = params["name"].as_str().unwrap_or("");
    let type_id = params["type_id"].as_u64().unwrap_or(0) as u16;
    db.read().register_type(name, TypeId(type_id));
    ok_json(serde_json::json!(null))
}

fn cmd_register_property(db: &DbHandle, params: &serde_json::Value) -> String {
    let name = params["name"].as_str().unwrap_or("");
    let prop_id = params["prop_id"].as_u64().unwrap_or(0) as u16;
    db.read().register_property(name, prop_id);
    ok_json(serde_json::json!(null))
}

fn cmd_register_relationship(db: &DbHandle, params: &serde_json::Value) -> String {
    let name = params["name"].as_str().unwrap_or("");
    let type_id = params["type_id"].as_u64().unwrap_or(0) as u16;
    db.read().register_relationship(name, TypeId(type_id));
    ok_json(serde_json::json!(null))
}

fn cmd_create_node(db: &DbHandle, params: &serde_json::Value) -> String {
    let type_id = params["type_id"].as_u64().unwrap_or(0) as u16;
    let properties = match params.get("properties").and_then(|p| p.as_object()) {
        Some(obj) => {
            obj.iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect::<Vec<(String, Value)>>()
        }
        None => Vec::new(),
    };
    let db = db.read();
    let props: Vec<(&str, Value)> = properties.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    let node_id = db.create_node(TypeId(type_id), props);
    ok_json(serde_json::json!({"node_id": node_id.0}))
}

fn cmd_get_node(db: &DbHandle, params: &serde_json::Value) -> String {
    let id = params["id"].as_u64().unwrap_or(0);
    let db = db.read();
    match db.get_node(NodeId(id)) {
        Some(node) => {
            let prop_names = db.prop_names_snapshot();
            let mut props = HashMap::new();
            for (pid, val) in &node.properties {
                let name = prop_names.get(pid).cloned().unwrap_or_else(|| format!("prop_{}", pid));
                props.insert(name, value_to_json(val));
            }
            ok_json(serde_json::json!({
                "id": node.id.0,
                "type_id": node.type_id.0,
                "properties": props,
            }))
        }
        None => err_json(&format!("Node {} not found", id)),
    }
}

fn cmd_delete_node(db: &DbHandle, params: &serde_json::Value) -> String {
    let id = params["id"].as_u64().unwrap_or(0);
    match db.read().remove_node(NodeId(id)) {
        Ok(()) => ok_json(serde_json::json!(null)),
        Err(e) => err_json(&e.to_string()),
    }
}

fn cmd_set_property(db: &DbHandle, params: &serde_json::Value) -> String {
    let id = params["node_id"].as_u64().unwrap_or(0);
    let name = params["name"].as_str().unwrap_or("");
    let value = params.get("value").map(json_to_value).unwrap_or(Value::Null);
    match db.read().set_node_property(NodeId(id), name, value) {
        Ok(()) => ok_json(serde_json::json!(null)),
        Err(e) => err_json(&e.to_string()),
    }
}

fn cmd_create_edge(db: &DbHandle, params: &serde_json::Value) -> String {
    let source = params["source"].as_u64().unwrap_or(0);
    let target = params["target"].as_u64().unwrap_or(0);
    let type_id = params["type_id"].as_u64().unwrap_or(0) as u16;
    match db.read().create_edge(NodeId(source), NodeId(target), TypeId(type_id)) {
        Ok(eid) => ok_json(serde_json::json!({"edge_id": eid.0})),
        Err(e) => err_json(&e.to_string()),
    }
}

fn cmd_delete_edge(db: &DbHandle, params: &serde_json::Value) -> String {
    let id = params["id"].as_u64().unwrap_or(0);
    match db.read().remove_edge(EdgeId(id)) {
        Ok(()) => ok_json(serde_json::json!(null)),
        Err(e) => err_json(&e.to_string()),
    }
}

fn cmd_neighbors(db: &DbHandle, params: &serde_json::Value) -> String {
    let id = params["id"].as_u64().unwrap_or(0);
    let dir = match params.get("direction").and_then(|d| d.as_str()) {
        Some("in") => Direction::In,
        Some("both") => Direction::Both,
        _ => Direction::Out,
    };
    match db.read().neighbors(NodeId(id), dir) {
        Ok(neighbors) => {
            let ids: Vec<u64> = neighbors.iter().map(|n| n.0).collect();
            ok_json(serde_json::json!(ids))
        }
        Err(e) => err_json(&e.to_string()),
    }
}

fn cmd_query(db: &DbHandle, params: &serde_json::Value) -> String {
    let language = params["language"].as_str().unwrap_or("sql");
    let query = params["query"].as_str().unwrap_or("");
    let db = db.read();
    match db.query(language, query) {
        Ok(rows) => {
            let prop_names = db.prop_names_snapshot();
            let json_rows: Vec<serde_json::Value> = rows.iter().map(|r| {
                let mut props = HashMap::new();
                for (pid, val) in &r.properties {
                    let name = prop_names.get(pid).cloned().unwrap_or_else(|| format!("prop_{}", pid));
                    props.insert(name, value_to_json(val));
                }
                serde_json::json!({
                    "node_id": r.node_id.0,
                    "properties": props,
                })
            }).collect();
            ok_json(serde_json::json!(json_rows))
        }
        Err(e) => err_json(&e.to_string()),
    }
}

fn cmd_status(db: &DbHandle) -> String {
    let db = db.read();
    let pm = db.plugin_manager();
    ok_json(serde_json::json!({
        "node_count": db.node_count(),
        "edge_count": db.edge_count(),
        "plugins": pm.list_plugins().iter().map(|p| &p.name).collect::<Vec<_>>(),
        "udfs": pm.list_udfs(),
        "algorithms": pm.list_algorithms(),
    }))
}

// ── Value Conversion ────────────────────────────────────────────────────────

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
        serde_json::Value::Array(arr) => {
            Value::List(Box::new(arr.iter().map(json_to_value).collect()))
        }
        serde_json::Value::Object(obj) => {
            let map: HashMap<String, Value> = obj.iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect();
            Value::Map(Box::new(map))
        }
    }
}

fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::json!(*b),
        Value::Int(i) => serde_json::json!(*i),
        Value::Float(f) => serde_json::json!(*f),
        Value::String(s) => serde_json::json!(s),
        Value::List(list) => {
            serde_json::Value::Array(list.iter().map(value_to_json).collect())
        }
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

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn make_db() -> Box<DbHandle> {
        Box::new(Arc::new(parking_lot::RwLock::new(Database::new())))
    }

    #[test]
    fn test_create_and_destroy() {
        let handle = bikodb_create();
        assert!(!handle.is_null());
        unsafe { bikodb_destroy(handle); }
    }

    #[test]
    fn test_execute_register_type() {
        let db = make_db();
        let result = execute_command(&db, r#"{"op":"register_type","name":"Person","type_id":1}"#);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["ok"], true);
    }

    #[test]
    fn test_execute_create_and_get_node() {
        let db = make_db();
        execute_command(&db, r#"{"op":"register_type","name":"Person","type_id":1}"#);
        execute_command(&db, r#"{"op":"register_property","name":"name","prop_id":0}"#);

        let result = execute_command(&db, r#"{"op":"create_node","type_id":1,"properties":{"name":"Alice"}}"#);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["ok"], true);
        let node_id = parsed["data"]["node_id"].as_u64().unwrap();

        let result = execute_command(&db, &format!(r#"{{"op":"get_node","id":{}}}"#, node_id));
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["ok"], true);
        assert_eq!(parsed["data"]["properties"]["name"], "Alice");
    }

    #[test]
    fn test_execute_create_edge() {
        let db = make_db();
        execute_command(&db, r#"{"op":"register_type","name":"Person","type_id":1}"#);
        execute_command(&db, r#"{"op":"create_node","type_id":1,"properties":{}}"#);
        execute_command(&db, r#"{"op":"create_node","type_id":1,"properties":{}}"#);

        let result = execute_command(&db, r#"{"op":"create_edge","source":1,"target":2,"type_id":10}"#);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["ok"], true);
        assert!(parsed["data"]["edge_id"].as_u64().is_some());
    }

    #[test]
    fn test_execute_neighbors() {
        let db = make_db();
        execute_command(&db, r#"{"op":"register_type","name":"Person","type_id":1}"#);
        execute_command(&db, r#"{"op":"create_node","type_id":1,"properties":{}}"#);
        execute_command(&db, r#"{"op":"create_node","type_id":1,"properties":{}}"#);
        execute_command(&db, r#"{"op":"create_edge","source":1,"target":2,"type_id":10}"#);

        let result = execute_command(&db, r#"{"op":"neighbors","id":1,"direction":"out"}"#);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["ok"], true);
        let neighbors = parsed["data"].as_array().unwrap();
        assert!(neighbors.contains(&serde_json::json!(2)));
    }

    #[test]
    fn test_execute_query_sql() {
        let db = make_db();
        execute_command(&db, r#"{"op":"register_type","name":"Person","type_id":1}"#);
        execute_command(&db, r#"{"op":"register_property","name":"name","prop_id":0}"#);
        execute_command(&db, r#"{"op":"create_node","type_id":1,"properties":{"name":"Alice"}}"#);
        execute_command(&db, r#"{"op":"create_node","type_id":1,"properties":{"name":"Bob"}}"#);

        let result = execute_command(&db, r#"{"op":"query","language":"sql","query":"SELECT * FROM Person"}"#);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["ok"], true);
        let rows = parsed["data"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_execute_delete_node() {
        let db = make_db();
        execute_command(&db, r#"{"op":"register_type","name":"Person","type_id":1}"#);
        execute_command(&db, r#"{"op":"create_node","type_id":1,"properties":{}}"#);

        let count1 = execute_command(&db, r#"{"op":"node_count"}"#);
        let p1: serde_json::Value = serde_json::from_str(&count1).unwrap();
        assert_eq!(p1["data"], 1);

        execute_command(&db, r#"{"op":"delete_node","id":1}"#);

        let count2 = execute_command(&db, r#"{"op":"node_count"}"#);
        let p2: serde_json::Value = serde_json::from_str(&count2).unwrap();
        assert_eq!(p2["data"], 0);
    }

    #[test]
    fn test_execute_status() {
        let db = make_db();
        let result = execute_command(&db, r#"{"op":"status"}"#);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["ok"], true);
        assert_eq!(parsed["data"]["node_count"], 0);
    }

    #[test]
    fn test_execute_unknown_op() {
        let db = make_db();
        let result = execute_command(&db, r#"{"op":"unknown_op"}"#);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["ok"], false);
    }

    #[test]
    fn test_execute_invalid_json() {
        let db = make_db();
        let result = execute_command(&db, "not json at all");
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["ok"], false);
    }

    #[test]
    fn test_execute_set_property() {
        let db = make_db();
        execute_command(&db, r#"{"op":"register_type","name":"Person","type_id":1}"#);
        execute_command(&db, r#"{"op":"register_property","name":"age","prop_id":1}"#);
        execute_command(&db, r#"{"op":"create_node","type_id":1,"properties":{}}"#);

        let result = execute_command(&db, r#"{"op":"set_property","node_id":1,"name":"age","value":30}"#);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["ok"], true);

        let node = execute_command(&db, r#"{"op":"get_node","id":1}"#);
        let p: serde_json::Value = serde_json::from_str(&node).unwrap();
        assert_eq!(p["data"]["properties"]["age"], 30);
    }

    #[test]
    fn test_ffi_execute_via_c_api() {
        let handle = bikodb_create();
        assert!(!handle.is_null());

        let cmd = CString::new(r#"{"op":"register_type","name":"Test","type_id":1}"#).unwrap();
        let result = unsafe { bikodb_execute(handle, cmd.as_ptr()) };
        assert!(!result.is_null());

        let result_str = unsafe { CStr::from_ptr(result) }.to_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(result_str).unwrap();
        assert_eq!(parsed["ok"], true);

        unsafe { bikodb_free_string(result); }
        unsafe { bikodb_destroy(handle); }
    }
}
