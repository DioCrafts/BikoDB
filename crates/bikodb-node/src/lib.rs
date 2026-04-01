// =============================================================================
// bikodb-node — Node.js / C FFI bindings para BikoDB
// =============================================================================
// Proporciona una interfaz C FFI minimalista optimizada para consumo desde
// Node.js (via ffi-napi / koffi) o cualquier otro lenguaje que soporte C FFI.
//
// A diferencia del crate bikodb-python (que usa un dispatcher JSON genérico),
// este crate expone funciones C individuales para cada operación, lo que
// permite un wrapper más idiomático en JavaScript/TypeScript.
//
// ## Arquitectura
// Node.js (ffi-napi/koffi) → C FFI (este crate) → Database (Rust)
//
// ## Funciones expuestas
//   bikodb_node_create()              → Crear instancia
//   bikodb_node_destroy()             → Destruir instancia
//   bikodb_node_register_type()       → Registrar tipo
//   bikodb_node_register_property()   → Registrar propiedad
//   bikodb_node_add_node()            → Crear nodo
//   bikodb_node_get_node()            → Obtener nodo (JSON)
//   bikodb_node_delete_node()         → Eliminar nodo
//   bikodb_node_set_property()        → Establecer propiedad
//   bikodb_node_add_edge()            → Crear arista
//   bikodb_node_delete_edge()         → Eliminar arista
//   bikodb_node_query()               → Ejecutar query (JSON)
//   bikodb_node_count()               → Contar nodos
//   bikodb_node_edge_count()          → Contar aristas
//   bikodb_node_free_string()         → Liberar string Rust
//
// ## Ejemplo Node.js (koffi)
// ```js
// const koffi = require('koffi');
// const lib = koffi.load('./libbikodb_node.dylib');
//
// const bikodb_create = lib.func('void* bikodb_node_create()');
// const bikodb_destroy = lib.func('void bikodb_node_destroy(void*)');
// const bikodb_register_type = lib.func('int32 bikodb_node_register_type(void*, str, uint64)');
// const bikodb_add_node = lib.func('uint64 bikodb_node_add_node(void*, uint64, str)');
// const bikodb_query = lib.func('str bikodb_node_query(void*, str, str)');
// const bikodb_free = lib.func('void bikodb_node_free_string(str)');
//
// const db = bikodb_create();
// bikodb_register_type(db, "Person", 1);
// const id = bikodb_add_node(db, 1, '{"name":"Alice"}');
// const result = bikodb_query(db, "sql", "SELECT * FROM Person");
// console.log(JSON.parse(result));
// bikodb_free(result);
// bikodb_destroy(db);
// ```
// =============================================================================

use bikodb_core::types::{EdgeId, NodeId, TypeId};
use bikodb_core::value::Value;
use bikodb_server::Database;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Arc;

/// Opaque handle.
type DbHandle = Arc<parking_lot::RwLock<Database>>;

// ── Lifecycle ───────────────────────────────────────────────────────────────

/// Creates a new BikoDB instance. Returns opaque pointer.
///
/// # Safety
/// Caller must call `bikodb_node_destroy()` when done.
#[no_mangle]
pub extern "C" fn bikodb_node_create() -> *mut DbHandle {
    Box::into_raw(Box::new(Arc::new(parking_lot::RwLock::new(Database::new()))))
}

/// Destroys an BikoDB instance.
///
/// # Safety
/// `handle` must be a valid pointer from `bikodb_node_create()`.
#[no_mangle]
pub unsafe extern "C" fn bikodb_node_destroy(handle: *mut DbHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)); }
    }
}

// ── Schema ──────────────────────────────────────────────────────────────────

/// Register a type. Returns 0 on success.
///
/// # Safety
/// `handle` and `name` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn bikodb_node_register_type(
    handle: *mut DbHandle,
    name: *const c_char,
    type_id: u16,
) -> i32 {
    let Some(db) = safe_handle(handle) else { return -1 };
    let Some(name) = safe_str(name) else { return -1 };
    db.read().register_type(name, TypeId(type_id));
    0
}

/// Register a property. Returns 0 on success.
///
/// # Safety
/// `handle` and `name` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn bikodb_node_register_property(
    handle: *mut DbHandle,
    name: *const c_char,
    prop_id: u16,
) -> i32 {
    let Some(db) = safe_handle(handle) else { return -1 };
    let Some(name) = safe_str(name) else { return -1 };
    db.read().register_property(name, prop_id);
    0
}

// ── Nodes ───────────────────────────────────────────────────────────────────

/// Creates a node. `props_json` is a JSON object string with property values.
/// Returns node_id on success, 0 on error.
///
/// # Safety
/// `handle` and `props_json` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn bikodb_node_add_node(
    handle: *mut DbHandle,
    type_id: u16,
    props_json: *const c_char,
) -> u64 {
    let Some(db) = safe_handle(handle) else { return 0 };
    let Some(json_str) = safe_str(props_json) else { return 0 };

    let properties = parse_properties(json_str);
    let db = db.read();
    let props: Vec<(&str, Value)> = properties.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    db.create_node(TypeId(type_id), props).0
}

/// Gets a node by ID. Returns a JSON string that must be freed with `bikodb_node_free_string()`.
/// Returns null on error.
///
/// # Safety
/// `handle` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn bikodb_node_get_node(
    handle: *mut DbHandle,
    node_id: u64,
) -> *mut c_char {
    let Some(db) = safe_handle(handle) else { return std::ptr::null_mut() };
    let db = db.read();
    match db.get_node(NodeId(node_id)) {
        Some(node) => {
            let prop_names = db.prop_names_snapshot();
            let mut props = serde_json::Map::new();
            for (pid, val) in &node.properties {
                let name = prop_names.get(pid).cloned().unwrap_or_else(|| format!("prop_{}", pid));
                props.insert(name, value_to_json(val));
            }
            let result = serde_json::json!({
                "id": node.id.0,
                "type_id": node.type_id.0,
                "properties": props,
            });
            to_c_string(&result.to_string())
        }
        None => std::ptr::null_mut(),
    }
}

/// Deletes a node. Returns 0 on success, -1 on error.
///
/// # Safety
/// `handle` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn bikodb_node_delete_node(
    handle: *mut DbHandle,
    node_id: u64,
) -> i32 {
    let Some(db) = safe_handle(handle) else { return -1 };
    match db.read().remove_node(NodeId(node_id)) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Sets a property on a node. `value_json` is a JSON value.
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// `handle`, `prop_name`, and `value_json` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn bikodb_node_set_property(
    handle: *mut DbHandle,
    node_id: u64,
    prop_name: *const c_char,
    value_json: *const c_char,
) -> i32 {
    let Some(db) = safe_handle(handle) else { return -1 };
    let Some(name) = safe_str(prop_name) else { return -1 };
    let Some(val_str) = safe_str(value_json) else { return -1 };

    let value: serde_json::Value = match serde_json::from_str(val_str) {
        Ok(v) => v,
        Err(_) => return -1,
    };
    let oxi_val = json_to_value(&value);
    match db.read().set_node_property(NodeId(node_id), name, oxi_val) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

// ── Edges ───────────────────────────────────────────────────────────────────

/// Creates an edge. Returns edge_id on success, 0 on error.
///
/// # Safety
/// `handle` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn bikodb_node_add_edge(
    handle: *mut DbHandle,
    source: u64,
    target: u64,
    type_id: u16,
) -> u64 {
    let Some(db) = safe_handle(handle) else { return 0 };
    match db.read().create_edge(NodeId(source), NodeId(target), TypeId(type_id)) {
        Ok(eid) => eid.0,
        Err(_) => 0,
    }
}

/// Deletes an edge. Returns 0 on success, -1 on error.
///
/// # Safety
/// `handle` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn bikodb_node_delete_edge(
    handle: *mut DbHandle,
    edge_id: u64,
) -> i32 {
    let Some(db) = safe_handle(handle) else { return -1 };
    match db.read().remove_edge(EdgeId(edge_id)) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

// ── Query ───────────────────────────────────────────────────────────────────

/// Runs a query (SQL/Cypher/Gremlin). Returns JSON array string.
/// Must be freed with `bikodb_node_free_string()`.
///
/// # Safety
/// `handle`, `language`, and `query` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn bikodb_node_query(
    handle: *mut DbHandle,
    language: *const c_char,
    query: *const c_char,
) -> *mut c_char {
    let Some(db) = safe_handle(handle) else { return std::ptr::null_mut() };
    let Some(lang) = safe_str(language) else { return std::ptr::null_mut() };
    let Some(q) = safe_str(query) else { return std::ptr::null_mut() };

    let db = db.read();
    match db.query(lang, q) {
        Ok(rows) => {
            let prop_names = db.prop_names_snapshot();
            let json_rows: Vec<serde_json::Value> = rows.iter().map(|r| {
                let mut props = serde_json::Map::new();
                for (pid, val) in &r.properties {
                    let name = prop_names.get(pid).cloned().unwrap_or_else(|| format!("prop_{}", pid));
                    props.insert(name, value_to_json(val));
                }
                serde_json::json!({
                    "node_id": r.node_id.0,
                    "properties": props,
                })
            }).collect();
            to_c_string(&serde_json::json!(json_rows).to_string())
        }
        Err(e) => {
            to_c_string(&serde_json::json!({"error": e.to_string()}).to_string())
        }
    }
}

// ── Counts ──────────────────────────────────────────────────────────────────

/// Returns node count.
///
/// # Safety
/// `handle` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn bikodb_node_count(handle: *mut DbHandle) -> u64 {
    let Some(db) = safe_handle(handle) else { return 0 };
    db.read().node_count() as u64
}

/// Returns edge count.
///
/// # Safety
/// `handle` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn bikodb_node_edge_count(handle: *mut DbHandle) -> u64 {
    let Some(db) = safe_handle(handle) else { return 0 };
    db.read().edge_count() as u64
}

// ── String Management ───────────────────────────────────────────────────────

/// Frees a string returned by any bikodb_node_* function.
///
/// # Safety
/// `ptr` must be from an bikodb_node_* function or null.
#[no_mangle]
pub unsafe extern "C" fn bikodb_node_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { drop(CString::from_raw(ptr)); }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

unsafe fn safe_handle(handle: *mut DbHandle) -> Option<&'static DbHandle> {
    if handle.is_null() {
        None
    } else {
        unsafe { Some(&*handle) }
    }
}

unsafe fn safe_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(ptr) }.to_str().ok()
    }
}

fn to_c_string(s: &str) -> *mut c_char {
    CString::new(s).map(CString::into_raw).unwrap_or(std::ptr::null_mut())
}

fn parse_properties(json_str: &str) -> Vec<(String, Value)> {
    let parsed: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    match parsed.as_object() {
        Some(obj) => obj.iter().map(|(k, v)| (k.clone(), json_to_value(v))).collect(),
        None => Vec::new(),
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
    fn test_lifecycle() {
        let handle = bikodb_node_create();
        assert!(!handle.is_null());
        unsafe { bikodb_node_destroy(handle); }
    }

    #[test]
    fn test_register_type() {
        let handle = bikodb_node_create();
        let name = CString::new("Person").unwrap();
        let ret = unsafe { bikodb_node_register_type(handle, name.as_ptr(), 1) };
        assert_eq!(ret, 0);
        unsafe { bikodb_node_destroy(handle); }
    }

    #[test]
    fn test_create_and_get_node() {
        let handle = bikodb_node_create();
        let type_name = CString::new("Person").unwrap();
        let prop_name = CString::new("name").unwrap();
        unsafe { bikodb_node_register_type(handle, type_name.as_ptr(), 1); }
        unsafe { bikodb_node_register_property(handle, prop_name.as_ptr(), 0); }

        let props = CString::new(r#"{"name":"Alice"}"#).unwrap();
        let nid = unsafe { bikodb_node_add_node(handle, 1, props.as_ptr()) };
        assert!(nid > 0);

        let json_ptr = unsafe { bikodb_node_get_node(handle, nid) };
        assert!(!json_ptr.is_null());
        let json_str = unsafe { CStr::from_ptr(json_ptr) }.to_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed["id"], nid);
        assert_eq!(parsed["properties"]["name"], "Alice");

        unsafe { bikodb_node_free_string(json_ptr); }
        unsafe { bikodb_node_destroy(handle); }
    }

    #[test]
    fn test_create_edge_and_counts() {
        let handle = bikodb_node_create();
        let name = CString::new("Person").unwrap();
        unsafe { bikodb_node_register_type(handle, name.as_ptr(), 1); }

        let empty = CString::new("{}").unwrap();
        let n1 = unsafe { bikodb_node_add_node(handle, 1, empty.as_ptr()) };
        let n2 = unsafe { bikodb_node_add_node(handle, 1, empty.as_ptr()) };
        assert!(n1 > 0 && n2 > 0);

        let eid = unsafe { bikodb_node_add_edge(handle, n1, n2, 10) };
        assert!(eid > 0);

        let nc = unsafe { bikodb_node_count(handle) };
        assert_eq!(nc, 2);
        let ec = unsafe { bikodb_node_edge_count(handle) };
        assert_eq!(ec, 1);

        unsafe { bikodb_node_destroy(handle); }
    }

    #[test]
    fn test_delete_node() {
        let handle = bikodb_node_create();
        let name = CString::new("T").unwrap();
        unsafe { bikodb_node_register_type(handle, name.as_ptr(), 1); }
        let empty = CString::new("{}").unwrap();
        unsafe { bikodb_node_add_node(handle, 1, empty.as_ptr()); }

        assert_eq!(unsafe { bikodb_node_count(handle) }, 1);
        let ret = unsafe { bikodb_node_delete_node(handle, 1) };
        assert_eq!(ret, 0);
        assert_eq!(unsafe { bikodb_node_count(handle) }, 0);

        unsafe { bikodb_node_destroy(handle); }
    }

    #[test]
    fn test_set_property() {
        let handle = bikodb_node_create();
        let tn = CString::new("T").unwrap();
        let pn = CString::new("age").unwrap();
        unsafe {
            bikodb_node_register_type(handle, tn.as_ptr(), 1);
            bikodb_node_register_property(handle, pn.as_ptr(), 1);
        }
        let empty = CString::new("{}").unwrap();
        let nid = unsafe { bikodb_node_add_node(handle, 1, empty.as_ptr()) };

        let prop = CString::new("age").unwrap();
        let val = CString::new("25").unwrap();
        let ret = unsafe { bikodb_node_set_property(handle, nid, prop.as_ptr(), val.as_ptr()) };
        assert_eq!(ret, 0);

        let json_ptr = unsafe { bikodb_node_get_node(handle, nid) };
        let json_str = unsafe { CStr::from_ptr(json_ptr) }.to_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed["properties"]["age"], 25);

        unsafe { bikodb_node_free_string(json_ptr); }
        unsafe { bikodb_node_destroy(handle); }
    }

    #[test]
    fn test_query_sql() {
        let handle = bikodb_node_create();
        let tn = CString::new("Person").unwrap();
        let pn = CString::new("name").unwrap();
        unsafe {
            bikodb_node_register_type(handle, tn.as_ptr(), 1);
            bikodb_node_register_property(handle, pn.as_ptr(), 0);
        }

        let p1 = CString::new(r#"{"name":"Alice"}"#).unwrap();
        let p2 = CString::new(r#"{"name":"Bob"}"#).unwrap();
        unsafe {
            bikodb_node_add_node(handle, 1, p1.as_ptr());
            bikodb_node_add_node(handle, 1, p2.as_ptr());
        }

        let lang = CString::new("sql").unwrap();
        let q = CString::new("SELECT * FROM Person").unwrap();
        let result_ptr = unsafe { bikodb_node_query(handle, lang.as_ptr(), q.as_ptr()) };
        assert!(!result_ptr.is_null());

        let result_str = unsafe { CStr::from_ptr(result_ptr) }.to_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(result_str).unwrap();
        assert!(parsed.as_array().unwrap().len() == 2);

        unsafe { bikodb_node_free_string(result_ptr); }
        unsafe { bikodb_node_destroy(handle); }
    }

    #[test]
    fn test_null_handle_safety() {
        let null_handle: *mut DbHandle = std::ptr::null_mut();
        assert_eq!(unsafe { bikodb_node_count(null_handle) }, 0);

        let result = unsafe { bikodb_node_get_node(null_handle, 1) };
        assert!(result.is_null());

        let name = CString::new("test").unwrap();
        assert_eq!(unsafe { bikodb_node_register_type(null_handle, name.as_ptr(), 1) }, -1);
    }

    #[test]
    fn test_delete_edge() {
        let handle = bikodb_node_create();
        let name = CString::new("T").unwrap();
        unsafe { bikodb_node_register_type(handle, name.as_ptr(), 1); }
        let empty = CString::new("{}").unwrap();
        unsafe {
            bikodb_node_add_node(handle, 1, empty.as_ptr());
            bikodb_node_add_node(handle, 1, empty.as_ptr());
        }
        let eid = unsafe { bikodb_node_add_edge(handle, 1, 2, 10) };
        assert!(eid > 0);
        assert_eq!(unsafe { bikodb_node_edge_count(handle) }, 1);

        let ret = unsafe { bikodb_node_delete_edge(handle, eid) };
        assert_eq!(ret, 0);
        assert_eq!(unsafe { bikodb_node_edge_count(handle) }, 0);

        unsafe { bikodb_node_destroy(handle); }
    }
}
