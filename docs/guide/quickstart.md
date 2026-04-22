# Quickstart

This quickstart focuses on the most concrete current BikoDB path: using the embedded `Database` API from Rust.

## 1. Build the workspace

```bash
git clone https://github.com/unnamedlab/BikoDB.git
cd BikoDB
cargo test --workspace
```

## 2. Create a tiny graph in-process

```rust
use bikodb_core::types::TypeId;
use bikodb_core::value::Value;
use bikodb_server::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Database::new();

    db.register_type("Person", TypeId(1));
    db.register_property("name", 0);
    db.register_property("age", 1);
    db.register_relationship("KNOWS", TypeId(10));

    let alice = db.create_node(TypeId(1), vec![
        ("name", Value::from("Alice")),
        ("age", Value::Int(30)),
    ]);

    let bob = db.create_node(TypeId(1), vec![
        ("name", Value::from("Bob")),
        ("age", Value::Int(25)),
    ]);

    db.create_edge(alice, bob, TypeId(10))?;

    let rows = db.query_sql("SELECT * FROM Person WHERE age > 20")?;
    println!("rows = {}", rows.len());

    Ok(())
}
```

This path demonstrates what BikoDB does best today:

- create typed graph nodes and edges,
- query the graph through the embedded facade,
- stay fully in-process without needing a packaged server.

## 3. Try the documented query subsets

### SQL read query

```sql
SELECT name, age FROM Person WHERE age > 20 ORDER BY age DESC LIMIT 10
```

### Cypher read query

```cypher
MATCH (p:Person)-[:KNOWS]->(f:Person)
RETURN p, f
```

### Gremlin traversal

```groovy
g.V().hasLabel('Person').out('KNOWS').values('name')
```

## 4. Optional: enable vector search

The vector path is currently best treated as experimental but usable for evaluation.

```rust
use bikodb_ai::embedding::DistanceMetric;

let mut db = Database::new();
db.enable_hnsw(3, DistanceMetric::Cosine);
```

Once HNSW is enabled you can insert vectors for nodes and run semantic search through the `Database` API.

## 5. Optional: use the HTTP router

If you need HTTP, use `bikodb_server::build_router` inside your own `axum` application.

Important limitation: the repo currently provides the router as a library component, not as a packaged server process with deployment docs.

## What to read next

- [Supported features & limitations →](/guide/supported-features)
- [Transactions & concurrency →](/architecture/transactions)
- [Multi-model consistency →](/architecture/multi-model-consistency)
