---
layout: home

hero:
  name: "BikoDB"
  text: "Multi-model graph runtime"
  tagline: "Graph, document, and vector primitives in one Rust codebase — with scope documented honestly."
  image:
    src: /logo.png
    alt: BikoDB logo
  actions:
    - theme: brand
      text: Get Started
      link: /guide/installation
    - theme: alt
      text: Quickstart
      link: /guide/quickstart
    - theme: alt
      text: View on GitHub
      link: https://github.com/unnamedlab/BikoDB

features:
  - icon: 🧠
    title: Graph-first core
    details: Concurrent graph storage, graph algorithms, and an embedded Database facade are the most concrete parts of the current project.
  - icon: 🔎
    title: Documented query subset
    details: SQL, Cypher, and Gremlin are supported as explicit subsets instead of being presented as full language implementations.
  - icon: 🧱
    title: Multi-model building blocks
    details: Documents, HNSW vector search, plugins, and live-query/event hooks exist today, but not all combinations share the same maturity.
  - icon: 🔐
    title: Transaction semantics explained
    details: The docs spell out what the graph transaction layer guarantees today and where conflict detection or cross-model atomicity is still limited.
  - icon: 📊
    title: Benchmarks with caveats
    details: Reproduction steps, current assumptions, and limits of the published comparison data are documented up front.
  - icon: ⚠️
    title: Stable vs experimental
    details: Each major module is classified as stable, experimental, or planned so external users can judge adoption risk quickly.
---
