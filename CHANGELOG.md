# Changelog

Changes to public behavior, result schemas and computation presets are recorded
here. GitHub releases link to immutable version tags; unreleased work belongs
under the next development section. The project currently has a pre-alpha API.

## Unreleased

- Replace the development branch's Git-pinned kernel with the exact published
  crates.io dependency `cocycle = "=0.1.1"` and its registry checksum.
- Adapt diagram serialization to the kernel's logical interval iterator by using
  `PersistenceDiagram::len()` for the output length. Python expressions, Arrow
  schemas and computation presets remain unchanged; no intermediate interval
  collection is introduced.
- Keep the published polars-tda `v0.1.0` tag and its historical Git dependency
  unchanged. This kernel update does not create a new polars-tda release.

## 0.1.0 — 2026-09-29 (GitHub pre-release)

First source preview of the point cloud → persistence → topology feature pipeline.

### Included

- Native `features(*coordinates, method="exact" | "approximate")` expressions
  produce ten numeric H0/H1 fields from one persistence computation per cloud.
- Column names, expressions and selectors work in Polars aggregation contexts,
  including grouped, lazy and window queries.
- Advanced `rips`, `lifetime_summary` and `betti_curve` operations support stored
  diagrams, versioned schemas and explicit approximation settings.
- Float64 coordinate borrowing for exact computation and native point input for
  sparse computation, backed by a pinned cocycle Git dependency.
- User guides, feature definitions, contributor guidance, reproducible benchmarks,
  GitHub issue/PR templates and build/test CI.

### Preview limitations

- Inputs currently require finite, non-null Float64 coordinates; other numeric
  types need explicit casts. The API and feature bundle are not yet stable.
- Approximate results may differ and are not guaranteed to be faster. Neither
  mode promises bounded-memory or incremental processing.
- GitHub provides source archives. No PyPI distribution or cross-platform binary
  wheel set is published. Linux x86_64 is the current validation platform.
- The benchmarked kernel optimization patch is an unapplied experiment; the
  production kernel dependency remains unchanged.

[Source and release notes](https://github.com/Aequiludium/polars-tda/releases/tag/v0.1.0)
