# Documentation

polars-tda turns coordinate columns into topology features through native Polars
expressions. The package is pre-alpha. The guides below describe implemented
behavior; proposed changes are explicitly marked and are not available options.

## For data analysts

1. [Quick start](../README.md#quick-start): compute a feature table.
2. [Using expressions](usage.md): prepare coordinates, aggregate groups, select
   model columns and reuse results.
3. [Feature dictionary](features.md): meanings, units, undefined values and presets.
4. [Expression contract](expression-contract.md): selectors, windows, output
   shapes and compatibility conventions.

## For advanced workflows

- [Diagram APIs](advanced-api.md): compute and store diagrams, then reuse them.
- [Diagram schema](result-schema.md): interval endpoints, coverage and provenance.

## For contributors

- [Contributing](../CONTRIBUTING.md): issues, pull requests, style and validation.

- [Design rationale](design-rationale.md): reference-library comparison,
  performance boundaries and how new operations fit into the project.
- [Scope and next decisions](api-discussion.md): current support and open work.
- [Development](../README.md#development): build and validation commands.
- [Initial benchmarks](benchmark-notes.md) and [sparse profiling](sparse-profiling.md):
  measured workloads and the unapplied kernel experiment.

The main API has one computation option, `method="exact" | "approximate"`.
The advanced `rips` API uses `method="exact" | "sparse"` and explicit mathematical
settings. These are different interfaces; advanced settings are not arguments
accepted by `features`.
