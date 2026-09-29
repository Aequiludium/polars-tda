# Scope and next decisions

## Implemented today

`features(*coordinates, method="exact" | "approximate")` is the primary entry
point. Fixed presets produce ten numeric H0/H1 fields from one native persistence
computation. `method` is the only computation option. Coordinate inputs must
currently be finite, non-null Float64; other dtypes require explicit casts.

- One row per point, multiple coordinate expressions per cloud, including selectors.
- Scalar aggregation with select, grouping and Polars window contexts; eager and lazy execution.
- Exact and approximate share the same feature Struct schema.
- Advanced `rips` supports `method="exact" | "sparse"`; sparse requires epsilon.
  Its metric policy defaults to unchecked, with no reported approximation bound.
- Versioned diagram Structs support storage and later descriptor calculations.
- `betti_curve` and `lifetime_summary` reuse diagrams without repeating persistence.
- Exact borrows coordinate chunks. Sparse uses native points, packing O(n*d)
  coordinates when required instead of using the callback's quadratic distance cache.
- Published cocycle 0.1.1, pinned exactly from crates.io on the development branch,
  without kernel modifications or copied algorithms.

See the [usage guide](usage.md), [feature dictionary](features.md),
[expression contract](expression-contract.md) and [advanced API](advanced-api.md).

## Proposed next work, not available API

1. Support additional numeric coordinate types without copying existing Float64
   inputs unnecessarily. Define supported types and integer precision behavior
   before implementation; do not silently accept strings or discard invalid points.
2. Review the default feature bundle for analyst use: interpretation, names,
   diagnostic counts and useful model inputs. Keep the current ten fields until
   an explicit, documented schema decision is implemented.
3. Add descriptors according to downstream needs. Prefer reusing stored diagrams
   or existing expressions when possible; separate point-cloud expressions must
   not imply that expensive work is automatically shared.
4. Expand representative performance measurements across dimensions and group
   sizes. Track the tested, unapplied candidate in [sparse profiling](sparse-profiling.md);
   measure total runtime and peak memory as well as input copies.

The [reference-library review](design-rationale.md) explains these proposals and
which current choices it supports. None of them adds a new parameter to the main
`features` API today.

## Deferred until a concrete workflow needs them

- Nested clouds, supplied distance matrices, diagram distances and representatives.
- Additional sparse sampling controls or a namespace/wrapper surface.
- Execution controls with explicit cancellation and resource guarantees; separate
  kernel prepare/compute calls have independent work budgets.

Kernel contributors need not implement plugin APIs for each algorithm. New public
operations are selected and adapted in this repository. Exact computation never
silently becomes approximate. No placeholder functions reserve these future APIs.
