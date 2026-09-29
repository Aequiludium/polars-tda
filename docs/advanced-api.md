# Advanced diagram workflows

For routine analysis use `polars_tda.features`. These lower-level operations
remain available when a workflow needs stored diagrams, custom dimensions or
explicit approximation settings. They are not required for the main feature API.
For ordinary feature tables and field selection, start with the [usage guide](usage.md).

## Usage

Each row is a point and each selected Float64 column is a coordinate. Aggregate
one cloud with `select`, or multiple clouds with `group_by().agg()`:

```python
import polars as pl
import polars_tda as tda

points = pl.DataFrame(
    {
        "sample": ["square"] * 4,
        "x": [0.0, 1.0, 1.0, 0.0],
        "y": [0.0, 0.0, 1.0, 1.0],
    }
)

diagrams = (
    points.lazy()
    .group_by("sample")
    .agg(tda.rips("x", "y", max_dimension=1).alias("diagram"))
)
features = diagrams.with_columns(
    tda.betti_curve("diagram", dimension=1, scales=[0.0, 1.0, 2.0]).alias("h1"),
    tda.lifetime_summary("diagram", dimension=1).alias("h1_lifetimes"),
).collect()
```

The square has one H1 interval `[1, sqrt(2))`, so `h1` is `[0, 1, 0]` and its
finite total persistence is `sqrt(2) - 1`. Expand summary fields with
`features.unnest("h1_lifetimes")`. For one cloud use
`points.select(tda.rips("x", "y").alias("diagram"))`.

The result is an ordinary Arrow Struct column. Save it to Parquet and reuse
`betti_curve` or `lifetime_summary` later without recomputing persistence. To
materialize once for multiple independent queries, collect/store `diagrams`
first; collecting the same lazy plan multiple times reruns it.

## API and semantics

- `rips(*coordinates, method="exact", max_dimension=1, max_edge_length=None, coefficient=2)`
  computes ordinary exact Euclidean Rips homology from H0 through the inclusive
  maximum dimension, over the selected prime field. Scales are edge lengths,
  not ball radii. A cutoff is inclusive. No normalization or sampling is applied.
- `betti_curve(diagram, *, dimension, scales)` returns `List(UInt64)` in the order
  of the finite, strictly increasing grid. Births are inclusive and finite deaths
  exclusive. Queries beyond known coverage or in uncomputed dimensions fail.
- `lifetime_summary(diagram, *, dimension)` returns finite count, total and
  maximum persistence, natural-log entropy, and counts of excluded essential and
  censored intervals. With no finite intervals, total is zero; maximum and entropy
  are null. A null diagram propagates to null in both feature expressions.

Coordinates must be finite, non-null Float64 values. Explicitly cast other dtypes,
for example `tda.rips(pl.col("x").cast(pl.Float64))`; conversion may allocate or
lose integer precision. Duplicates are passed through to the kernel, which omits
zero-length finite intervals. An empty cloud yields an empty diagram
with the requested dimensions recorded as computed.

See [result schema](result-schema.md) for endpoint and coverage semantics.
Nested clouds, distance-matrix inputs, representatives and diagram distances
are not yet part of the public API.

## Sparse Rips

Use the same expression and downstream features:

```python
sparse = points.select(
    tda.rips("x", "y", method="sparse", epsilon=0.5).alias("diagram")
)
```

`method: Literal["exact", "sparse"]` defaults to exact. Sparse requires an explicit
finite `epsilon > 0`. It uses the kernel's native sparse construction and
higher-simplex blocker. The adapter never substitutes an ordinary flag complex.

- Exact uses `max_edge_length`; sparse uses `max_filtration_value` to cap modified
  filtration values. Supplying options for the wrong method raises an error.
- Sparse `metric_policy: Literal["check", "assume", "unchecked"]` defaults to
  `unchecked`, which skips triangle checks and reports no approximation bound.
  `assume` explicitly asserts the metric hypothesis. `check` examines every
  triangle without tolerance, costs O(n³ d) for d coordinates, and can reject
  floating-point Euclidean distances due to rounding.
- For `epsilon < 1` with `check` or `assume`, metadata records the kernel's
  conditional ideal-arithmetic factor `1/(1-epsilon)`. It is not an absolute
  diagram-distance guarantee or a floating-point error certificate. For
  `epsilon >= 1`, no bound is reported under any policy.
- Coverage, essentiality, Betti curves and lifetime summaries all refer to the
  **approximate filtration**, not the exact Rips filtration. A complete sparse
  result does not assert that it equals the exact result.
- Sampling starts at group-local vertex zero, with deterministic original-ID
  tie-breaking. Zero-distance duplicates are omitted by the kernel. The result
  retains the permutation, insertion radii and retained original vertex IDs.
  Input row order can affect sparse sampling; sort first when that matters.

## Memory and execution

For exact Rips, the adapter borrows Float64 buffers, including slices and multiple
chunks, without packing coordinates. Neither method calls NumPy. It allocates small per-column chunk
metadata. Polars itself may gather rows for groups or allocate cast results.

- Exact without a cutoff: compute one condensed `n*(n-1)/2` Float64 distance buffer
  and lend it to the kernel. Its values alone occupy `4*n*(n-1)` bytes.
- Exact with a cutoff: compute distances through a callback and retain threshold edges,
  with O(n) point-index storage. All unordered pairs are still visited. A large
  cutoff may retain O(n²) edges; this is not a spatial neighbor search.
- Sparse: borrow a single contiguous coordinate, otherwise pack O(nd) row-major
  coordinates for the native point kernel. This avoids the sparse callback
  interface's O(n²) distance cache. Sampling still uses O(n²) distance evaluations,
  graph storage can be O(n²), and metric checking adds O(n³) evaluations.
- Reduction workspaces and owned outputs also allocate. No whole-pipeline
  zero-allocation, memory cap or speedup guarantee is claimed.
- Feature expressions validate and reconstruct interval objects and use kernel
  descriptors. This costs time and memory proportional to the diagram, without
  repeating persistent homology. Betti event sorting adds O(m log m) work.
