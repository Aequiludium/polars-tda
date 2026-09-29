# Feature dictionary

`features(*coordinates, method="exact")` aggregates one point cloud into a Struct
of ten numeric fields. Use `select` for a whole frame or `group_by().agg()` for
one cloud per entity/window. `unnest` turns the Struct into ordinary columns.

`method: Literal["exact", "approximate"]` is the only computation setting. The
coordinate arguments are column names or Polars expressions. Multi-column
expressions and selectors are dimensions of one cloud. See the
[expression contract](expression-contract.md) for context and composition rules. No persistent
homology terminology is required to call the function.

## Columns and interpretation

Every suffix below occurs with both prefixes: `h0_` for connected components and
`h1_` for loops. These describe structures appearing and disappearing as the
connection distance increases. A lifetime is the disappearance distance minus
the appearance distance.

| Suffix | Dtype | Definition | No finite intervals |
| --- | --- | --- | --- |
| finite_count | UInt64 | Number of finite, positive-length intervals | 0 |
| total_persistence | Float64 | Sum of finite lifetimes | 0.0 |
| max_persistence | Float64 | Maximum finite lifetime | null |
| entropy | Float64 | Shannon entropy of normalized finite lifetimes, natural log | null |
| essential_count | UInt64 | Number of intervals that never end | Count independently |

For example, `h0_total_persistence` sums the lifetimes of components before they
merge. `h1_max_persistence` is the longest finite loop lifetime. Neither is a
universal quality score: interpret them in the context of the data's sampling,
units, noise and preprocessing. `h1_finite_count` counts intervals across scales;
it is not the number of loops at a particular scale. Use the advanced Betti
curve operation if that quantity is needed.

Never-ending intervals are excluded from all finite lifetime statistics and
counted separately. Zero-length finite intervals are omitted by the kernel. One
finite interval has entropy zero. Empty clouds produce count/total zero and
maximum/entropy null in both dimensions. A singleton or coincident-point cloud
has one essential H0 interval and no finite intervals. Empty groups produce no
rows, whereas selecting an empty frame computes one empty cloud.

## Choosing model inputs

The current bundle includes both descriptive measurements and diagnostic counts;
it is not a recommendation to feed all ten columns into every model.

- `h0_max_persistence` measures the longest-lived finite separation before
  components merge. Its value depends on coordinate scale and can reflect outliers.
- `h1_max_persistence` measures the longest-lived finite loop. It is not a
  confidence probability or a direct measure of data quality.
- `finite_count` counts positive-length intervals over the filtration. It is
  sensitive to sampling and is not a count at one selected connection distance.
- `entropy` describes how finite lifetimes are distributed, not their absolute
  scale. Its interpretation also depends on how many intervals exist.
- `essential_count` reports intervals excluded from finite lifetime summaries.
  For a nonempty finite cloud under the full exact Rips filtration, H0 has one
  essential interval and H1 has none: eventually the complex is a full simplex.
  These fields are primarily diagnostic in that setting, not varying predictors.
  For approximate output, interpret them in its own filtration as documented below.

Retain the existing fields when storing results; choose model columns explicitly
with Polars. See the [usage example](usage.md). Revisiting default field membership
and names before a stable release is [proposed work](api-discussion.md), not an
implemented schema change. Null does not mean zero persistence: a maximum can be
undefined because no finite intervals exist.

## Coordinate preparation

Inputs must be Float64 without null, NaN or infinite coordinates. Use explicit
Polars casts for other dtypes, for example `tda.features(pl.col("x").cast(pl.Float64))`.
Float64-only is the current implementation limit; automatic numeric conversion
is not yet supported. Casting large integers can lose precision even when the
cast succeeds. The library neither drops rows nor
standardizes coordinates silently.

Use comparable units across coordinate columns and the same preprocessing across
groups. Uniformly scaling all coordinates multiplies lifetime totals and maxima;
finite counts and normalized lifetime entropy are unchanged in exact arithmetic.
Different point counts and sampling densities can also affect the features.
Add ordinary Polars counts or other covariates to your analysis when needed.

For approximate computation, row order can affect deterministic sparse sampling.
Sort rows consistently within entities if repeatability across reordered inputs
matters. Record the package version and selected method with generated features.

## Library-managed presets

These details document reproducibility; they are not extra arguments analysts
must configure. Both presets use Euclidean distance, ordinary H0/H1 persistence
over F2, edge-length scales and the entire filtration. There is no cutoff, so
no censored intervals are omitted from the feature table.

| Method | Implementation in this version |
| --- | --- |
| exact (default) | Exact Rips with borrowed coordinates and condensed distances |
| approximate | Native sparse Rips, epsilon 0.5, start vertex zero, no positive-radius subsampling, unchecked metric policy |

The approximate preset follows the same kernel contract as the advanced sparse
API. It reports no certified approximation bound, and feature values need not
match the exact preset. No per-feature error guarantee is implied. It avoids a
quadratic distance cache but still allocates graph/reduction storage; on measured
small clouds it is slower than exact. See [measurements](benchmark-notes.md).

Preset/schema changes must be documented with a package version; they must not
vary silently by cloud size or machine. The current implementation uses the
pinned kernel revision and the documented fixed settings for every cloud.
