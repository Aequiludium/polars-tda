# Public expression contract

This document defines the current analyst-facing conventions. The package is
pre-alpha: these rules are a compatibility target for a stable release, not a
claim that every proposed future operation is implemented. The main API remains:

```python
def features(
    *coordinates: str | pl.Expr,
    method: Literal["exact", "approximate"] = "exact",
) -> pl.Expr: ...
```

These are project conventions, not universal requirements for Polars plugins.
Their rationale and proposed refinements are recorded in the
[design review](design-rationale.md).

`method` is the only computation option. Context, column selection, scaling,
filtering, renaming and projection use ordinary Polars operations.

## 1. Data shape is explicit

One row is one point; selected columns are its coordinates. All arguments jointly
form one cloud in the current aggregation context. Strings denote column names.
Expressions may transform coordinates but must preserve alignment across columns.
The native call checks equal lengths; equal length alone cannot establish that
independently filtered columns still refer to the same source rows. Filter the
whole frame or use the same row selection for every coordinate.

These forms describe the same coordinate set:

```python
tda.features("x", "y")
tda.features(*["x", "y"])
tda.features(pl.col(["x", "y"]))
```

Multi-column expressions, regexes and selectors expand into dimensions of a
single cloud. They do not generate a separate analysis per column. Column
selection follows the resolved Polars expression order. Group keys, IDs and
other numeric covariates are not automatically excluded from a selector: select
only intended coordinates. A selector matching no coordinates is an error.

The current coordinate dtype is Float64. Explicit casts use Polars expressions;
there is no hidden coercion, row deletion, standardization, deduplication or
literal broadcasting by the adapter. Scalar expressions mixed with multi-row
coordinates must be expanded by the caller first. List/Array/Struct point-cloud
storage and distance matrices are not alternate interpretations of this input.
Float64-only is the current implementation limit, not a permanent requirement:
[broader numeric input support](api-discussion.md) is proposed but not implemented.

## 2. The operation is an aggregation

`features` returns one Struct for one cloud, not one result for each input point.
It registers as a non-elementwise scalar aggregation.

| Polars context | Meaning |
| --- | --- |
| select | One cloud from all rows in that context |
| group_by(...).agg | One cloud and one result per group |
| with_columns(expr.over(...)) | One cloud per group; broadcast its result to the group's rows |
| with_columns(expr) without over | One global cloud; broadcast its result to every row |
| group_by_dynamic(...).agg | One cloud per Polars-defined time bin |
| rolling(...).agg | One cloud per Polars-defined rolling window |

Window sorting, boundaries, grouping keys and output row order belong to Polars.
There are no `by`, `window`, `every`, `return_dataframe` or `output_format`
parameters on `features`. Overlapping windows may repeat expensive work; the
plugin currently implements no incremental persistence algorithm.

Eager and lazy queries use the same numerical contract and output dtype. The
streaming execution engine is supported, but a cloud still requires its complete
input and native working memory. This is not an online or bounded-memory TDA
algorithm, and arbitrary batches must not be treated as independent clouds.

## 3. Results are ordinary typed data

The result is a fixed Struct of the ten fields in [the feature dictionary](features.md).
Field names, order, dtypes and null rules are identical for exact/approximate,
empty/nonempty input, and eager/lazy execution. Lazy planning must infer the dtype
without running topology computations. Do not encode results in JSON strings,
Python objects or lists whose length depends on observed topology.

Follow Polars naming: the default result name derives from the first resolved
coordinate; callers select an outer name with `.alias("topology")`. Alias does
not rename nested feature fields. `unnest` expands the Struct into numeric
columns, and ordinary Struct projection selects a subset:

```python
clouds = df.group_by("sample").agg(tda.features("x", "y").alias("topology"))
feature_table = clouds.unnest("topology")
loop_feature = clouds.select(
    "sample",
    pl.col("topology").struct.field("h1_total_persistence").alias("loop_lifetime"),
)
```

The primary API does not add feature-selection or renaming parameters. Consumers
choose their model columns using standard Polars expressions. Multiple feature
Structs may coexist under distinct aliases; their inner names overlap, so callers
must rename selected fields when combining them in a flat table.

## 4. Method is a semantic choice

`exact` and `approximate` have the same input/output contract. Approximate is
explicitly opt-in, may change numerical features, and does not mean fast mode.
Keep implementation-specific names such as sparse, callback and expanded out of
the analyst method enumeration. Never switch methods silently based on input size.

The library controls fixed presets, documented in [features](features.md). They
must be deterministic for a fixed ordered input and software environment, and
must not vary with chunk layout. Approximate sampling can depend on row order;
callers can sort through Polars when repeatability across input reorderings is
required. Cross-platform floating-point bit identity is not promised.

Stable-release policy: do not silently rename/reorder fields, change dtypes,
null meanings or approximation presets. Such changes need a documented version
boundary. Additive fields can also break exact schemas and downstream models;
treat feature-bundle membership as part of the schema contract. Store method and
package version alongside downstream datasets when reproducibility matters.

## 5. Missing and invalid data have different meanings

Null/NaN/infinite coordinates are input errors. Do not drop a point, select a
fallback method or return a partially computed feature vector. Invalid method
values are rejected when constructing the Python expression; input schema/data
errors surface from Polars planning or execution with a meaningful message.

An empty cloud is valid: counts and totals are zero and finite maximum/entropy
are null. Null here means the statistic is undefined, not that the operation
failed. A singleton is also valid. Essential intervals have their own counts and
are excluded from finite lifetime statistics. Keep these rules unchanged for
both methods; do not fill undefined statistics with zero merely to ease modeling.

## 6. Composition should not create hidden work

A call constructs a normal `pl.Expr`; it does not execute an eager computation.
Each native feature invocation computes persistence once and derives both H0/H1
summaries without a serialized intermediate diagram. Projection of one output
field does not currently select a cheaper homology algorithm.

Separate invocations or separate collections of a lazy plan are not guaranteed
to share work. For repeated downstream analyses, materialize/store the feature
Struct or use the advanced stored-diagram workflow. Individual descriptors over
stored diagrams are compatible with this design. New point-cloud convenience
functions must make repeated computation costs clear; do not assume independent
expressions share the same persistence calculation.

## 7. Keep the public surface small

`tda.features(...)` is the primary function-style expression factory. It naturally
accepts multiple coordinate columns. A custom `.tda` namespace, DataFrame wrapper,
configuration object or fit/transform estimator is not needed for this contract.
Consider an alternative surface when a demonstrated workflow benefits from it.
The current preference is scoped to this package; it is not a prohibition on
namespaces, wrappers or estimators in numerical libraries.

The existing `rips`, `betti_curve` and `lifetime_summary` exports remain compatible
advanced operations, documented separately. This document does not move or remove
them. Their mathematical tuning parameters are not templates for new knobs on
`features`. Future non-scalar descriptors need their own stable output-shape
contract; they must not silently change the existing numeric feature Struct.

## Verification and references

`tests/test_expression_contract.py` covers selector expansion, grouped/window
behavior, time bins, rolling windows, projection and invalid input shapes.
Other feature tests cover mathematical values, presets, chunks, empty clouds and
missing inputs. The selector correction also applies to the advanced `rips`
coordinate aggregation. Previously `pl.col(["x", "y"])` expanded into separate
one-dimensional analyses; `input_wildcard_expansion=True` makes it a single
multidimensional analysis as intended.

Polars documents these registration flags and expression integration in its
[plugin API](https://docs.pola.rs/api/python/stable/reference/api/polars.plugins.register_plugin_function.html)
and [expression plugin guide](https://docs.pola.rs/user-guide/plugins/expr_plugins/).
Actual behavior was checked against this project's pinned Polars version; the
live stable documentation may describe a different version.
