# Using topology features in Polars

One row is one point. Selected coordinate columns jointly describe one cloud;
a group identifies a separate cloud. A point itself does not receive an
independent topology score.

## Prepare coordinates and compute a feature table

This complete example uses integer source coordinates, casts them explicitly,
and computes one result per sample. It works with the current API:

```python
import polars as pl
import polars_tda as tda

points = pl.DataFrame(
    {
        "sample": ["small"] * 4 + ["large"] * 4,
        "x": [0, 1, 1, 0, 0, 2, 2, 0],
        "y": [0, 0, 1, 1, 0, 0, 2, 2],
    }
)
coordinates = ["x", "y"]
prepared = points.with_columns(pl.col(coordinates).cast(pl.Float64))

query = (
    prepared.lazy()
    .group_by("sample")
    .agg(
        pl.len().alias("n_points"),
        tda.features(*coordinates, method="exact").alias("topology"),
    )
)
clouds = query.collect().sort("sample")
feature_table = clouds.unnest("topology")

# Select and rename model inputs without running topology again.
model_inputs = clouds.select(
    "sample",
    "n_points",
    pl.col("topology").struct.field("h0_max_persistence").alias("merge_distance"),
    pl.col("topology").struct.field("h1_max_persistence").alias("loop_lifetime"),
)
```

`clouds` contains two rows with one Struct per row. `feature_table` contains the
sample key, point count and ten feature columns. For these squares,
`loop_lifetime` is approximately 0.414214 and 0.828427 for small and large,
respectively. Doubling coordinate units doubles lifetime-based measurements;
it does not imply a better or worse sample.

Currently coordinates must be finite, non-null Float64. Automatic conversion of
other numeric types is a [proposed improvement](api-discussion.md), not implemented
behavior. Casting large integers to Float64 can lose distinctions between points;
a successful cast does not prove exact representability. Choose suitable units
and precision before analysis. Null, NaN and infinite coordinates produce errors.
If your analysis calls for removing invalid points, filter the entire frame
before aggregation so every coordinate still refers to the same row. The plugin
does not silently filter, scale or deduplicate points.

## Select columns and choose a method

The following snippets continue the example above:

```python
# All selected columns form ONE cloud, including multi-column expressions.
one_cloud = prepared.filter(pl.col("sample") == "small").select(
    tda.features(pl.col(coordinates)).alias("topology")
)
approximate = prepared.group_by("sample").agg(
    tda.features(*coordinates, method="approximate").alias("topology")
)
```

A Python list is unpacked with `*coordinates`. Strings mean column names, not
literal coordinate values. Select coordinate columns deliberately: a numeric
selector can otherwise include IDs, timestamps encoded as numbers or labels.
The plugin does not automatically exclude such columns.

Exact and approximate return the same fields and dtypes. Approximation can
change the values, is not guaranteed to be faster, and has no per-feature error
guarantee. Its current preset can depend on point order. For reproducibility,
keep a consistent row order and record the method, package version and
preprocessing alongside the resulting table. See [presets](features.md#library-managed-presets).

## Groups, windows and row-level joins

Use Polars to define which rows form a cloud. With `group_by(...).agg(...)`, each
group produces one row. To attach a group's features to all its input rows:

```python
annotated = prepared.with_columns(
    tda.features(*coordinates).over("sample").alias("topology")
)
```

Without `.over("sample")`, `with_columns` broadcasts the result of the entire
cloud to all rows. Time bins and rolling windows use Polars' `group_by_dynamic`
and `rolling` aggregation contexts; their sorting and boundary rules still
apply. Overlapping windows can repeat expensive work. See the
[context table](expression-contract.md#2-the-operation-is-an-aggregation).

## Reuse results and interpret missing values

The example collects `query` once and derives multiple tables from `clouds`.
Collecting the same lazy query again can repeat persistence computation. Selecting
one field directly from `tda.features(...)` still computes the current full
bundle; projection is not a cheaper algorithm. Keep or save the materialized
Struct when multiple consumers need its fields.

To derive different descriptors from the same persistence diagram, use the
[stored-diagram workflow](advanced-api.md). You do not need a diagram column for
the main `features` workflow.

No finite intervals means count and total are zero, while maximum and entropy
are null. Entropy zero instead means a defined distribution concentrated in one
interval. Do not automatically replace nulls with zero unless that treatment is
appropriate for the downstream model. See [feature meanings](features.md).

The native adapter borrows compatible input buffers where possible, but type
conversion, Polars grouping, approximate layout packing, distance storage and
reduction workspaces may allocate. Streaming execution does not make a cloud an
incremental or bounded-memory computation. The [memory contract](advanced-api.md#memory-and-execution)
describes these costs separately.
