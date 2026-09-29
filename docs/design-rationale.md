# Expression design rationale

Status: design review recorded on 2026-09-29. This document explains choices and
proposals; the [expression contract](expression-contract.md) describes current
behavior. No API or kernel change is implied by a proposal here.

## Evidence from polars_ds

We inspected Python expression factories, Rust implementations and related tests
in [polars_ds_extension](https://github.com/abstractqqq/polars_ds_extension/tree/ad0c8ef5b914452ff9bdefb68c26ef3e041e86ac),
pinned to commit `ad0c8ef5b914452ff9bdefb68c26ef3e041e86ac`. This was a source
review, not an execution or performance comparison with that library.

| Observed implementation | Application to polars-tda |
| --- | --- |
| `query_knn_entropy(*features)` accepts names/expressions and registers a scalar aggregation | Multiple columns jointly form an input; Polars supplies aggregation context |
| `query_binary_metrics` returns a Struct of several metrics from shared intermediate work | Bundle descriptors that share an expensive persistence calculation |
| `typing.py` defines `Literal` aliases for methods and policies | Keep method choices typed, without exposing Rust enums |
| `query_mid_range` and several other features compose ordinary Polars expressions | A new public function need not require a new native symbol |
| The KNN path converts numeric columns and packs a row-major buffer | Numeric convenience and allocation costs must be considered separately |

Sources: [aggregation and composed expressions](https://github.com/abstractqqq/polars_ds_extension/blob/ad0c8ef5b914452ff9bdefb68c26ef3e041e86ac/python/polars_ds/exprs/ts_features.py),
[metric expression factories](https://github.com/abstractqqq/polars_ds_extension/blob/ad0c8ef5b914452ff9bdefb68c26ef3e041e86ac/python/polars_ds/exprs/metrics.py),
[shared native metrics](https://github.com/abstractqqq/polars_ds_extension/blob/ad0c8ef5b914452ff9bdefb68c26ef3e041e86ac/src/num_ext/tp_fp.rs),
[Literal aliases](https://github.com/abstractqqq/polars_ds_extension/blob/ad0c8ef5b914452ff9bdefb68c26ef3e041e86ac/python/polars_ds/typing.py),
and [numeric packing](https://github.com/abstractqqq/polars_ds_extension/blob/ad0c8ef5b914452ff9bdefb68c26ef3e041e86ac/src/utils/mod.rs).

These examples support the expression-factory and Struct design. They do not
establish a universal Polars rule about naming, the number of parameters, null
handling or whether to offer estimators. The reference library also has
[separate pipeline abstractions](https://github.com/abstractqqq/polars_ds_extension/tree/ad0c8ef5b914452ff9bdefb68c26ef3e041e86ac/python/polars_ds/pipeline).
Our smaller surface follows this project's current workflows.

## Decisions retained

- The main operation is `features(*coordinates, method="exact" | "approximate")`.
  Analysts choose a computation mode; the package manages the documented preset.
- One input row is one point. Grouping, windows, scaling and field selection use
  ordinary Polars operations, with one scalar Struct per cloud.
- One native feature invocation computes persistence once and derives the bundle
  directly. A serialized intermediate diagram is optional, not mandatory.
- Advanced diagram operations remain available and retain their existing names.
- Exact never silently switches to approximate. Approximate does not mean faster.

The single method option, ten-field bundle and function-style surface are product
choices. Plugin flags must correctly describe execution and result shape; they
do not dictate those product choices. See the
[Polars plugin API](https://docs.pola.rs/api/python/stable/reference/api/polars.plugins.register_plugin_function.html).

## Refinements proposed, not implemented

**Numeric inputs.** Float64-only is the current adapter limitation. A future
adapter can preserve borrowed Float64 buffers while converting other supported
numeric types only when needed. Before implementing this, specify accepted types,
integer precision rules, errors and conversion costs. In particular, a strict
cast alone does not establish lossless integer-to-float conversion. Null handling
and normalization remain separate decisions.

**Individual descriptors.** Shared expensive computation is the concern, not the
existence of scalar functions. A descriptor over a stored diagram can be useful
without rerunning persistence; current `lifetime_summary` and `betti_curve` already
follow that model. Avoid encouraging independent point-cloud calls for every
statistic unless their repeated-work cost is explicit or sharing is established.

**Default fields.** The ten current fields remain implemented and unchanged. Review
their usefulness and names before a stable release, distinguishing model inputs
from diagnostic counts. Fixed output shape does not prove that every field is
informative for every task. Field membership and naming changes need a documented
version boundary, including when adding fields.

## Performance boundaries

Native execution avoids a Python row loop but does not imply zero allocation.
The exact adapter borrows Float64 coordinate chunks; sparse input may require
O(n*d) packing. Polars itself may gather groups and allocate conversions, and the
kernel owns distance/graph data and reduction workspaces. The reference KNN
implementation's packing is evidence of its layout choice, not a requirement for
all plugins and not a comparative performance result.

Measure total latency, peak memory and representative group sizes as well as
copies. Do not promise common-subexpression sharing between independent plugin
calls. See [memory behavior](advanced-api.md#memory-and-execution) and
[local measurements](benchmark-notes.md).

## Adding an operation without coupling kernel development

Kernel contributors continue implementing and testing cocycle algorithms through
its own APIs. They do not need to add a Polars adapter for every new algorithm.
This repository opts into public operations when an analyst workflow needs them.

| Change | Location and responsibility |
| --- | --- |
| Option validation, expression composition, public signature | `python/polars_tda/expressions.py`; use existing Polars operations where sufficient |
| Borrowing or preparing coordinate buffers | `src/coordinates.rs`; make allocation and dtype behavior explicit |
| Native plugin registration and kernel invocation | `src/expressions.rs`; set shape flags and output dtype correctly |
| Diagram serialization, coverage and schema validation | `src/diagram.rs`; preserve stored-result compatibility |
| Homology or descriptor algorithm | The cocycle dependency, through its existing public API where available |
| User semantics and adapter verification | `docs/` and `tests/`; check numerical meaning, output shape and supported contexts |

This table describes the current small layout, not a requirement to create a new
module for each function. Split modules when their responsibilities warrant it.
Do not duplicate a kernel algorithm here to expose it. If a needed capability is
missing from the kernel's public API, evaluate that specific kernel change
separately; independence does not guarantee that every future feature needs no
kernel work.

The Git-pinned dependency is compiled by Cargo. A kernel update is explicit and
tested; it does not automatically enlarge the Python API. See
[project boundaries](../README.md#kernel-and-project-boundaries).
